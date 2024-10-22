import torch
from torch import nn
import torch.nn.functional as F



from .backbone import build_backbone          # CNN Backbone作成用
from .transformer import build_transformer    # Transformer作成用
from .matcher import build_matcher            # ハンガリアン法の作成用

from util.misc import NestedTensor, nested_tensor_from_tensor_list, is_dist_avail_and_initialized
from util.misc import get_world_size, accuracy



class DETR(nn.Module):
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss=False):
         
        super().__init__()

        ## オブジェクトクエリの数
        self.num_queries = num_queries

        ## Transformer
        self.transformer = transformer

        ## 埋め込み次元数
        hidden_dim = transformer.d_model
    
        ## クラス分類用のFFN
        self.class_embed = nn.Linear(hidden_dim, num_classes+1)

        ## BBox予測用のFFN
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        ## オブジェクトクエリの作成
        ## self.query_embed.weightを学習可能なオブジェクトクエリとして使用
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        # print("self.query_embed: ", self.query_embed)              # self.query_embed:  Embedding(100, 256)

        ## ????
        self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)

        ## CNN Backboneの作成
        self.backbone = backbone

        ## 使用するlossの判断???
        self.aux_loss = aux_loss
    

    def forward(self, samples: NestedTensor):

        ## samplesをNestedTensorに変更
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        
        ## CNN Backboneに入力して画像を特徴マップに変換
        features, pos = self.backbone(samples)
        # print("len(features): ", len(features))    # len(features):  1

        src, mask = features[-1].decompose()
        # print("src.shape: ", src.shape)     # src.shape:  torch.Size([7, 2048, 28, 36])
        # print("mask.shape: ", mask.shape)   # mask.shape:  torch.Size([7, 28, 36])

        assert mask is not None

        ## Transformerに入力
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]

        ## 形状確認
        # print("hs.shape: ", hs.shape)   # hs.shape:  torch.Size([6, 7, 100, 256])

        ## クラス分類用の出力
        outputs_class = self.class_embed(hs)

        ## 形状確認
        # print("outputs_class.shape: ", outputs_class.shape)   # outputs_class.shape:  torch.Size([6, 7, 100, 92])

        ## BBox推定用の出力
        outputs_coord = self.bbox_embed(hs).sigmoid()

        ## 形状確認
        # print("output_coord.shape: ", outputs_coord.shape)   # output_coord.shape:  torch.Size([6, 7, 100, 4])

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]}
        # print("self.aux_loss: ", self.aux_loss)     # self.aux_loss:  True
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        
        return out


    

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]



## 258
class PostProcess(nn.Module):
    """  モデルの出力をCOCO APIが期待する形状に変形する  """
    @torch.no_grad()
    def forward(self, outputs, target_size):

        raise ValueError(f"not implemented")



class SetCriterion(nn.Module):

    def __init__(self,
                 num_classes,   # クラス数（COCOなら91）
                 matcher,       # ラベルと出力をマッチさせるハンガリアン法
                 weight_dict,   # 各損失に対する重み
                 eos_coef,      # 背景に対する損失の重み
                 losses         # 損失の種類
                ):

        super().__init__()

        ## クラス数
        self.num_classes = num_classes

        ## ハンガリアン法
        self.matcher = matcher

        ## 各損失に対する重み
        self.weight_dict = weight_dict

        ## 背景に対する損失の重み
        self.eos_coef = eos_coef

        ## 損失の種類
        self.losses = losses

        ## ???
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    ## 108
    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):

        ## クラス分類損失の計算
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        # print("src_logits.shape: ", src_logits.shape)   # src_logits.shape:  torch.Size([7, 100, 92])
        # raise ValueError(f"not implemented")

        ## (バッチ内の何番目のデータか，何番目のオブジェクトクエリの出力がラベルと対応しているか)を表すタプルを格納
        idx = self._get_src_permutation_idx(indices)
        # print("idx: ", idx)
        # """
        # idx:  (tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4,
        # 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6]), tensor([80, 92,  9, 19, 31, 81,  5, 26, 31, 33, 35, 36, 46, 60, 63,  7, 37, 53,
        # 57, 62, 69, 76, 87,  8, 54, 67, 73, 81, 85, 94,  6, 14, 29, 43, 63, 84,
        #  2,  7, 17, 43]))
        # """
        # raise ValueError(f"not implemented")


        ## 形状確認（ハンガリアン法の結果の再確認）
        # print("len(indices): ", len(indices))    # len(indices):  7
        # print("indices[0]: ", indices[0])        # indices[0]:  (tensor([80, 92]), tensor([0, 1]))
        # print("indices[1]: ", indices[1])        # indices[1]:  (tensor([ 9, 19, 31, 81]), tensor([2, 1, 0, 3]))

        
        ## ラベルの獲得
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        # print("target_classes_o: ", target_classes_o)
        # """
        # target_classes_o:  tensor([82, 79, 34,  1,  1,  1,  6,  6,  6,  6,  6,  6,  6,  6,  6, 82, 47, 81,
        # 44, 47, 79, 78, 51, 84, 74, 84, 76, 72, 72, 74,  1,  1,  1,  1,  1,  4,
        # 67, 51, 59, 48], device='cuda:0')
        # """

        ## オブジェクトクエリの出力用ラベルの雛形を作成（全て背景で埋める）
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        # print("target_classes: ", target_classes)

        ## 出力に対応した位置のラベルを背景から物体のラベルに変更
        ## （print(target_classes)で表示するとわかりやすい）
        target_classes[idx] = target_classes_o
        # print("target_classes: ", target_classes)


        # for t, (_, j) in zip(targets, indices):

        #     print('t["labels"][j]: ', t["labels"][j])
        #     """
        #     t["labels"][j]:  tensor([82, 79], device='cuda:0')
        #     t["labels"][j]:  tensor([34,  1,  1,  1], device='cuda:0')
        #     t["labels"][j]:  tensor([6, 6, 6, 6, 6, 6, 6, 6, 6], device='cuda:0')
        #     t["labels"][j]:  tensor([82, 47, 81, 44, 47, 79, 78, 51], device='cuda:0')
        #     t["labels"][j]:  tensor([84, 74, 84, 76, 72, 72, 74], device='cuda:0')
        #     t["labels"][j]:  tensor([1, 1, 1, 1, 1, 4], device='cuda:0')
        #     t["labels"][j]:  tensor([67, 51, 59, 48], device='cuda:0')
        #     """


        ## 分類損失の計算
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        # print("loss_ce: ", loss_ce)     # loss_ce:  tensor(5.0847, device='cuda:0', grad_fn=<NllLoss2DBackward0>)

        ## 記録
        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
            # print("losses['class_error']: ", losses['class_error'])    # losses['class_error']:  tensor(100., device='cuda:0')

        return losses

        
    

    def loss_cardinality(self, outputs, targets, indices, num_boxes):

        pred_logits = outputs['pred_logits']
        device = pred_logits.device

        raise ValueError(f"not implemented")



    ## 193
    def _get_src_permutation_idx(self, indices):

        ## インデックスに従って予測を並び替える
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])   # バッチ内の何番目のデータかを表すインデックスを格納
        src_idx = torch.cat([src for (src, _) in indices])                                       # 何番目のオブジェクトクエリの出力がラベルに対応した出力かを格納

        # print("batch_idx: ", batch_idx)
        # """
        # batch_idx:  tensor([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 4,
        # 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6])
        # """

        # print("src_ix: ", src_idx)
        # """
        # src_ix:  tensor([80, 92,  9, 19, 31, 81,  5, 26, 31, 33, 35, 36, 46, 60, 63,  7, 37, 53,
        # 57, 62, 69, 76, 87,  8, 54, 67, 73, 81, 85, 94,  6, 14, 29, 43, 63, 84,
        #  2,  7, 17, 43])
        # """
        # raise ValueError(f"not implemented")

        return batch_idx, src_idx






    ## 205
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):

        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            # 'boxes': self.loss_boxes,
            # 'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):

        ## 形状確認
        # print("len(outputs): ", len(outputs))    # len(outputs):  3
        # print("len(targets): ", len(targets))    # len(targets):  7


        ## aux_outputsを取り除く
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        ## ハンガリアン法によって出力とラベルの一致を決定
        indices = self.matcher(outputs_without_aux, targets)

        ## 形状確認
        # print("len(indices): ", len(indices))    # len(indices):  7
        # print("indices[0]: ", indices[0])        # indices[0]:  (tensor([80, 92]), tensor([0, 1]))
        # print("indices[1]: ", indices[1])        # indices[1]:  (tensor([ 9, 19, 31, 81]), tensor([2, 1, 0, 3]))

        ## 3.1節最後の文章の正規化
        ## ボックスの数を獲得
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # print("num_boxes: ", num_boxes)     # num_boxes:  40.0


        ## 損失の計算
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        



        

        raise ValueError(f"not implemented")


## 289
class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):

        super().__init__()

        ## 層数
        self.num_layers = num_layers

        ## 隠れ層
        h = [hidden_dim] * (num_layers - 1)

        ## 線形層の作成
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
    
    def forward(self, x):

        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        
        return x


## 304
def build(args):

    """  データセットに応じてクラス数を決定  """
    num_classes = 20 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250

    """  デバイスの決定  """
    device = torch.device(args.device)

    """  CNN Backboneの定義  """
    backbone = build_backbone(args)
    
    """  Transformerの定義  """
    transformer = build_transformer(args)

    """  modelの定義  """
    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )

    """  セグメンテーションタスクのためのモデル作成  """
    if args.masks:
        # model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
        raise ValueError(f"not implemented")
    

    """  ハンガリアン法の作成  """
    matcher = build_matcher(args)

    """  損失に対する重み  """
    ## クラス分類とbboxの損失に対する重み
    weight_dict = {'loss_ce': 1, 'loss_bbox':args.bbox_loss_coef}
    
    ## GIoU損失に対する重み
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    ## セグメンテーションタスクでの損失に対する重み
    if args.masks:
        weight_dict["loss_mask"] = args.mask_loss_coef
        weight_dict["loss_dice"] = args.dice_loss_coef
    
    ## 補助損失（各層の出力を用いた損失）に対する重み
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    # print(weight_dict)


    """  損失関数の作成  """
    losses = ['labels', 'boxes', 'cardinality']
    if args.masks:
        losses += ["masks"]
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
    # print(criterion)
    criterion.to(device)
    
    ## ????
    postprocessors = {'bbox': PostProcess()}
    if args.masks:
        postprocessors['segm'] = PostProcessSegm()
        if args.dataset_file == "coco_panoptic":
            is_thing_map = {i: i <= 90 for i in range(201)}
            postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion, postprocessors
    