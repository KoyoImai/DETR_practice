import torch
from torch import nn


from .backbone import build_backbone          # CNN Backbone作成用
from .transformer import build_transformer    # Transformer作成用
from .matcher import build_matcher            # ハンガリアン法の作成用

from util.misc import NestedTensor, nested_tensor_from_tensor_list



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
        print("output_coord.shape: ", outputs_coord.shape)


        raise ValueError(f"not implemented")



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

        print(self)


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
    