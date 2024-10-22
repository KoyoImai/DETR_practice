from torch import nn
import torch
from scipy.optimize import linear_sum_assignment

from util.box_ops import box_cxcywh_to_xyxy, generalized_box_iou


class HungarianMatcher(nn.Module):

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):

        """
        cost_class: Class coefficient in the matching cost
        cost_bbox:  L1 box coefficient in the matching cost
        cost_giou:  giou box coefficient in the matching cost

        """
        super().__init__()

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    ## 理解放棄
    @torch.no_grad()
    def forward(self, outputs, targets):

        ## 形状確認
        # print("outputs['pred_logits'].shape: ", outputs['pred_logits'].shape)   # outputs['pred_logits'].shape:  torch.Size([7, 100, 92])
        # print("outputs['pred_boxes].shape: ", outputs['pred_boxes'].shape)      # outputs['pred_boxes].shape:  torch.Size([7, 100, 4])
        # print("len(targets): ", len(targets))                                   # len(targets):  7
        # print("targets['labels'][0].shape: ", targets[0]['labels'].shape)       # targets['labels'][0].shape:  torch.Size([2])
        # print("targets['boxes'][0].shape: ", targets[0]['boxes'].shape)         # targets['boxes'][0].shape:  torch.Size([2, 4])

        ## バッチサイズと出力クエリの数を獲得
        bs, num_queries = outputs['pred_logits'].shape[:2]

        ## コスト行列を計算するために平坦化
        out_prob = outputs['pred_logits'].flatten(0, 1).softmax(-1)   # [バッチサイズ*クエリ数, クラス数]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)                # [バッチサイズ*クエリ数, 4]

        ## 形状確認
        # print("out_prob.shape: ", out_prob.shape)     # out_prob.shape:  torch.Size([700, 92])
        # print("out_bbox.shape: ", out_bbox.shape)     # out_bbox.shape:  torch.Size([700, 4])

        ## 
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
        # print("tgt_ids.shape: ", tgt_ids.shape)                # tgt_ids.shape:  torch.Size([40])
        # print("tgt_bbox.shape: ", tgt_bbox.shape)              # tgt_bbox.shape:  torch.Size([40, 4])

        ## 分類コストを計算する
        ## (バッチ内の全てのクエリの出力から，ラベルに対応するものだけ取り出してコストを計算)
        cost_class = -out_prob[:, tgt_ids]
        # print("cost_class.shape: ", cost_class.shape)    # cost_class.shape:  torch.Size([700, 40])

        ## bboxコストを計算
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        # print("cost_bbox.shape: ", cost_bbox.shape)         #cost_bbox.shape:  torch.Size([700, 40])

        ## giouコストを計算
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))
        # print("cost_giou.shape: ", cost_giou.shape)         # cost_giou.shape:  torch.Size([700, 40])

        ## 最終的なコストを計算
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        # print("C.shape: ", C.shape)            # C.shape:  torch.Size([700, 40])
        C = C.view(bs, num_queries, -1).cpu()
        # print("C.shape: ", C.shape)            # C.shape:  torch.Size([7, 100, 40])

        ## コストCをもとに，最適な割り当てを決定
        sizes = [len(v["boxes"]) for v in targets]
        # print("len(C.split(sizes, -1)): ", len(C.split(sizes, -1)))           # len(C.split(sizes, -1)):  7
        # print("C.split(sizes, -1)[0].shape: ", C.split(sizes, -1)[0].shape)   # C.split(sizes, -1)[0].shape:  torch.Size([7, 100, 2])
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        # print("sizes: ", sizes)             # sizes:  [2, 4, 9, 8, 7, 6, 4]
        # print("len(sizes): ", len(sizes))   # len(sizes):  7

        # for i, c in enumerate(C.split(sizes, -1)):
            
        #     # print("c.shape: ", c.shape)
        #     """
        #     c.shape:  torch.Size([7, 100, 2])
        #     c.shape:  torch.Size([7, 100, 4])
        #     c.shape:  torch.Size([7, 100, 9])
        #     c.shape:  torch.Size([7, 100, 8])
        #     c.shape:  torch.Size([7, 100, 7])
        #     c.shape:  torch.Size([7, 100, 6])
        #     c.shape:  torch.Size([7, 100, 4])
        #     """
            
        #     # print("c[i].shape: ", c[i].shape)
        #     """
        #     c[i].shape:  torch.Size([100, 2])
        #     c[i].shape:  torch.Size([100, 4])
        #     c[i].shape:  torch.Size([100, 9])
        #     c[i].shape:  torch.Size([100, 8])
        #     c[i].shape:  torch.Size([100, 7])
        #     c[i].shape:  torch.Size([100, 6])
        #     c[i].shape:  torch.Size([100, 4])
        #     """
            
        #     #print("linear_sum_assignment(c[i]): ", linear_sum_assignment(c[i]))
        #     """
        #     linear_sum_assignment(c[i]):  (array([80, 92]), array([0, 1]))   ## 80番目のオブジェクトクエリと0番目のラベル，92番目のオブジェクトクエリと1番目のラベルの組み合わせが最適
        #     linear_sum_assignment(c[i]):  (array([ 9, 19, 31, 81]), array([2, 1, 0, 3]))
        #     linear_sum_assignment(c[i]):  (array([ 5, 26, 31, 33, 35, 36, 46, 60, 63]), array([6, 0, 3, 7, 4, 8, 1, 5, 2]))
        #     linear_sum_assignment(c[i]):  (array([ 7, 37, 53, 57, 62, 69, 76, 87]), array([1, 2, 5, 0, 7, 4, 6, 3]))
        #     linear_sum_assignment(c[i]):  (array([ 8, 54, 67, 73, 81, 85, 94]), array([6, 4, 5, 3, 0, 1, 2]))
        #     linear_sum_assignment(c[i]):  (array([ 6, 14, 29, 43, 63, 84]), array([1, 3, 5, 4, 2, 0]))
        #     linear_sum_assignment(c[i]):  (array([ 2,  7, 17, 43]), array([3, 2, 0, 1]))
        #     """
        # raise ValueError(f"not implemented")


        ## indicesの内容をTensor化して返す
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]


        raise ValueError(f"not implemented")






def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)