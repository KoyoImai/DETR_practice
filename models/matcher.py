from torch import nn


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





def build_matcher(args):
    return HungarianMatcher(cost_class=args.set_cost_class, cost_bbox=args.set_cost_bbox, cost_giou=args.set_cost_giou)