import torch
from torch import nn
import torchvision
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import torch.nn.functional as F


from .position_encoding import build_position_encoding

from util.misc import NestedTensor, is_main_process


class FrozenBatchNorm2d(torch.nn.Module):

    def __init__(self, n):

        super(FrozenBatchNorm2d, self).__init__()
        
        # register_bufferを使用することで，学習による最適化の対象になる
        # （ただし，ここでは全て凍結して学習中に更新することはしない）
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))


    def forward(self, x):

        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias




class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()

        ## backboneの形状を確認
        # print(backbone)

        ## CNN Backboneを学習しない場合　or 特定のパラメータでない場合，requires_grad_をFalseに変更
        ## (layer1内のconv1とかはrequires_grad_をFalseに変更)
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        
        ## セグメンテーションタスクの場合，各layerの出力を返す
        ## 物体検出タスクの場合，layer4の出力のみを返す
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}

        ## 指定したサブモジュールから中間出力を取得するためのモジュール
        ## （https://github.com/sebamenabar/Pytorch-IntermediateLayerGetter）
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

        ## チャネル数
        self.num_channels = num_channels


    def forward(self, tensor_list: NestedTensor):

        ## CNN Backboneの各層の出力を獲得
        xs = self.body(tensor_list.tensors)

        ## outの初期化
        out: Dict[str, NestedTensor] = {}
        print("out: ", out)    # out:  {}

        ## CNN Backboneの各層の出力毎に処理を実行（物体検出の場合，1層分の出力しか使用しないため処理も1回のみ）
        ## paddingを確認するためのmaskの形状を特徴マップに応じて変更
        for name, x in xs.items():

            ## 形状確認
            # print("x.shape: ", x.shape)   # x.shape:  torch.Size([7, 2048, 28, 36])

            m = tensor_list.mask
            assert m is not None
            # print("m.shape: ", m.shape)   # m.shape:  torch.Size([7, 894, 1151])
            # print("m[None].shape: ", m[None].shape)   # m[None].shape:  torch.Size([1, 7, 894, 1151])
            # print("name: ", name)         # name:  0
            
            # print("x.shape[-2:]: ", x.shape[-2:])    # x.shape[-2:]:  torch.Size([28, 36])
            # print("m[None].float().shape: ", m[None].float().shape)   # m[None].float().shape:  torch.Size([1, 7, 894, 1151])
            # print("F.interpolate(m[None].float(), size=x.shape[-2:]).shape: ",
            #       F.interpolate(m[None].float(), size=x.shape[-2:]).shape)       # F.interpolate(m[None].float(), size=x.shape[-2:]).shape:  torch.Size([1, 7, 28, 36])
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            # print("mask.shape: ", mask.shape)     # mask.shape:  torch.Size([7, 28, 36])

            # print("x.shape: ", x.shape)         # x.shape:  torch.Size([7, 2048, 28, 36])
            # print("mask.shape: ", mask.shape)   # mask.shape:  torch.Size([7, 28, 36])

            out[name] = NestedTensor(x, mask)

        # raise ValueError(f"not implemented")
        
        return out


        



        


class Backbone(BackboneBase):

    def __init__(self, name: str,
                 train_backbone: bool,          # CNN Backboneを学習するかの確認
                 return_interm_layers: bool,    # セグメンテーションタスクかの確認
                 dilation: bool                 # args.dilationによって，DETRの論文にある"DC5"に変更可能
                ):
        
        ## CNN Backboneの作成
        ## （name=args.backboneでdefaultはResNet50）
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d
        )
        # print("backbone: ", backbone)

        ## チャネル数の決定
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048

        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):

    def __init__(self, backbone, position_embedding):

        super().__init__(backbone, position_embedding)

        ## 中身の確認
        # print(self[0])   # (ResNet50の構造)
        # print(self[1])   # PositionEmbeddingSine()


    def forward(self, tensor_list: NestedTensor):

        ## CNN Backboneに画像（+padding確認用のマスク）を入力
        xs = self[0](tensor_list)

        ## 
        out: List[NestedTensor] = []

        ## posの初期化
        pos = []

        for name, x in xs.items():

            out.append(x)

            ## 位置埋め込み
            pos.append(self[1](x).to(x.tensors.dtype))

        # raise ValueError(f"not implemented")
        
        return out, pos





def build_backbone(args):

    ## 位置埋め込みの作成
    position_embedding = build_position_encoding(args)

    ## CNN backboneを学習するかの確認
    train_backbone = args.lr_backbone > 0

    ## セグメンテーションタスクかの確認
    return_interm_layers = args.masks

    ## CNN Backboneの作成
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)

    ## CNN BackboneとPosition Enbeddingの連結
    model = Joiner(backbone, position_embedding)

    ## modelのチャネル数を定義
    model.num_channels = backbone.num_channels

    return model