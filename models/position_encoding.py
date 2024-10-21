import math
from torch import nn
import torch


from util.misc import NestedTensor


## 12
class PositionEmbeddingSine(nn.Module):

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):

        super().__init__()
        self.num_pos_feats = num_pos_feats    # Transformerの埋め込み次元数
        self.temperature = temperature        # 温度パラメータ
        self.normalize = normalize            # 正規化の有無

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale


    def forward(self, tensor_list: NestedTensor):

        ## tensor_list（データ型: NestedTensor）から属性tensors，maskを取り出す
        x = tensor_list.tensors
        mask = tensor_list.mask

        ## 形状確認
        # print("x.shape: ", x.shape)         # x.shape:  torch.Size([7, 2048, 28, 36])
        # print("mask.shape: ", mask.shape)   # mask.shape:  torch.Size([7, 28, 36])

        assert mask is not None

        ## maskを反転させる
        not_mask = ~mask

        ## 縦横に累積和を計算することで位置情報を計算する
        y_embed = not_mask.cumsum(1, dtype=torch.float32)   # 縦方向に累積和を計算
        x_embed = not_mask.cumsum(2, dtype=torch.float32)   # 横方向に累積和を計算

        ##  形状確認
        # print("y_embed.shape: ", y_embed.shape)   # y_embed.shape:  torch.Size([7, 28, 36])
        # print("x_embed.shape: ", x_embed.shape)   # x_embed.shape:  torch.Size([7, 28, 36])

        ## normalizeを行う場合
        # print("self.normalize: ", self.normalize)    # self.normalize:  True
        if self.normalize:
            eps = 1e-6

            ## 最後の行から最大値を取り出して，y_embed全体を除算し，self.scaleでスケーリングする
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale

            ## 最後の列から最大値を取り出して，x_embed全体を除算し，self.scaleでスケーリングする
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        
        ## 各次元ごとの位置情報に基づいたスケーリングファクターを作成
        # 埋め込み次元数分の要素を持った配列
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # print("dim_t.shape: ", dim_t.shape)   # dim_t.shape:  torch.Size([128])
        # print("dim_t: ", dim_t)    # dim_t:  tensor([  0.,   1.,   2.,   3., ..., 124., 125., 126., 127.], device='cuda:0')

        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        # print("dim_t.shape: ", dim_t.shape)   # dim_t.shape:  torch.Size([128])
        # print("dim_t: ", dim_t)    # dim_t:  tensor([1.0000e+00, 1.0000e+00, 1.1548e+00, 1.1548e+00, ..., 7.4989e+03, 7.4989e+03, 8.6596e+03, 8.6596e+03], device='cuda:0')
        

        ## 偶数次元に対してsin，奇数次元に対してcosを適用
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # print("pos_x.shape: ", pos_x.shape)      # pos_x.shape:  torch.Size([7, 28, 36, 128])
        # print("pos_y.shape: ", pos_y.shape)      # pos_y.shape:  torch.Size([7, 28, 36, 128])

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # print("pos_x.shape: ", pos_x.shape)      # pos_x.shape:  torch.Size([7, 28, 36, 128])
        # print("pos_y.shape: ", pos_y.shape)      # pos_y.shape:  torch.Size([7, 28, 36, 128])
        # print("torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).shpae: ",
        #       torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).shape)        # torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).shpae:  torch.Size([7, 28, 36, 64, 2])


        ## x方向とy方向の位置埋め込みを結合し，最終的な位置エンコーディングを作成
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        # print("pos.shape: ", pos.shape)     # pos.shape:  torch.Size([7, 256, 28, 36])
        # print("pos[0][0][0][0]: ", pos[0][0][0][0])    # pos[0][0][0][0]:  tensor(0.2487, device='cuda:0')
        # print("pos[0][1][0][0]: ", pos[0][1][0][0])    # pos[0][1][0][0]:  tensor(0.9686, device='cuda:0')
        # print("pos[0][2][0][0]: ", pos[0][2][0][0])    # pos[0][2][0][0]:  tensor(0.2159, device='cuda:0')
        # print("pos[1][0][0][0]: ", pos[1][0][0][0])    # pos[1][0][0][0]:  tensor(0.3247, device='cuda:0')

        # print("pos[0][0][0][0:4]: ", pos[0][0][0][0:4])    # pos[0][0][0][0:4]:  tensor([0.2487, 0.2487, 0.2487, 0.2487], device='cuda:0')
        # print("pos[0][1][0][0:4]: ", pos[0][1][0][0:4])    # pos[0][1][0][0:4]:  tensor([0.9686, 0.9686, 0.9686, 0.9686], device='cuda:0')
        # print("pos[1][0][0][0:4]: ", pos[1][0][0][0:4])    # pos[1][0][0][0:4]:  tensor([0.3247, 0.3247, 0.3247, 0.3247], device='cuda:0')

        # print("pos[0][0][0:4][0]: ", pos[0][0][0:4][0])    # 
        # print("pos[0][1][0:4][0]: ", pos[0][1][0:4][0])    # 
        # print("pos[1][0][0:4][0]: ", pos[1][0][0:4][0])    # 

        # # テンソルをリストに変換
        # tensor_list = pos.tolist()

        # # テキストファイルに書き込む
        # with open('./tensor.txt', 'w') as f:
        #     f.write(str(tensor_list))
        
        return pos




def build_position_encoding(args):

    ## Transformerの埋め込み次元数を2で割る（余りは切り捨て）
    N_steps = args.hidden_dim // 2

    ## 位置埋め込みの方法によって処理を変更
    if args.position_embedding in ('v2', 'sine'):
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        raise ValueError(f"not implemented {args.position_embedding}")
    else:
        raise ValueError(f"not supported {args.position_embedding}")
    
    return position_embedding