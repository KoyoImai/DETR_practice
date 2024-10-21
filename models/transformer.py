from torch import nn, Tensor
import torch.nn.functional as F
import copy
import torch

from typing import Optional, List





class Transformer(nn.Module):

    def __init__(self, 
                 d_model=512,                    # 埋め込み次元数
                 nhead=8,                        # ヘッド数
                 num_encoder_layers=6,           # エンコーダレイヤー数
                 num_decoder_layers=6,           # デコーダレイヤー数
                 dim_feedforward=2048,           # FFNの次元数
                 dropout=0.1,                    # ドロップアウト率
                 activation="relu",              # 活性化関数
                 normalize_before=False,         # 正規化を先にやるか後にやるか
                 return_intermediate_dec=False):

        super().__init__()


        ## Encoderの各層を作成
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        
        ## エンコーダの正規化
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        ## Encoderの作成
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        ## Decoderの各層を作成
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        
        ## Decoderの正規化
        decoder_norm = nn.LayerNorm(d_model)

        ## Decoderの作成
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)
        
        ## モデルのパラメータを初期化
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead



    ## modelのパラメータを初期化
    def _reset_parameters(self):

        for p in self.parameters():

            ## バイアスは初期化の対象外
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    
    def forward(self, src, mask, query_embed, pos_embed):

        ## 形状確認
        # print("src.shape: ", src.shape)                  # src.shape:  torch.Size([7, 256, 28, 36])
        # print("pos_embed.shape: ", pos_embed.shape)      # pos_embed.shape:  torch.Size([7, 256, 28, 36])
        # print("query_embed.shape: ", query_embed.shape)  # query_embed.shape:  torch.Size([100, 256])
        # print("mask.shape: ", mask.shape)                # mask.shape:  torch.Size([7, 28, 36])

        ### 平坦化を実行（flatten NxCxHxW to HWxNxC）
        ## srcの形状取得
        bs, c, h, w = src.shape
        
        ## 平坦化
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)

        ## 形状確認
        # print("src.shape: ", src.shape)                  # src.shape:  torch.Size([1008, 7, 256])
        # print("pos_embed.shape: ", pos_embed.shape)      # pos_embed.shape:  torch.Size([1008, 7, 256])
        # print("query_embed.shaoe: ", query_embed.shape)  # query_embed.shaoe:  torch.Size([100, 7, 256])
        # print("mask.shape: ", mask.shape)                # mask.shape:  torch.Size([7, 1008])

        # print("query_embed[0][0][0:5]: ", query_embed[0][0][0:5])   # query_embed[0][0][0:5]:  tensor([ 0.7833,  0.9219,  0.1647, -1.5888, -1.4114], device='cuda:0',
        # print("query_embed[0][1][0:5]: ", query_embed[0][1][0:5])   # query_embed[0][1][0:5]:  tensor([ 0.7833,  0.9219,  0.1647, -1.5888, -1.4114], device='cuda:0',
        # print("query_embed[1][0][0:5]: ", query_embed[1][0][0:5])   # query_embed[1][0][0:5]:  tensor([-1.1989, -0.6165, -0.1948, -1.8192, -2.1575], device='cuda:0',

        # print("pos_embed[0][0][0:5]: ", pos_embed[0][0][0:5])   # pos_embed[0][0][0:5]:  tensor([0.2487, 0.9686, 0.2159, 0.9764, 0.1874], device='cuda:0')
        # print("pos_embed[0][1][0:5]: ", pos_embed[0][1][0:5])   # pos_embed[0][1][0:5]:  tensor([0.3247, 0.9458, 0.2825, 0.9593, 0.2455], device='cuda:0')
        # print("pos_embed[1][0][0:5]: ", pos_embed[1][0][0:5])   # pos_embed[1][0][0:5]:  tensor([0.2487, 0.9686, 0.2159, 0.9764, 0.1874], device='cuda:0')

        # print("pos_embed[0][0][0]: ", pos_embed[0][0][0])   # pos_embed[0][0][0]:  tensor(0.2487, device='cuda:0')
        # print("pos_embed[1][0][0]: ", pos_embed[1][0][0])   # pos_embed[1][0][0]:  tensor(0.2487, device='cuda:0')
        # print("pos_embed[2][0][0]: ", pos_embed[2][0][0])   # pos_embed[2][0][0]:  tensor(0.2487, device='cuda:0')


        ## Transformer Decoderの出力の雛形
        tgt = torch.zeros_like(query_embed)

        ## Transformer Encoderに入力
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)

        ## Transformer Decoderに入力
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)

        ## 形状確認
        # print("memory.shape: ", memory.shape)  # memory.shape:  torch.Size([1008, 7, 256])
        # print("hs.shape: ", hs.shape)          # hs.shape:  torch.Size([6, 100, 7, 256])
        # print("memory.permute(1, 2, 0).shape: ", memory.permute(1, 2, 0).shape)    # memory.permute(1, 2, 0).shape:  torch.Size([7, 256, 1008])
        # print("memory.permute(1, 2, 0).view(bs, c, h, w).shape: ", 
        #        memory.permute(1, 2, 0).view(bs, c, h, w).shape)                    # memory.permute(1, 2, 0).view(bs, c, h, w).shape:  torch.Size([7, 256, 28, 36])
        # print("hs.transpose(1, 2).shape: ", hs.transpose(1, 2).shape)              # hs.transpose(1, 2).shape:  torch.Size([6, 7, 100, 256])

        # raise ValueError(f"not implemented")

        return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, h, w)
        


## 62
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):

        super().__init__()

        ## Encoder層を決められた数だけ複製
        self.layers = _get_clones(encoder_layer, num_layers)

        ## Encoderの層数
        self.num_layers = num_layers

        ## 正規化の順序
        self.norm = norm


    def forward(self,
                src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        output = src

        ## Encoderの各層に入力
        for layer in self.layers:
            output = layer(output,
                           src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask,
                           pos=pos)
        
        ## 正規化
        # print("self.norm: ", self.norm)   # self.norm:  None
        if self.norm is not None:
            output = self.norm(output)

        return output

        raise ValueError(f"not implemented")




 ## 86
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):

        super().__init__()

        ## Decoder層を決められた数だけ複製
        self.layers = _get_clones(decoder_layer, num_layers)

        ## Decoderの層数
        self.num_layers = num_layers

        ## 正規化の順序
        self.norm = norm

        ## ?????
        self.return_intermediate = return_intermediate


    def forward(self,
                tgt,      # Decoder出力の雛形
                memory,   # Encoder出力
                tgt_mask: Optional[Tensor] = None,      # なし
                memory_mask: Optional[Tensor] = None,   # なし
                tgt_key_padding_mask: Optional[Tensor] = None,      # なし
                memory_key_padding_mask: Optional[Tensor] = None,   # padding用のマスク
                pos: Optional[Tensor] = None,         # 位置埋め込み
                query_pos: Optional[Tensor] = None    # オブジェクトクエリ
               ):
        
        ## 確認
        # print("tgt_mask: ", tgt_mask)                            # tgt_mask:  None
        # print("memory_mask: ", memory_mask)                      # memory_mask:  None
        # print("tgt_key_padding_mask: ", tgt_key_padding_mask)    # tgt_key_padding_mask:  None
        
        ## 出力の初期化
        output = tgt

        ## 中間出力の初期化????
        intermediate = []

        ## Decoderの各層に各層に入力して出力を獲得
        for layer in self.layers:
            output = layer(output,    # 出力の雛形(query_posと同じ形状の0埋め) or 前のDecoder layerの出力
                           memory,    # Encoderの出力
                           tgt_mask,  # なし
                           memory_mask=memory_mask,  # なし
                           tgt_key_padding_mask=tgt_key_padding_mask,       # なし
                           memory_key_padding_mask=memory_key_padding_mask, # Padding用のマスク
                           pos=pos,   # 位置埋め込み
                           query_pos=query_pos # オブジェクトクエリ
                          )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            # print("torch.stack(intermediate).shape: ", torch.stack(intermediate).shape)    # torch.stack(intermediate).shape:  torch.Size([6, 100, 7, 256])
            return torch.stack(intermediate)
        
        print("output.unsqueeze(0).shape: ", output.unsqueeze(0).shape)

        raise ValueError(f"not implemented")
        
        return output.unsqueeze(0)



## 127
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048,dropout=0.1,
                 activation="relu", normalize_before=False):
        
        super().__init__()


        ## Multi-Head Attentionの作成
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        ## FFNの作成
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        ## 正規化&ドロップアウトの作成
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        ## 活性化間数の作成
        self.activation = _get_activation(activation)

        ## 正規化の順番
        self.normalize_before = normalize_before

    
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        
        ## 形状確認
        # print("tensor.shpae: ", tensor.shape)  # tensor.shpae:  torch.Size([1008, 7, 256])
        # print("pos.shpae: ", pos.shape)        # pos.shpae:  torch.Size([1008, 7, 256])

        # print("pos[0][0][0:5]: ", pos[0][0][0:5])   # pos[0][0][0:5]:  tensor([0.2487, 0.9686, 0.2159, 0.9764, 0.1874], device='cuda:0')
        # print("pos[1][0][0:5]: ", pos[1][0][0:5])   # pos[1][0][0:5]:  tensor([0.2487, 0.9686, 0.2159, 0.9764, 0.1874], device='cuda:0')
        # print("pos[2][0][0:5]: ", pos[2][0][0:5])   # pos[2][0][0:5]:  tensor([0.2487, 0.9686, 0.2159, 0.9764, 0.1874], device='cuda:0')
        # print(ghjk)
        
        return tensor if pos is None else tensor + pos

    
    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        

        # print("src_key_padding_mask.shape: ", src_key_padding_mask.shape)   # src_key_padding_mask.shape:  torch.Size([7, 1008])
        
        ## queryとkeyの位置埋め込み
        q = k = self.with_pos_embed(src, pos)

        ## 形状確認
        # print("q.shape: ", q.shape)   # q.shape:  torch.Size([1008, 7, 256])
        # print("k.shape: ", k.shape)   # k.shape:  torch.Size([1008, 7, 256])

        ## Self-Attentionの計算
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        # attn_map = self.self_attn(q, k, value=src, attn_mask=src_mask,
        #                           key_padding_mask=src_key_padding_mask)[1]
        # print("src2.shape: ", src2.shape)           # src2.shape:  torch.Size([1008, 7, 256])
        # print("attn_map.shape: ", attn_map.shape)   # attn_map.shape:  torch.Size([7, 1008, 1008])

        ## Add
        src = src + self.dropout1(src2)

        ## Norm
        src = self.norm1(src2)

        ## FFN
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))

        ## Add
        src = src + self.dropout2(src2)

        ## Norm
        src = self.norm2(src)


        ## 形状確認
        # print("src.shape: ", src.shape)     # src.shape:  torch.Size([1008, 7, 256])

        return src

        raise ValueError(f"not implemented")




    
    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


## 187
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        
        super().__init__()

        ## Self-Attentionの作成
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        ## Cross-Attentionの作成
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        ## FFNの作成
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        ## 正規化&ドロップアウトの作成
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        ## 活性化間数の作成
        self.activation = _get_activation(activation)

        ## 正規化の順番
        self.normalize_before = normalize_before


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,  
                     tgt,       # Decoder出力の雛形 or 前Decoder layerの出力
                     memory,    # Encoderの出力
                     tgt_mask: Optional[Tensor] = None,     # なし
                     memory_mask: Optional[Tensor] = None,  # なし
                     tgt_key_padding_mask: Optional[Tensor] = None,     # なし
                     memory_key_padding_mask: Optional[Tensor] = None,  # Padding用のマスク
                     pos: Optional[Tensor] = None,        # 位置埋め込み
                     query_pos: Optional[Tensor] = None   # オブジェクトクエリ
                    ):
        
        ## 形状確認
        # print("tgt.shape: ", tgt.shape)               # tgt.shape:  torch.Size([100, 7, 256])
        # print("query_pos.shape: ", query_pos.shape)   # query_pos.shape:  torch.Size([100, 7, 256])
        
        ## 位置埋め込み
        ## (0埋めされたtgtか前Decoder Layerの出力tgtにオブジェクトクエリquery_posを加算)
        q = k = self.with_pos_embed(tgt, query_pos)

        ## Self-Attention
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        
        ## Add
        tgt = tgt + self.dropout1(tgt2)

        ## Norm
        tgt = self.norm1(tgt)

        ## Cross-Attention
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),   # Decoder内のSelf-Attentionの出力にオブジェクトクエリの値を加算
                                   key=self.with_pos_embed(memory, pos),        # Encoderの最終出力に位置埋め込みを加算
                                   value=memory,                                # Encoderの最終出力
                                   attn_mask=memory_mask,                       # なし
                                   key_padding_mask=memory_key_padding_mask     # Padding用のマスク
                                  )[0]
        
        ## Add
        tgt = tgt + self.dropout2(tgt2)

        ## Norm
        tgt = self.norm2(tgt)

        ## FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        ## Add
        tgt = tgt + self.dropout3(tgt)

        ## Norm
        tgt = self.norm3(tgt)

        ## 形状確認
        # print("tgt.shape: ", tgt.shape)   # tgt.shape:  torch.Size([100, 7, 256])

        return tgt

    
    def forward(self,
                tgt,     # 前のDecoder layerの出力 or 出力の雛形
                memory,  # Encoderの出力
                tgt_mask: Optional[Tensor] = None,     # なし
                memory_mask: Optional[Tensor] = None,  # なし
                tgt_key_padding_mask: Optional[Tensor] = None,      # なし
                memory_key_padding_mask: Optional[Tensor] = None,   # Padding用のマスク
                pos: Optional[Tensor] = None,         # 位置埋め込み
                query_pos: Optional[Tensor] = None    # オブジェクトクエリ
               ):
        
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        
        raise ValueError(f"not implemented")





def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,               # 埋め込み次元数
        dropout=args.dropout,                  # ドロップアウト率
        nhead=args.nheads,                     # ヘッド数
        dim_feedforward=args.dim_feedforward,  # FFNの次元数
        num_encoder_layers=args.enc_layers,    # エンコーダの数
        num_decoder_layers=args.dec_layers,    # デコーダの数
        normalize_before=args.pre_norm,        # 正規化を先にやるか後にやるか
        return_intermediate_dec=True,          # ????
    )




## 272
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])



# 289
def _get_activation(activation):

    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")