import argparse
from pathlib import Path
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, DistributedSampler
import datasets
import time




import util.misc as utils
from models import build_model                     # モデルの作成
from datasets import build_dataset                 # データセットの作成
from datasets import get_coco_api_from_dataset     # データセットの作成
# from engine import evaluate                        # 検証
from engine import train_one_epoch                 # 訓練


def get_args_parser():

    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--batch_size', default=7, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    parser.add_argument('--epochs', default=300, type=int)


    ## モデルのパラメータ
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    

    ## CNN Backbone用
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")



    ## Transformer用
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    

    # Segmentation用
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")
    

    ## 損失
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    
    ## 損失関数の重み
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    

    ## Matcher（出力とラベルを一致させる役割）
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    ## データセットのパラメータ
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str)
    parser.add_argument('--num_workers', default=2, type=int)


    ## その他いろいろ
    # 実行結果の出力先
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')


    return parser



def main(args):

    """  DDPの初期化  """
    utils.init_distributed_mode(args)

    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    
    """  デバイスの決定  """
    device = torch.device(args.device)


    """  シード値の固定  """
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    """  モデル，損失関数，（モデルの出力をCOCO-APIが期待する形状に変化する何か）の定義  """
    model, criterion, postprocessors = build_model(args)
    model.to(device)

    ## モデルのDDPの設定
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)


    """  最適化手法の定義  """
    ## モデルのパラメータ
    param_dicts = [
        ## TransformerとFFNのパラメータ
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        ## CNN Backboneのパラメータ
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    
    ## スケジューラの定義
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    """  データセットの作成  """
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set="val", args=args)
    # print("len(dataset_train): ", len(dataset_train))    # len(dataset_train):  118287
    # print("len(dataset_val): ", len(dataset_val))        # len(dataset_val):  5000

    """  Samplerの作成  """
    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)
    # print("len(sampler_train): ", len(sampler_train))   # len(sampler_train):  118287
    # print("len(sampler_val): ", len(sampler_val))       # len(sampler_val):  5000

    """  Batch Samplerの作成  """
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True
    )

    """  DataLoaderの作成  """
    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)
    # print("len(data_laoder_train): ", len(data_loader_train))   # len(data_laoder_train):  59143
    # print("len(data_loader_val): ", len(data_loader_val))       # len(data_loader_val):  2500

    """  データセットの作成 2  """
    if args.dataset_file == "coco_panoptic":
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)


    """  事前学習済みモデルの読み込み  """
    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    """  学習途中のパラメータがある場合，それを読み込む  """
    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        
        ## 学習途中のモデルの重みを読み込む
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            
            ## 学習途中のoptimizerを読み込む
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            ## 学習途中のスケジューラを読み込む
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            
            ## 学習途中のエポックを読み込む
            args.start_epoch = checkpoint['epoch'] + 1


    """  検証を行う場合  """
    if args.eval:
        raise ValueError(f"not implemented")


    """  訓練の実行  """
    print("Start Training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        
        ## DDPを使用する場合，smplerのepochを設定
        if args.distributed:
            sampler_train.set_epoch(epoch)

        ## 1epoch分の学習を実行
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch, args.clip_max_norm
        )







if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    main(args)