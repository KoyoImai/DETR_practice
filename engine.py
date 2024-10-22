import torch
from typing import Iterable



import util.misc as utils






def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_laoder: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0):
    
    ## modelとcriterionをtrainモードに変更
    model.train()
    criterion.train()

    ## loggerの設定
    metric_logger = utils.MetricLogger(delimiter=" ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)

    ## 表示頻度の設定
    print_freq = 10

    for samples, targets in metric_logger.log_every(data_laoder, print_freq, header):

        """
        メモ帳:
            dataloaderからデータを取り出す時，misc.py内のcollate_fnメソッドが呼び出され，バッチとしての形状が整えられる．
            この時，バッチ内の画像の最大サイズに合わせて，そのほかの画像がパッディングされる．
            ついでに，NestedTensorに変更される
        """

        # print("samples.tensors.shape: ", samples.tensors.shape)      # samples.tensors.shape:  torch.Size([7, 3, 894, 1151])
        # print("targets[0]['boxes']: ", targets[0]['boxes'])          # targets[0]['boxes']:  tensor([[0.0644, 0.5086, 0.1199, 0.9641], [0.7721, 0.7326, 0.4138, 0.5079]])
        # print("targets[0]['labels']: ", targets[0]['labels'])        # targets[0]['labels']:  tensor([82, 79])
        # print("targets[0]['image_id']: ", targets[0]['image_id'])    # targets[0]['image_id']:  tensor([463309])
        # print("targets[0]['area']: ", targets[0]['area'])            # targets[0]['area']:  tensor([ 78839.1953, 136313.7969])
        # print("targets[0]['iscrowd']: ", targets[0]['iscrowd'])      # targets[0]['iscrowd']:  tensor([0, 0])
        # print("targets[0]['orig_size']: ", targets[0]['orig_size'])  # targets[0]['orig_size']:  tensor([427, 640])
        # print("targets[0]['size']: ", targets[0]['size'])            # targets[0]['size']:  tensor([ 768, 1151])


        ## samplesとtargetsをgpuに配置
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]


        ## modelにデータを入力
        outputs = model(samples)

        ## 損失の計算
        loss_dict = criterion(outputs, targets)

        raise ValueError(f"not implemented")

