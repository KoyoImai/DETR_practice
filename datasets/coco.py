
from pathlib import Path

import torch
import datasets.transforms as T
import torchvision
from pycocotools import mask as coco_mask




class CocoDetection(torchvision.datasets.CocoDetection):
    
    def __init__(self, img_folder, ann_file, transforms, return_masks):

        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        
        ## CocoDetectionのgetitemメソッドによってidxに対応したimgとtargetを獲得
        img, target = super(CocoDetection, self).__getitem__(idx)

        ## idxに対応するimage_idを獲得
        image_id = self.ids[idx]

        ## targetを辞書形式に変換
        target = {'image_id': image_id, 'annotations': target}

        ## ????
        ## bboxなどのアノテーション情報を調整している
        img, target = self.prepare(img, target)

        # print("img.size: ", img.size)

        ## データにデータ拡張を加える
        if self._transforms is not None:
            img, target = self._transforms(img, target)

        # print("img.shape: ", img.shape)     # img.shape:  torch.Size([3, 512, 682])

        
        return img, target




class ConvertCocoPolysToMask(object):

    def __init__(self, return_masks=False):

        self.return_masks = return_masks

    def __call__(self, image, target):

        ## 画像の縦・横のサイズを獲得
        w, h = image.size

        ## image_idを獲得しTensor化
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])
        # print("image_id: ", image_id)        # image_id:  tensor([463309]) など

        ## targetからannotations情報を抜き出す
        anno = target["annotations"]
        # print("len(anno): ", len(anno))

        ## クラウド（群集）でないもののみを取り出す
        anno = [obj for obj in anno if 'iscrowed' not in obj or obj['iscropwd'] == 0]

        ## bboxの情報をリストとして取り出す
        boxes = [obj["bbox"] for obj in anno]   # [[bboxの左上x座標, bboxの左上y座標，幅，　高さ], ..., [bboxの左上x座標, bboxの左上y座標，幅，　高さ]]

        ## boxesをTensor化
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)

        ## boxesの内容を[[bboxの左上x座標, bboxの左上y座標，bboxの右下x座標, bboxの右下y座標], ..., [bboxの左上x座標, bboxの左上y座標，bboxの右下x座標, bboxの右下y座標]]
        boxes[:, 2:] += boxes[:, :2]

        ## bboxの座標を画像のサイズに合わせて調整
        boxes[:, 0::2].clamp_(min=0, max=w)   # x座標を最小0，最大w
        boxes[:, 1::2].clamp_(min=0, max=h)   # y座標を最小0，最大h

        ## クラス情報をリストとして取り出す
        classes = [obj["category_id"] for obj in anno]
        # print("classes: ", classes)        # classes:  [82, 79]

        ## classesをTensor化
        classes = torch.tensor(classes, dtype=torch.int64)

        ## セグメンテーションタスクの場合，セグメンテーション用のアノテーションの処理も実行
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        ## 姿勢推定用のアノテーションに関する処理 
        # print("anno and 'keypoints' in anno[0]: ", anno and 'keypoints' in anno[0])  # anno and 'keypoints' in anno[0]:  False
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        ## 無効なbboxのアノテーションを削除
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        ## target情報の更新
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        ## coco apiの形状に変更
        area = torch.tensor([obj["area"] for obj in anno])    # セグメンテーション領域内の合計ピクセル数
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])


        return image, target




def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')





def build(image_set, args):
    
    ## データセットがあるディレクトリまでのパス
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'

    ## 訓練用と検証用のデータがそれぞれあるディレクトリまでのパス
    mode = "instances"
    PATHS = {
        "train": (root / "images" / "train2017", root / "annotations" / f'{mode}_train2017.json'),
        "val": (root / "images" / "val2017", root / "annotations" / f"{mode}_val2017.json")
    }

    ## 訓練用か検証用どちらかのパス
    img_folder, ann_file = PATHS[image_set]

    ## データセットの作成
    dataset = CocoDetection(img_folder, ann_file, transforms=make_coco_transforms(image_set), return_masks=args.masks)

    return dataset