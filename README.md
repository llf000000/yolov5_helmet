## train.py

### 代码

```python
def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default=ROOT / 'yolov5s.pt', help='initial weights path')
    parser.add_argument('--cfg', type=str, default='', help='model.yaml path')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--hyp', type=str, default=ROOT / 'data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')
    parser.add_argument('--epochs', type=int, default=100, help='total training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='total batch size for all GPUs, -1 for autobatch')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--rect', action='store_true', help='rectangular training')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--nosave', action='store_true', help='only save final checkpoint')
    parser.add_argument('--noval', action='store_true', help='only validate final epoch')
    parser.add_argument('--noautoanchor', action='store_true', help='disable AutoAnchor')
    parser.add_argument('--noplots', action='store_true', help='save no plot files')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--bucket', type=str, default='', help='gsutil bucket')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='image --cache ram/disk')
    parser.add_argument('--image-weights', action='store_true', help='use weighted image selection for training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--multi-scale', action='store_true', help='vary img-size +/- 50%%')
    parser.add_argument('--single-cls', action='store_true', help='train multi-class data as single-class')
    parser.add_argument('--optimizer', type=str, choices=['SGD', 'Adam', 'AdamW'], default='SGD', help='optimizer')
    parser.add_argument('--sync-bn', action='store_true', help='use SyncBatchNorm, only available in DDP mode')
    parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)') # workers设置进程个数，最好设置为0
    parser.add_argument('--project', default=ROOT / 'runs/train', help='save to project/name')
    parser.add_argument('--name', default='exp', help='save to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--quad', action='store_true', help='quad dataloader')
    parser.add_argument('--cos-lr', action='store_true', help='cosine LR scheduler')
    parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing epsilon')
    parser.add_argument('--patience', type=int, default=100, help='EarlyStopping patience (epochs without improvement)')
    parser.add_argument('--freeze', nargs='+', type=int, default=[0], help='Freeze layers: backbone=10, first3=0 1 2')
    parser.add_argument('--save-period', type=int, default=-1, help='Save checkpoint every x epochs (disabled if < 1)')
    parser.add_argument('--seed', type=int, default=0, help='Global training seed')
    parser.add_argument('--local_rank', type=int, default=-1, help='Automatic DDP Multi-GPU argument, do not modify')

    # Logger arguments
    parser.add_argument('--entity', default=None, help='Entity')
    parser.add_argument('--upload_dataset', nargs='?', const=True, default=False, help='Upload data, "val" option')
    parser.add_argument('--bbox_interval', type=int, default=-1, help='Set bounding-box image logging interval')
    parser.add_argument('--artifact_alias', type=str, default='latest', help='Version of dataset artifact to use')

    return parser.parse_known_args()[0] if known else parser.parse_args()
```

![image-20231103111302842](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103111302842.png)

best.py和last.py是保存的一些训练模型数据。

best.py是训练最好的模型数据

last.py是最新的模型数据。

## detect.py

```python
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold 只有当置信度大于0.25的时候才相信是一个目标
        iou_thres=0.45,  # NMS IOU threshold 预选框与预选框之间的重叠比例超过0.45为同一个框。当=0的时候，表示只要不重叠，就是一个框
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results 实时显示检测结果
        save_txt=False,  # save results to *.txt 保存到文本文件中
        save_csv=False,  # save results in CSV format
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3 指示种类出现
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name 保存位置
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment 是否将结果数据继续保存在同一个文件夹里
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
):
```

训练好的模型放在了autodl-tmp/yolov5-5.0/runs/train/exp6/weights/best.pt

点击复制best.pt文件并粘贴到detect.py文件的同一目录。

在detect.py文件里修改权重：

将default设置为best.pt

```python
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
```



## 云端训练文件

ssh -p 47980 root@connect.neimeng.seetacloud.com

### 解压压缩包

```
!unzip /content/yolov5-5.0.zip -d /content/yolov5
```

![image-20231102113244529](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231102113244529.png)

### 删除文件

```
!rm -rf /content/yolov5/ MACOSX
```

 ![image-20231102113423583](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231102113423583.png)

### 进入当前文件夹

```
%cd /content/yolov5/yolov5-5.0
```

![image-20231102113511391](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231102113511391.png)

### 下载requirement

```
!pip install -r requirements.txt
```

![image-20231102113542110](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231102113542110.png)

### 加载tensorboard

ext是tian'jia

```
%load_ext tensorboard
```

![image-20231102113651748](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231102113651748.png)

```cmd
%load_ext tensorboard
```

### 重新加载tensorboard

![image-20231102113718298](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231102113718298.png)

```
%reload_ext tensorboard
```

### 启动tensorboard

```
%tensorboard --logdir=runs/train
```

### 运行python程序

启动矩阵推理

```
！python train.py --rect
```

## COCO

```
# Classes
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
```

## 自制数据集及训练

> 标注
>
> 自己获得数据集（手动）-人工标注
>
> 自己获得数据集-半人工标注
>
> 仿真数据集（GAN，数字图像处理的方式）

### Tutorials

![image-20231103121824355](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103121824355.png)

### 数据集

### 制作自己的数据集

网址：

```
https://www.makesense.ai/
```

#### makesense.ai

进入网址：

![image-20231103122715654](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103122715654.png)

上传数据，启动目标检测

![image-20231103123230921](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103123230921.png)

预定义类别：

![image-20231103123422927](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103123422927.png)

启动时添加类别：

![image-20231103124021375](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103124021375.png)

AI：

![image-20231103124643459](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103124643459.png)

export：

![image-20231103124320009](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103124320009.png)

![image-20231103124745441](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103124745441.png)

#### 1.1 Create dataset.yaml

```python
1.1 Create dataset.yaml
COCO128 is an example small tutorial dataset composed of the first 128 images in COCO train2017. These same 128 images are used for both training and validation to verify our training pipeline is capable of overfitting. data/coco128.yaml, shown below, is the dataset config file that defines 1) the dataset root directory path and relative paths to train / val / test image directories (or *.txt files with image paths) and 2) a class names dictionary:

# download command/URL (optional) 如果数据集对应路径不存在，则通过参考链接自动下载。
download: https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip
    
# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/coco128  # dataset root dir
train: images/train2017  # train images (relative to 'path') 128 images
val: images/train2017  # val images (relative to 'path') 128 images
test:  # test images (optional)

# Classes (80 COCO classes) 80个类别
names:
  0: person
  1: bicycle
  2: car
  ...
  77: teddy bear
  78: hair drier
  79: toothbrush
```

coco128数据集文件夹在和yolov5-5.0文件夹的同一目录

coco128.yaml文件在yolov5-5.0的文件夹里面的data文件夹里，

以下是直观的文件组织目录：

**数据集：**

![image-20231106114343114](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231106114343114.png)

**yaml对应数据集的路径：**

![image-20231106111329810](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231106111329810.png)

![image-20231106111250843](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231106111250843.png)

![image-20231106111348186](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231106111348186.png)

##### 修改data路径：

```python
 parser.add_argument('--data', type=str, default='data/coco128.yaml', help='data.yaml path')
```

**coco128.yaml**

autodl-tmp/yolov5-5.0/data/coco128.yaml

![image-20231106111735062](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231106111735062.png)

**train.py**

autodl-tmp/yolov5-5.0/train.py

**--data：**

![image-20231106111606481](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231106111606481.png)

**--weights：**

训练完之后就修改weights的参数（yolov5.pt->best.pt）：

best.pt的相对位置：autodl-tmp/yolov5-5.0/runs/train/exp7/weights/best.pt

![image-20231106113124106](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231106113124106.png)

![image-20231106113455712](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231106113455712.png)

##### yaml

在yolov5目标检测项目里，yaml文件用来当做中介，是模型训练时的data数据集，然后通过键值对映射到真正的数据集，训练集，种类数量，种类名称。

YAML（YAML Ain't Markup Language）是一种人类可读、用于数据序列化的格式。它采用了类似于其他编程语言中的键值对的结构。YAML文件通常用于配置文件和数据交换，具有简洁、易读、易写的特点。

YAML文件使用缩进和空格来表示层次结构，而不是像XML或JSON那样使用标签或大括号。这使得YAML文件更加清晰、易于阅读和编辑。

下面是一个简单的YAML示例：

```yaml
yamlCopy Code# 注释以 # 开头
name: John Smith
age: 30
email: john@example.com
address:
  street: 123 Main St
  city: New York
  country: USA
```

在上面的示例中，`name`，`age`，`email`和`address`都是键，它们对应的值可以是字符串、数字、布尔值或其他复杂类型。`address`键下面有一个嵌套的层次结构，使用缩进来表示。

YAML还支持列表和复杂的数据结构，例如包含多个对象的列表或嵌套的字典。此外，YAML还可以使用特殊标记来表示日期、正则表达式等特殊类型。

总而言之，YAML是一种用于配置文件和数据序列化的格式，它采用简洁、易读、易写的结构，并使用缩进和空格来表示层次结构。

训练的结果：



#### 1.2 Create Labels

> After using an annotation tool to label your images, export your labels to **YOLO format**, with one `*.txt` file per image (if no objects in image, no `*.txt` file is required). The `*.txt` file specifications are:
>
> - One row per object
> - Each row is `class x_center y_center width height` format.
> - Box coordinates must be in **normalized xywh** format (from 0 - 1). If your boxes are in pixels, divide `x_center` and `width` by image width, and `y_center` and `height` by image height.
> - Class numbers are zero-indexed (start from 0).

> ### txt文件的解释
>
> 2  0.111111  0.22222  0.333333  0.444444
>
> 第一个：类别
>
> 第二个：x的中心
>
> 第三个：y的中心
>
> 第四个：宽度
>
> 第五个：高度
>
> 后面四位数字是做了归一化的（0到1）

#### 1.3 Organize Directories

Organize your train and val images and labels according to the example below. YOLOv5 assumes `/coco128` is inside a `/datasets` directory **next to** the `/yolov5` directory. **YOLOv5 locates labels automatically for each image** by replacing the last instance of `/images/` in each image path with `/labels/`. For example:

```
../datasets/coco128/images/im0.jpg  # image
../datasets/coco128/labels/im0.txt  # label
```

![img](https://user-images.githubusercontent.com/26833433/134436012-65111ad1-9541-4853-81a6-f19a3468b75f.png)

![image-20231103125609380](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103125609380.png)

#### 在原来官方yaml数据集的基础上修改成自己的数据集（mydata.yaml）

修改train文件夹的路径：mydata/images/train

修改val文件夹的路径：

![image-20231103130339129](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103130339129.png)

修改类别数量

修改具体累呗

在train.py文件里修改路径：

![image-20231103130649543](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103130649543.png)

在detect.py文件里修改路径：

训练的模型、测试的数据

![image-20231103130441570](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103130441570.png)

#### 总结

只改动三个大地方：

- 复制粘贴一个新的coco128.yaml文件的内容，文件命名为mydata.yaml：

  - 路径位置：autodl-tmp/yolov5-5.0/data/mydata.yaml
  - 修改四处地方

  ![image-20231106120317574](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231106120317574.png)

- 创建自己的数据集，命名为mydata：

  - 组织mydata文件夹的结构：

  ![image-20231103125609380](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103125609380.png)

- 主要是在train.py文件的main函数里改动两个：

1、weight权重文件的路径

2、yaml数据集文件的路径

![image-20231106115848089](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231106115848089.png)

## exclude无关数据集

防止花时间去检索没必要的数据集

![image-20231103130856810](C:\Users\10596\AppData\Roaming\Typora\typora-user-images\image-20231103130856810.png)
