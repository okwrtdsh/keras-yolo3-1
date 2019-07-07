## 【必須項目】ソースコード

スコアを再現できるソースコード一式を、ファイルストレージサービス（例えばGoogle Drive, OneDrive, Dropbox, GitHub, GitHub Gistなど）にアップロードし、共有可能なURLをこちらに記入してください。
https://drive.google.com/open?id=17R3xF6VxM-kMNdiJADUX6uakJK94jrlK
https://github.com/okwrtdsh/yolo3
https://github.com/okwrtdsh/anaconda3
https://hub.docker.com/r/okwrtdsh/anaconda3/

## 【必須項目】解答手順

ソースコードの動作環境や利用方法などの解答手順をこちらに記載してください。

### docker-compose.ymlを作成

```
$ cat docker-compose.yml
version: '3'
services:
  jupyter:
    # 環境に合わせて
    image: okwrtdsh/anaconda3:keras-10.0-cudnn7
    #image: okwrtdsh/anaconda3:keras-9.2-cudnn7
    #image: okwrtdsh/anaconda3:keras-9.1-cudnn7
    ports:
      - 8888:8888
    volumes:
      - .:/src/notebooks
      - //datasets:/src/notebooks/datasets
      - /etc/localtime:/etc/localtime:ro
    # - .jupyter:/root/.jupyter
```

### dockerを起動

```
$ docker-compose up -d
```

### レポジトリをclone

```
$ git clone https://github.com/okwrtdsh/yolo3.git
```

### 以下のようなディレクトリ構成にする

```
$ tree
.
├── docker-compose.yml
├── fuji.zip
├── input
│   ├── sample_submission.csv
│   ├── test_images
│   ├── train_annotations # あとで生成する
│   ├── train_annotations_gamma25 # あとで生成する
│   ├── train_annotations_gamma4 # あとで生成する
│   ├── train_anns
│   ├── train_images
│   ├── train_images_gamma25 # あとで生成する
│   └── train_images_gamma4 # あとで生成する
└── yolo3 # git clone したディレクトリ
```

### docker内に入る
```
$ docker exec -it CONTAINER_ID bash
$ cd /src/notebooks/yolo3
# ガンマ補正
$ python convert_gamma.py -i ../input/train_images -o ../input/train_images_gamma4 -g 4
$ python convert_gamma.py -i ../input/train_images -o ../input/train_images_gamma25 -g 0.25
# voc formatのanotation fileを生成
$ python convert_fujifilm_to_voc_format.py \
    --ano_only \
    -i ../input/train_images \
    -a ../input/train_annotations
$ python convert_fujifilm_to_voc_format.py \
    --ano_only \
    -i ../input/train_images_gamma4 \
    -a ../input/train_annotations_gamma4
$ python convert_fujifilm_to_voc_format.py \
    --ano_only \
    -i ../input/train_images_gamma25 \
    -a ../input/train_annotations_gamma25
# 学習する場合
$ python train_fujifilm2nd.py -c config_fujifilm2nd_gamma_multi.json
# 推定する場合
# google driveからfujifilm2nd_gamma_multi.h5をダウンロードし、yolo3のディレクトリに保存
# box推定
$ python predict_fujifilm2nd.py \
    -c ./config_fujifilm2nd_gamma_multi.json \
    -i ../input/test_images/ \
    -o ../input/test_annotations_output_gamma_multi/ \
    --ano_only
# 回答
$ python predict_from_box.py \
    -i ../input/test_images/ \
    -a ../input/test_annotations_output_gamma_multi/ \
    -o ../output_tmp.csv \
    -m 0.6 \
    -c 0.88 \
    -ww 1.1 \
    -wd 0.5 \
    -s test
# 誤検出を補完
$ python repeat.py -i output_tmp.csv -o output_tmp2.csv
$ python repeat.py -i output_tmp2.csv -o output.csv
```

---
## 【任意項目】創意工夫

創意工夫した点に関しまして、こちらに記述してください。

1. boxからtextへ変換を考える。
boxのxy座標からクラスタリングできるのではと考えた。
trainのannotationでの実行結果(正解ラベルとの完全一致率)は以下の通り。

DBSCAN: 5433/6162 = 88.18%
KMeans: 5701/6162 = 92.52%
GMM: 5282/6162 = 85.72%

Kを決めることができるKMeansが最も良くなったが、2012.01.23のような場合に中心からの距離によって分類されるので201_201_23と分類されてしまう。
この場合、boxがあっていても1/3のスコアになってしまう。
(2*1 / (3+3) = 33.33%)

密度ベースのクラスタリングであるDBSCANではeps(クラスに属する境界値)を座標ごとにboxを並べたときのboxの中心同士の距離の平均と各boxの幅の平均の半分(eps=dist_avg+(width_avg/2))と設定しクラスタリングした。
2012.01.23のようなケースでもよくクラスタリングされたが、11 24 98というようなケース（おそらく1が多いケース）でepsがシビアになり1＿1_2_4_9_8のようにすべて別のクラスタとして分類されてしまったり、11_24_9_8のように一部別のクラスタとして分類されてしまうことがあり精度が低下した。
この場合、boxがあっていてもスコアが段階的に低下する。
(
2*0 / (3+6) = 0%
2*1 / (3+5) = 25%
2*2 / (3+4) = 57.14%
)

trainのデータを分析したところ、
”％y %-m %-d”というformatが圧倒的に多く、2000年をこえると、”％y %m %d”のフォーマットで01 02 03が"%d %m %y"や"%m %d %y"などのフォーマットと区別がつかなくなることからか減少し、逆に年を4桁表示する"%Y %-m %-d"や"%Y %m %d"といったフォーマットが増えている。２０１０年以降ではこの２つのフォーマットで統一されていた。

この性質を利用し、gestalt pattern maching(GPM)といるアルゴリズムで文字列の最も類似しているものを選択することで、精度を向上できるのではと考えた。
初期値をboxのxy座標順に並べたときの文字列（空白文字なし）、KMeansでクラスタリングしたときの文字列、DBSCANでクラスタリングしたときの文字列の３種類で先程と同様の実験を行った。

GPM: 5922/6162 = 96.11%
GPM_KMeans: 5709/6162 = 92.65%
GPM_DBSCAN: 6094/6162 = 98.90%

GPMでは例えば99123が99_12_3なのか、99_1_23なのか区別がつかない。

KMeansで初期値を決めた場合は、先程の201_201_23の場合に2012_01_23と2010_01_23等の距離が同じになってしまい、トータルで多少改善したがKMeansの出力に依存してGPM単体よりも悪い結果であった。

密度ベースであるDBSCANは、GPM単体で区別できないケースが改善され、DBSCAN単体で増えてしまったクラスタをGPMによってまとめることができ、精度が大幅に改善した。
完全一致率が98.90%と非常に高く、学習データのノイズのないboxであればほぼ正確にboxから日付のtextに変換できるようになった。
今回のスコアの算出法で計算すると学習データは99.88％の精度であった。


2. boxの検出
benchmarkでは画像からキャプショニングを行っていたが、1.によってboxからtextに変換できるようになっていたので、画像からtextに変換するよりも、画像からboxを正確に検出するほうが、学習が簡単になり精度が高くなるのではと考えた。

そこで物体検出で定評のあるYOLOv3を用いて学習を行った。

データを良く確認していなかった時に、日付の文字列は７セグの文字列であると仮定して、boxで切り抜いて文字ごとのtemplateを作成し、７セグの文字列が赤ぽいのでRGBのRに注目して２値化し、単純にtemplateマッチングでbox検出を行っていた。
（実際は７セグ以外の文字列や白色や青色の文字列があった。）
この時に２値化しようとした際に、しきい値がシビアで文字列とそれ以外で分離が難しいことがわかった。
そこでgamma補正を行うことで、しきい値を設定しやすくなるだろうと考えた。
このアプローチは誤検出が多くて、提出してみたものの空ファイルのスコアよりも下がってしまうので、途中で切り替えた。

上記のようなことがあったため、
データの水増しとして、gamma補正をおこなった。
gammaを4と0.25で水増ししたデータで学習した。
8と日を誤検出することがあったので0-9と日の11クラスの検出問題を学習した。

3. その他
学習したモデルでtestデータに対してboxを推定してスコアの高いboxを加えて更に学習した（ブートストラップ）。
(ブートストラップとは、少数のデータに付けられたラベルに基づいて、他のラベルなしデータを分類する。そして、分類結果をも訓練ラベルとして扱い分類器を再学習する。この手続きを繰り返す手法。)
誤検出したものも正解のラベルとして学習してしまうため、返って精度が落ちてしまった。

明らかな誤検出を減らすため、文字領域を検出するよう学習させた。
期間内に十分な学習が行えなかったため、精度が少し低下してしまった。
文字領域をannotationとして組み込むなど、もう少し工夫したかった。

データが連続している区間あったため、前後の予測結果から推定値の距離が一定以内の場合に置き換える処理を行った。
若干であるが精度が改善した。






# YOLO3 (Detection, Training, and Evaluation)

## Dataset and Model

Dataset | mAP | Demo | Config | Model
:---:|:---:|:---:|:---:|:---:
Kangaroo Detection (1 class) (https://github.com/experiencor/kangaroo) | 95% | https://youtu.be/URO3UDHvoLY | check zoo | http://bit.do/ekQFj
Raccoon Detection (1 class) (https://github.com/experiencor/raccoon_dataset) | 98% | https://youtu.be/lxLyLIL7OsU | check zoo | http://bit.do/ekQFf
Red Blood Cell Detection (3 classes) (https://github.com/experiencor/BCCD_Dataset) | 84% | https://imgur.com/a/uJl2lRI | check zoo | http://bit.do/ekQFc
VOC (20 classes) (http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) | 72% | https://youtu.be/0RmOI6hcfBI | check zoo | http://bit.do/ekQE5

## Todo list:
- [x] Yolo3 detection
- [x] Yolo3 training (warmup and multi-scale)
- [x] mAP Evaluation
- [x] Multi-GPU training
- [x] Evaluation on VOC
- [ ] Evaluation on COCO
- [ ] MobileNet, DenseNet, ResNet, and VGG backends

## Detection

Grab the pretrained weights of yolo3 from https://pjreddie.com/media/files/yolov3.weights.

```python yolo3_one_file_to_detect_them_all.py -w yolo3.weights -i dog.jpg``` 

## Training

### 1. Data preparation 

Download the Raccoon dataset from from https://github.com/experiencor/raccoon_dataset.

Organize the dataset into 4 folders:

+ train_image_folder <= the folder that contains the train images.

+ train_annot_folder <= the folder that contains the train annotations in VOC format.

+ valid_image_folder <= the folder that contains the validation images.

+ valid_annot_folder <= the folder that contains the validation annotations in VOC format.
    
There is a one-to-one correspondence by file name between images and annotations. If the validation set is empty, the training set will be automatically splitted into the training set and validation set using the ratio of 0.8.

### 2. Edit the configuration file
The configuration file is a json file, which looks like this:

```python
{
    "model" : {
        "min_input_size":       352,
        "max_input_size":       448,
        "anchors":              [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  373,326],
        "labels":               ["raccoon"]
    },

    "train": {
        "train_image_folder":   "/home/andy/data/raccoon_dataset/images/",
        "train_annot_folder":   "/home/andy/data/raccoon_dataset/anns/",      
          
        "train_times":          10,             # the number of time to cycle through the training set, useful for small datasets
        "pretrained_weights":   "",             # specify the path of the pretrained weights, but it's fine to start from scratch
        "batch_size":           16,             # the number of images to read in each batch
        "learning_rate":        1e-4,           # the base learning rate of the default Adam rate scheduler
        "nb_epoch":             50,             # number of epoches
        "warmup_epochs":        3,              # the number of initial epochs during which the sizes of the 5 boxes in each cell is forced to match the sizes of the 5 anchors, this trick seems to improve precision emperically
        "ignore_thresh":        0.5,
        "gpus":                 "0,1",

        "saved_weights_name":   "raccoon.h5",
        "debug":                true            # turn on/off the line that prints current confidence, position, size, class losses and recall
    },

    "valid": {
        "valid_image_folder":   "",
        "valid_annot_folder":   "",

        "valid_times":          1
    }
}

```

The ```labels``` setting lists the labels to be trained on. Only images, which has labels being listed, are fed to the network. The rest images are simply ignored. By this way, a Dog Detector can easily be trained using VOC or COCO dataset by setting ```labels``` to ```['dog']```.

Download pretrained weights for backend at:

https://1drv.ms/u/s!ApLdDEW3ut5fgQXa7GzSlG-mdza6

**This weights must be put in the root folder of the repository. They are the pretrained weights for the backend only and will be loaded during model creation. The code does not work without this weights.**

### 3. Generate anchors for your dataset (optional)

`python gen_anchors.py -c config.json`

Copy the generated anchors printed on the terminal to the ```anchors``` setting in ```config.json```.

### 4. Start the training process

`python train.py -c config.json`

By the end of this process, the code will write the weights of the best model to file best_weights.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file). The training process stops when the loss on the validation set is not improved in 3 consecutive epoches.

### 5. Perform detection using trained weights on image, set of images, video, or webcam
`python predict.py -c config.json -i /path/to/image/or/video`

It carries out detection on the image and write the image with detected bounding boxes to the same folder.

## Evaluation

`python evaluate.py -c config.json`

Compute the mAP performance of the model defined in `saved_weights_name` on the validation dataset defined in `valid_image_folder` and `valid_annot_folder`.
