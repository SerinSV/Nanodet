# NanoDet

- NanoDet is a FCOS-style one-stage anchor-free object detection model which using Generalised Focal Loss as classification and regression loss. 
- FCOS (Fully Convolutional One-Stage): FCOS is an object detection framework that performs object detection in a single stage using fully convolutional networks. Unlike two-stage detectors like Faster R-CNN, which rely on a separate region proposal network (RPN) to generate anchor boxes, FCOS eliminates the need for anchors and performs object detection directly on the feature map.
- Anchor-Free Object Detection: In traditional object detection models, anchor boxes are predefined boxes of various sizes and aspect ratios that are placed on the image. The model then predicts whether each anchor box contains an object and adjusts its position and size. Anchor-free object detection models, like FCOS, do not use predefined anchor boxes. Instead, they directly predict the bounding boxes and class probabilities for objects in the image.
- Generalised Focal Loss (GFL): The Focal Loss, introduced in the RetinaNet object detection model, is designed to address the class imbalance problem in object detection datasets. It assigns higher weights to hard examples (misclassified or difficult-to-classify examples) and lower weights to easy examples. Generalised Focal Loss (GFL) extends this concept by considering additional factors such as class distribution and overlapping instances, making it a more versatile loss function for object detection tasks.
- Classification Loss: In object detection, the classification loss is a term that measures the discrepancy between the predicted class probabilities and the ground truth labels for each object. It helps the model learn to accurately classify objects into different classes or categories.
- Regression Loss: The regression loss measures the difference between the predicted bounding box coordinates (e.g., coordinates for the top-left and bottom-right corners) and the ground truth bounding box coordinates for each object. It allows the model to learn to accurately localise and predict the object's position and size.
- By combining the FCOS architecture with the Generalised Focal Loss (GFL), NanoDet achieves efficient and accurate object detection, making it suitable for scenarios with limited computational resources or real-time applications.
- Use this github link to refer vastly [NanoDet](https://github.com/RangiLyu/nanodet.git)

****


## Installation 

### Requirements

* Linux or MacOS
* CUDA >= 10.2
* Python >= 3.7
* Pytorch >= 1.10.0, <2.0.0

### Step

1. Create a conda virtual environment and then activate it.

```shell script
 conda create -n nanodet python=3.8 -y
 conda activate nanodet
```

2. Install pytorch

```shell script
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch -c conda-forge
```

3. Clone this repository

```shell script
git clone https://github.com/RangiLyu/nanodet.git
cd nanodet
```

4. Install requirements

```shell script
pip install -r requirements.txt
```
- Before installing 
Goto requirements.txt → Change pycocotools to pycocotools-windows


5. Setup NanoDet
```shell script
python setup.py develop
```

****

# Inferencing

* Inference images

```bash
python demo/demo.py image --config CONFIG_PATH --model MODEL_PATH --path IMAGE_PATH
```

* Inference video

```bash
python demo/demo.py video --config CONFIG_PATH --model MODEL_PATH --path VIDEO_PATH
```

* Inference webcam

```bash
python demo/demo.py webcam --config CONFIG_PATH --model MODEL_PATH --camid YOUR_CAMERA_ID
```


- When inferencing this an error would occur:

- OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

- To solve this paste the following in demo.py:
				os.environ['KMP_DUPLICATE_LIB_OK']='True'
- Also Cuda won’t support so change 'cuda' to 'cpu' in code where cuda occurs
****


## Model Zoo

NanoDet supports variety of backbones. Go to the [***config*** folder](config/) to see the sample training config files.

Model                 | Backbone           |Resolution|COCO mAP| FLOPS |Params | Pre-train weight |
:--------------------:|:------------------:|:--------:|:------:|:-----:|:-----:|:-----:|
NanoDet-m             | ShuffleNetV2 1.0x  | 320*320  |  20.6  | 0.72G | 0.95M | [Download](https://drive.google.com/file/d/1ZkYucuLusJrCb_i63Lid0kYyyLvEiGN3/view?usp=sharing) |
NanoDet-Plus-m-320 (***NEW***)     | ShuffleNetV2 1.0x | 320*320  |  27.0  | 0.9G  | 1.17M | [Weight](https://drive.google.com/file/d/1Dq0cTIdJDUhQxJe45z6rWncbZmOyh1Tv/view?usp=sharing) &#124; [Checkpoint](https://drive.google.com/file/d/1YvuEhahlgqxIhJu7bsL-fhaqubKcCWQc/view?usp=sharing)
NanoDet-Plus-m-416 (***NEW***)     | ShuffleNetV2 1.0x | 416*416  |  30.4  | 1.52G | 1.17M | [Weight](https://drive.google.com/file/d/1FN3WK3FLjBm7oCqiwUcD3m3MjfqxuzXe/view?usp=sharing) &#124; [Checkpoint](https://drive.google.com/file/d/1gFjyrl7O8p5APr1ZOtWEm3tQNN35zi_W/view?usp=sharing)
NanoDet-Plus-m-1.5x-320 (***NEW***)| ShuffleNetV2 1.5x | 320*320  |  29.9  | 1.75G | 2.44M | [Weight](https://drive.google.com/file/d/1Xdlgu5lxiS3w6ER7GE1mZpY663wmpcyY/view?usp=sharing) &#124; [Checkpoint](https://drive.google.com/file/d/1qXR6t3TBMXlz6GlTU3fxiLA-eueYoGrW/view?usp=sharing)
NanoDet-Plus-m-1.5x-416 (***NEW***)| ShuffleNetV2 1.5x | 416*416  |  34.1  | 2.97G | 2.44M | [Weight](https://drive.google.com/file/d/16FJJJgUt5VrSKG7RM_ImdKKzhJ-Mu45I/view?usp=sharing) &#124; [Checkpoint](https://drive.google.com/file/d/17sdAUydlEXCrHMsxlDPLj5cGb-8-mmY6/view?usp=sharing)


*Notice*: The difference between `Weight` and `Checkpoint` is the weight only provide params in inference time, but the checkpoint contains training time params.


****

# Steps for Colab

1. Change runtime to gpu


2. Mount drive
```shell script
from google.colab import drive
drive.mount('/content/drive')
```
3. Clone Repository
```shell script
!git clone https://github.com/SerinSV/Nanodet.git
```
4. Change the directory to the uploaded repository


```shell script
!pip install -r requirements.txt
```
```shell script
!pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchtext==0.14.1 torchaudio==0.13.1 torchdata==0.5.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
```shell script
!python setup.py develop
```

* Inference images

```bash
python demo/demo.py image --config CONFIG_PATH --model MODEL_PATH --path IMAGE_PATH
```

* Inference video

```bash
python demo/demo.py video --config CONFIG_PATH --model MODEL_PATH --path VIDEO_PATH
```

* Inference webcam

```bash
python demo/demo.py webcam --config CONFIG_PATH --model MODEL_PATH --camid YOUR_CAMERA_ID
```

Besides, We provide a notebook [here](https://github.com/SerinSV/Nanodet/blob/main/demo/Nanodet_colab.ipynb) to demonstrate how to make it work in colab.

****

## How to Train

1. **Prepare dataset**

    If your dataset annotations are pascal voc xml format, refer to [config/nanodet_custom_xml_dataset.yml](config/nanodet_custom_xml_dataset.yml)

    Otherwise, if your dataset annotations are YOLO format ([Darknet TXT](https://github.com/AlexeyAB/Yolo_mark/issues/60#issuecomment-401854885)), refer to [config/nanodet-plus-m_416-yolo.yml](config/nanodet-plus-m_416-yolo.yml)

    Or convert your dataset annotations to MS COCO format[(COCO annotation format details)](https://cocodataset.org/#format-data).

2. **Prepare config file**

    Copy and modify an example yml config file in config/ folder.

    Change ***save_dir*** to where you want to save model.

    Change ***num_classes*** in ***model->arch->head***.

    Change image path and annotation path in both ***data->train*** and ***data->val***.

    Set gpu ids, num workers and batch size in ***device*** to fit your device.

    Set ***total_epochs***, ***lr*** and ***lr_schedule*** according to your dataset and batchsize.

    If you want to modify network, data augmentation or other things, please refer to [Config File Detail](docs/config_file_detail.md)

3. **Start training**

   NanoDet is now using [pytorch lightning](https://github.com/PyTorchLightning/pytorch-lightning) for training.

   For both single-GPU or multiple-GPUs, run:

   ```shell script
   python tools/train.py CONFIG_FILE_PATH
   ```

4. **Visualize Logs**

    TensorBoard logs are saved in `save_dir` which you set in config file.

    To visualize tensorboard logs, run:

    ```shell script
    cd <YOUR_SAVE_DIR>
    tensorboard --logdir ./
    ```

****



