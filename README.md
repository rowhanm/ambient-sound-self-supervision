# Implementation of "Ambient Sound Provides Supervision for Visual Learning" using PyTorch

0. Install requirements from `requirements.txt`
0.1. Install ffmpeg - https://ffmpeg.org/download.html
0.2. Download pretrained weights - https://drive.google.com/drive/folders/1dqdUiZIlkR3SiaK8tayLaUR0WejKqsct?usp=sharing

1. Download AudioSet

`cat eval_segments.csv | ./download.sh`

2. Extract frames and audio separately from AudioSet

`python create_dataset.py`

3.a Save statistical summary features:

`python feature_saver.py`

or

3.b Save MFCC features:

`python get_mfcc.py` - This creates HDF5 files which are used for training.

4. Train the pretext model:

`python pretext_train_5_alexnet.py` or `python pretext_train_5_resnet.py`

5. Evaluate on Pascal VOC classification:

(Download VOC dataset first)
`python download_voc.py`

`python finetune_voc.py`

6. Evaluate linear classifier on Imagenet:

`python linear_imagenet_classifier.py`
 
7. Visualize learned models:
7a. Gradient Ascent:
`python grad_ascent.py`

7b. Top activation retrieval:
`python activ_retrieval.py`


Authors: Rohan Mahadev, Florence Lu

(Might be unstable, might have to make some changes to hardcoded paths to run) 
