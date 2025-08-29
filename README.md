# AdCo
<a href="https://github.com/marktext/marktext/releases/latest">
   <img src="https://img.shields.io/badge/AdCo-v1.0.0-green">
   <img src="https://img.shields.io/badge/platform-Linux%20%7C%20Mac%20-green">
   <img src="https://img.shields.io/badge/Language-python3-green">
   <img src="https://img.shields.io/badge/dependencies-tested-green">
   <img src="https://img.shields.io/badge/licence-GNU-green">
</a>   

This is a modified version of original AdCo code with following changes:

1- The original code was meant to run on multiple GPUs. Although you can actually use that code for a single-GPU setup, it seemed to me that it would add some delay and perhaps computational overhead. The code in this repo can be run on a single GPU easily.

2- The architecture of conventional ResNets, as proposed in the paper "Deep Residual Learning for Image Recognition," was modified to meet the needs of the CIFAR-10 dataset (other options may be employed given sufficient processing power).

3- The transformations were adjusted to match the CIFAR-10 images.

4- I tried to find the best possible hyperparameters.
  

AdCo is published on [CVPR2021](https://openaccess.thecvf.com/content/CVPR2021/html/Hu_AdCo_Adversarial_Contrast_for_Efficient_Learning_of_Unsupervised_Representations_From_CVPR_2021_paper.html).

Copyright (C) 2020 Qianjiang Hu*, Xiao Wang*, Wei Hu, Guo-Jun Qi

License: MIT for academic use.

Contact: erfankolsoumian@gmail.com






## Installation  
CUDA version should be 10.1 or higher. 
### 1. [`Install git`](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) 
### 2. Clone the repository in your computer 
```
git clone git@github.com:maple-research-lab/AdCo.git && cd AdCo
```

### 3. Build dependencies.   
You have two options to install dependency on your computer:
#### 3.1 Install with pip and python(Ver 3.6.9).
##### 3.1.1[`install pip`](https://pip.pypa.io/en/stable/installing/).
##### 3.1.2  Install dependency in command line.
```
pip install -r requirements.txt --user
```
If you encounter any errors, you can install each library one by one:
```
pip install torch==1.7.1
pip install torchvision==0.8.2
pip install numpy==1.19.5
pip install Pillow==5.1.0
pip install tensorboard==1.14.0
pip install tensorboardX==1.7
```

#### 3.2 Install with anaconda
##### 3.2.1 [`install conda`](https://docs.conda.io/projects/conda/en/latest/user-guide/install/macos.html). 
##### 3.2.2 Install dependency in command line
```
conda create -n AdCo python=3.6.9
conda activate AdCo
pip install -r requirements.txt 
```
Each time when you want to run my code, simply activate the environment by
```
conda activate AdCo
conda deactivate(If you want to exit) 
```



## Usage

### Unsupervised Training
#### Single Crop
##### 1 Without symmetrical loss:
```
python3 main_adco.py --sym=1 --lr=0.04 --memory_lr=3.0 --moco_t=0.14 --mem_t=0.03 --data=./datasets/imagenet2012 --dist_url=tcp://localhost:10001 --multiprocessing_distributed=0 --gpu 0 --arch resnet18 --workers=4 --epochs  800 --batch_size 256 --lr_final 0.0006 --moco_m 0.999 --cluster 8192
```
##### 2 With symmetrical loss:
```
python3 main_adco.py --sym=1 --lr=0.04 --memory_lr=3.0 --moco_t=0.14 --mem_t=0.03 --data=./datasets/imagenet2012 --dist_url=tcp://localhost:10001 --multiprocessing_distributed=0 --gpu 0 --arch resnet18 --workers=4 --epochs  800 --batch_size 256 --lr_final 0.0006 --moco_m 0.999 --cluster 8192
```


### Linear Classification
With a pre-trained model, we can easily evaluate its performance on desired dataset with:
```
python3 lincls.py --data=./datasets/imagenet2012 --dist-url=tcp://localhost:10001 --pretrained=input.pth.tar
```

Performance:
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">pre-train<br/>network</th>
<th valign="bottom">pre-train<br/>epochs</th>
<th valign="bottom">Crop</th>
<th valign="bottom">Symmetrical<br/>Loss</th>
<th valign="bottom">AdCo<br/>KNN acc.</th>
<!-- TABLE BODY -->
<tr><td align="left">ResNet-18</td>
<td align="center">800</td>
<td align="center">Single</td>
<td align="center">No</td>
<td align="center">88.13</td>

<tr><td align="left">ResNet-18</td>
<td align="center">800</td>
<td align="center">Single</td>
<td align="center">Yes</td>
<td align="center">90.27</td>



</tr>
</tbody></table>




## Citation:
[AdCo: Adversarial Contrast for Efficient Learning of Unsupervised Representations from Self-Trained Negative Adversaries](https://arxiv.org/pdf/2011.08435.pdf).  
```
@inproceedings{hu2021adco,
  title={Adco: Adversarial contrast for efficient learning of unsupervised representations from self-trained negative adversaries},
  author={Hu, Qianjiang and Wang, Xiao and Hu, Wei and Qi, Guo-Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1074--1083},
  year={2021}
}
```

