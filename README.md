# Per-Architecture Training-Free Metric Optimization for Neural Architecture Search (PO-NAS)
This repository is the official implementation of [Per-Architecture Training-Free Metric Optimization for Neural Architecture Search].

## Requirements

- [Pytorch v2.1.0 or later](https://pytorch.org)

It is recommended to create a new conda environment.

```bash
conda create -n PO-NAS python=3.8
conda activate PO-NAS
pip install -r requirements.txt
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```

## Training-Free Metrics
### NAS-Bench-201
We directly utilize the precomputed training-free metrics from the [Zero-Cost-NAS](https://github.com/SamsungLabs/zero-cost-nas). To reproduce, please refer to their code.
Additionally, we download the necessary precomputed results from the [Google Drive](https://drive.google.com/drive/folders/1mSKVpH5vqTB1shrKnraKDJy_983dEyQJ), 
and place them into the `NAS-Bench-201/data` folder:
```
NAS-Bench-201/data/
    nb2_cf10_seed42_dlrandom_dlinfo1_initwnone_initbnone.p
    nb2_cf100_seed42_dlrandom_dlinfo1_initwnone_initbnone.p
    nb2_im120_seed42_dlrandom_dlinfo1_initwnone_initbnone.p
```
The validation accuracy for each architecture, which serves as the objective evaluation metric for Bayesian Optimization (BO), is already included in these files. 
To optimize the querying process and reduce time costs, we specifically retrieve the validation accuracy on CIFAR-10 after 12 epochs of training (referred to as "hp=12") from the NAS-Bench-201 dataset. 

This information, along with the associated computational costs, is compiled and stored in the file `data/nb2_cf10_hp12_info.p`. 
Additionally, the test accuracies are gathered and saved in `data/nb2_cf10_test_accuracy.p`, `data/nb2_cf100_test_accuracy.p` and `data/nb2_im120_test_accuracy.p`. 
All these data are sourced from the [NAS-Bench-201 API](https://github.com/D-X-Y/NAS-Bench-201).
### TransNAS-Bench-101
We directly use the training-free metrics computed by [NASLib](https://github.com/automl/NASLib/tree/zerocost). To reproduce the results, please refer to their code. Please download the computed results for [TransNAS-Bench-101-micro](https://drive.google.com/file/d/1SBOVAyhLCBTAJiU_fo7hLRknNrGNqFk7/view) and [TransNAS-Bench-101-macro](https://drive.google.com/file/d/1teH8JcQsamZngUD_DMQyNkCoUYYSTM0M/view), and put them into the `TransNAS-Bench-101-micro/data` and `TransNAS-Bench-101-macro/data` folder:
```
TransNAS-Bench-101-micro/data/
    zc_transbench101_micro.json
```
```
TransNAS-Bench-101-macro/data/
    zc_transbench101_macro.json
```
### For DARTS Search Space
We manually compute the training-free metrics value. While CIFAR-10 and CIFAR-100 can be automatically downloaded using torchvision, 
ImageNet needs to be manually downloaded (preferably to an SSD) following the [instructions](https://github.com/pytorch/examples/tree/main/imagenet).
Please refer to [DARTS](https://github.com/quark0/darts) project for more details.

## Search
### NAS-Bench-201
Below are examples of usage. Please refer to `parse_arguments()` for other possible arguments. For different tasks, please adjust the `diff_threshold` and `loss_threshold` parameters based on the specific total loss and direction loss of the surrogate model.
```bash
python NAS-Bench-201/search_nb201.py --task C10 --diff_threshold 0.1 --loss_threshold 0.1
python NAS-Bench-201/search_nb201.py --task C100 --diff_threshold 0.1 --loss_threshold 0.1
python NAS-Bench-201/search_nb201.py --task IN-16 --diff_threshold 0.1 --loss_threshold 0.1
```
### TransNAS-Bench-101-micro
Below are examples of usage. Please refer to `parse_arguments()` for other possible arguments. For different tasks, please adjust the `diff_threshold` and `loss_threshold` parameters based on the specific total loss and direction loss of the surrogate model.
```bash
python TransNAS-Bench-101-micro/search_tnb101_micro.py --task class_scene --diff_threshold 0.1 --loss_threshold 0.1
python TransNAS-Bench-101-micro/search_tnb101_micro.py --task normal --diff_threshold 0.1 --loss_threshold 0.1
```
### TransNAS-Bench-101-macro
Below are examples of usage. Please refer to `parse_arguments()` for other possible arguments. For different tasks, please adjust the `diff_threshold` and `loss_threshold` parameters based on the specific total loss and direction loss of the surrogate model.
```bash
python TransNAS-Bench-101-macro/search_tnb101_macro.py --task class_scene --diff_threshold 0.1 --loss_threshold 0.1
python TransNAS-Bench-101-macro/search_tnb101_macro.py --task normal --diff_threshold 0.1 --loss_threshold 0.1
```
### DARTS
Below are examples of usage. Please refer to `parse_arguments()` for other possible arguments. For different tasks, please adjust the `diff_threshold` and `loss_threshold` parameters based on the specific total loss and direction loss of the surrogate model.
```bash
python DARTS/search.py --dataset cifar10 --diff_threshold 0.1 --loss_threshold 0.1
python DARTS/search.py --dataset cifar100 --diff_threshold 0.1 --loss_threshold 0.1
python DARTS/search.py --dataset imagenet --data /path/to/imagenet/dataset/ --epochs 3 --total_iters 10 --n_sample 10000 --seed 0 --drop_path_prob 0.0 --learning_rate 1 --train_portion 0.25 --batch_size 800 --diff_threshold 0.1 --loss_threshold 0.1
```
The genotype of the architecture with the optimal validation performance will be printed at the end. Please update this genotype in `DARTS/genotypes.py`. 
Our obtained architecture is already listed in the file. To train and evaluate the performance, run the following command:
```bash
python DARTS/train_cifar10.py --auxiliary --cutout --arch CIFAR_10_arch --save log_name --data path/to/cifar10/data/ --dataset cifar10 --learning_rate 0.025 --auxiliary_weight 0.4 --drop_path_prob 0.2 
python DARTS/train_cifar100.py --auxiliary --cutout --arch CIFAR_100_arch --save log_name --data path/to/cifar100/data/ --dataset cifar100 --learning_rate 0.035 --learning_rate_min 0.0001 --auxiliary_weight 0.6 --drop_path_prob 0.3
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node=8 --master_port 30408 train_imagenet.py --arch ImageNet_arch --save log_name --auxiliary --data_dir path/to/imagenet/data/
```
You can replace the `--arch` configure with your own architecture.

## Results
We list the main results below. For more details, please refer to our paper.

<img src="images/Figure2.jpg">
<img src="images/Table2.jpg">
<img src="images/Table3.jpg">
<img src="images/Table4.jpg">

