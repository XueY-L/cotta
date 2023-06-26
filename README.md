# CoTTA: Cotinual Test-Time Adaptation
Code for continual test-time adaptation methods for classification and segmentation
+ our CVPR 2022 paper [Continual Test-Time Domain Adaptation](https://arxiv.org/abs/2203.13591) 
+ [AdaBN / BN Adapt](https://www.sciencedirect.com/science/article/abs/pii/S003132031830092X)
+ [TENT](https://arxiv.org/abs/2006.10726)

## Prerequisite
Please create and activate the following conda envrionment. To reproduce our results, please kindly create and use this environment.
```bash
# It may take several minutes for conda to solve the environment
conda update conda
conda env create -f environment.yml
conda activate cotta 
```

## Experiment 
### CIFAR10-to-CIFAR10C-standard task
```bash
# Tested on RTX2080TI
cd cifar
bash run_cifar10.sh 
```
### CIFAR10-to-CIFAR10C-gradual task
```bash
# Tested on RTX2080TI
bash run_cifar10_gradual.sh
```
### CIFAR100-to-CIFAR100C task
```bash
# Tested on RTX3090
bash run_cifar100.sh
```

### ImageNet-to-ImageNetC task 
```bash
# Tested on RTX3090
cd imagenet
bash run.sh
```

### Cityscapes-to-ACDC segmentation task
Since April 2022, we also offer the segmentation code based on Segformer.
You can download it [here](https://github.com/qinenergy/cotta/issues/6)
```
## environment setup
conda env create -f environment_segformer.yml
pip install -e . --user
conda activate segformer
## Run
bash run_base.sh
bash run_tent.sh
bash run_cotta.sh
# Example rerun logs are included in ./example_logs/base.log, tent.log, and cotta.log.
## License for Cityscapses-to-ACDC code
Non-commercial. Code is heavily based on Segformer. Please also check Segformer's LICENSE.
```

## Citation
Please cite our work if you find it useful.
```bibtex
@inproceedings{wang2022continual,
  title={Continual Test-Time Domain Adaptation},
  author={Wang, Qin and Fink, Olga and Van Gool, Luc and Dai, Dengxin},
  booktitle={Proceedings of Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## Acknowledgement 
+ TENT code is heavily used. [official](https://github.com/DequanWang/tent) 
+ KATANA code is used for augmentation. [official](https://github.com/giladcohen/KATANA) 
+ Robustbench [official](https://github.com/RobustBench/robustbench) 

## Data links
+ ImageNet-C [Download](https://zenodo.org/record/2235448#.Yj2RO_co_mF)
+ [Supplementary](https://drive.qin.ee/api/raw/?path=/cv/cvpr2022/03679-supp-1.pdf)

For questions regarding the code, please contact wang@qin.ee .
