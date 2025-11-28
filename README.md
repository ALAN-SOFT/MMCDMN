# MMCDMN: A Multimodal Cognitive Decision-Making Network for Intelligent Feeding in Recirculating Aquaculture Systems

Implementation of paper - Review stage


## System Requirements
- **Operating System**: Ubuntu 20.04/22.04 (Recommended) or Windows 10/11 (WSL2 required)
- **GPU**: NVIDIA GPU (≥8GB VRAM recommended)
- **CUDA**: 11.3+ (Must match PyTorch version)
- **Python**: 3.9 or 3.10

## Installation Steps

  Clone Repository
```bash
git clone 
cd MMCDMN
```

  Create Python Virtual Environment
```bash
python -m venv MMCDMN
source MMCDMN/bin/activate  # Linux
# or MMCDMN\Scripts\activate  # Windows
```

  Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```


## Training

Data preparation: https://github.com/ALAN-SOFT/Demonstration-Dataset
git clone 


Single GPU training

``` shell

python train.py --workers 8 --device 0 --batch-size 32 --data data/fish.yaml --img 640 640 --cfg cfg/training/yolofish.yaml --weights

python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/fish.yaml --img 1280 1280 --cfg cfg/training/yolofish2.yaml --weights 
```

Multiple GPU training

``` shell
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data data/fish.yaml --img 640 640 --cfg cfg/training/yolofish.yaml

python -m torch.distributed.launch --nproc_per_node 8 --master_port 9527 train_aux.py --workers 8 --device 0,1,2,3,4,5,6,7 --sync-bn --batch-size 128 --data data/fish.yaml --img 1280 1280 --cfg cfg/training/yolofish1.yaml
```

## Transfer learning

[`yolo11_fish.pt`](https://github.com/) [`yolo11_fish.pt`](https://github.com/)



## Acknowledgements

<details><summary> <b>Expand</b> </summary>
  
* [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
  
</details>
