# Enhance-A-Video

This repository is the official implementation of [Enhance-A-Video: Better Generared Video for Free](https://oahzxl.github.io/Enhance_A_Video/).

![demo](assets/demo.png)

## News
- 2024-12-20: Enhance-A-Video is now available for [CogVideoX](https://github.com/THUDM/CogVideo) and [HunyuanVideo](https://github.com/Tencent/HunyuanVideo)!

## Getting Started

Install the dependencies:

```bash
conda create -n feta python=3.10
conda activate feta
pip install ipykernel
python -m ipykernel install --user --name feta --display-name "feta"
pip install -r requirements.txt
```

Generate videos:

```bash
python cogvideox.py
python hunyuanvideo.py
```
