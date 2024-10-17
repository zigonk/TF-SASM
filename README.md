# TF-SASM: Training-free Spatial-aware Sparse Memory for Multi-object Tracking
[![arXiv](https://img.shields.io/badge/arXiv-2407.04327-COLOR.svg)](https://arxiv.org/abs/2407.04327)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/paper/tf-sasm-training-free-spatial-aware-sparse)](https://paperswithcode.com/paper/tf-sasm-training-free-spatial-aware-sparse)

This repository is an official implementation of [TF-SASM](https://arxiv.org/abs/2407.04327).


## Introduction

**TL; DR.** TF-SASM introduces a novel sparse-memory mechanism that stores longer temporal information and maintains the diversity of object appearance, enabling more efficient use of limited memory. Our implementation is based on the MOTRv2 baseline.

![Overview](https://raw.githubusercontent.com/zyayoung/oss/main/motrv2_main.jpg)

**Abstract.**  In this paper, we propose a novel memory-based approach that selectively stores critical features based on object motion and overlapping awareness, aiming to enhance efficiency while minimizing redundancy. As a result, our method not only store longer temporal information with limited number of stored features in the memory, but also diversify states of a particular object to enhance the association performance. Our approach significantly improves over MOTRv2 in the DanceTrack test set, demonstrating a gain of 2.0% AssA score and 2.1% in IDF1 score.

## Main Results

### DanceTrack

| **HOTA** | **DetA** | **AssA** | **MOTA** | **IDF1** |                                           **URL**                                           |
| :------: | :------: | :------: | :------: | :------: | :-----------------------------------------------------------------------------------------: |
|   71.2   |   83.3   |   61.0   |   92.0   |   73.8   | [model](https://drive.google.com/file/d/1EA4lndu2yQcVgBKR09KfMe5efbf631Th/view?usp=share_link) |


## Installation

The codebase is built on top of [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR), [MOTR](https://github.com/megvii-research/MOTR), [MOTRv2](https://github.com/megvii-research/MOTRv2)

### Requirements

* Install pytorch using conda (optional)

    ```bash
    conda create -n motrv2 python=3.7
    conda activate motrv2
    conda install pytorch=1.8.1 torchvision=0.9.1 cudatoolkit=10.2 -c pytorch
    ```

* Other requirements
    ```bash
    conda env update --file environment.yml --prune
    ```

* Build MultiScaleDeformableAttention
    ```bash
    cd ./models/ops
    sh ./make.sh
    ```

## Usage

### Inference on DanceTrack Test Set

```bash
# run a simple inference on our pretrained weights
./tools/simple_inference.sh ./motrv2_dancetrack.pth
```

## Acknowledgements

- [MOTR](https://github.com/megvii-research/MOTR)
- [ByteTrack](https://github.com/ifzhang/ByteTrack)
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)
- [OC-SORT](https://github.com/noahcao/OC_SORT)
- [DanceTrack](https://github.com/DanceTrack/DanceTrack)
- [BDD100K](https://github.com/bdd100k/bdd100k)
- [MOTRv2](https://github.com/megvii-research/MOTRv2)
