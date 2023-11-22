# miccai
# MDRL

This is the implementation for the accepted MICCAI 2023 paper Modularity-Constrained Dynamic Representation Learning for Interpretable Brain Disorder Analysis with Functional MRI. This framework consists of three parts: (1) dynamic graph construction, (2) modularity-constrained spatiotemporal graph neural network (MSGNN) for dynamic feature learning, and (3) prediction and biomarker detection.The whole implementation is built upon [PyTorch](https://pytorch.org).

# Folder Structure

This repository is organized into the following folders:

    - `./main.py`: The main functions for training and testing.
    - `./data_pre.py`: Data preparation.
    - `./net`: Models.
    
 We used the following datasets:

- HAND
- ABIDE (Can be downloaded [here](http://fcon_1000.projects.nitrc.org/indi/abide/))
- MDD (Can be downloaded [here](http://rfmri.org/REST-meta-MDD))

Please place the preprocessed dataset files under the root folder. 

# Dependencies  

The framework needs the following dependencies:

```
torch~=1.13.0
numpy~=1.21.5
torch_scatter~=2.1.0+pt113cu117
scipy~=1.9.3
einops~=0.5.0
```


Many thanks to Dr Byung-Hoon Kim for sharing their project [STAGIN](https://github.com/egyptdj/stagin).
