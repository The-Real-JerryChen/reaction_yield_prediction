# UAM
Official implementation for paper "Uncertainty-aware Yield Prediction with Multi-modal Molecular Features".

## Requirements

Required packages and recommended version:

```
python (>=3.10)
pytorch (>= 1.12.1_cuda11.3)
transformers (>=4.37.2)
rdkit (>=2023.9.5)
torch-geometric
torch-scatter
mixture_of_experts
wandb
...
```

## Datasets

We use three datasets, two of which are High-throughput experiment (HTE) datasets and the third one is constructed from patent literature by expert chemists. For the HTE dataset, we provide the raw data under ./data. For the ACR dataset, pls check [here](https://pubs.rsc.org/en/content/articlehtml/2023/sc/d3sc03902a)

## Pre-training

We use InfoNCE loss to pre-train the SMILES encoder and the Molecular Graph encoder. Pls run the following command:

```
python pretrain.py
```

## Training & Evaluation

To train the model in the paper, run this command:

```
python train.py
```

Note that the original model is implemented using DGL and customized Transformer. While in this repo, we use PyG and Pytorch.transformer_encoder. So the best hyper-parmeters will be slightly different.

You can use following command to find the best configurations.

```
python sweep.py
```


## References
If you find this repository useful in your research, please cite the following paper:
```
@inproceedings{chen2024uncertainty,
  title={Uncertainty-Aware Yield Prediction with Multimodal Molecular Features},
  author={Chen, Jiayuan and Guo, Kehan and Liu, Zhen and Isayev, Olexandr and Zhang, Xiangliang},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2024}
}
```
