# time-distance-transformer
Implementation of vision transformers for sparse, irregularly sampled imaging. Please cite the preprint if you find this helpful:
[Time-distance vision transformers in lung cancer diagnosis from longitudinal computed tomography](https://arxiv.org/abs/2209.01676)

## Prequisite datasets and resources
* **NLST dataset**: This model was validated using an imaging cohort from the National Lung Screening Trial. Lung screening CTs are available through the NIH: https://cdas.cancer.gov/nlst/
* **tumor_cifar**: Synthetic dataset used as proof of concept: https://github.com/MASILab/tumor-cifar
* **Pulmonary nodule detection from Liao et al.[1]**: Model that proposes pulmonary nodule ROIs. Both the raw ROIs or feature vectors from this algorithm are used as input to our model: https://github.com/lfz/DSB2017

## Training
1. Install dependencies from `requirements.txt`.
2. Edit config files in `config/*.YAML` to point to data location.
    * Generate image paths and store as dictionary with each entry being a cross validation fold.
    * see paper for choices of positional and time embeddings
3. Run `main_nlst.py`
    * If input is whole image or ROI, masked autoencoder pretraining is necessary: `python main_nlst.py --config nlst.YAML --pretrain --kfold 0`
    * If input is feature vectors, pretrainin is NOT needed: `python main_nlst.py --config nlst.YAML --train --kfold 0`

## References
1. Liao F, Liang M, Li Z, Hu X, Song S. Evaluate the Malignancy of Pulmonary Nodules Using the 3D Deep Leaky Noisy-or Network. IEEE Transactions on Neural Networks and Learning Systems. 2017;30(11):3484-3495. doi:10.1109/tnnls.2019.2892409.

---
The contents covered by this repository, including code and pretrained models in the docker container, are free for noncommercial usage (CC BY-NC 4.0). Please check the LICENSE.md file for more details of the copyright information.