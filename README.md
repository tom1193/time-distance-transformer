# time-distance-transformer
Implementation of vision transformers for sparse, irregularly sampled imaging. Please cite this paper if you find this helpful:
[Time-distance vision transformers in lung cancer diagnosis from longitudinal computed tomography]()

## Prequisite datasets and resources
* **NLST dataset**: This model was validated using an imaging cohort from the National Lung Screening Trial. Lung screening CTs are available through the NIH: https://cdas.cancer.gov/nlst/
* **tumor_cifar**: Synthetic dataset used as proof of concept: https://github.com/MASILab/tumor-cifar
* **Pulmonary nodule detection from Liao et al.**: Model that proposes pulmonary nodule ROIs. Both the raw ROIs or feature vectors from this algorithm are used as input to our model: https://github.com/lfz/DSB2017

## Training
1. Install dependencies from `requirements.txt`.
2. Edit config files in `config/*.YAML` to point to data location.
    * Generate image paths and store as dictionary with each entry being a cross validation fold.
    * see paper for choices of positional and time embeddings
3. Run `main_nlst.py`
    * If input is whole image or ROI, masked autoencoder pretraining is necessary: `python main_nlst.py --config nlst.YAML --pretrain --kfold 0`
    * If input is feature vectors, pretrainin is NOT needed: `python main_nlst.py --config nlst.YAML --train --kfold 0`

