#  Dual Adversarial Neural Transfer for Sequence Labeling

Source codes for 
* Joey Tianyi Zhou, Hao Zhang, Di Jin, Hongyuan Zhu, Meng Fang, Rick Siow Mong Goh and Kenneth Kwok, "Dual Adversarial Neural Transfer for Low-Resource Named Entity Recognition", The 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019, Long Paper, oral), Florence, Italy, 2019.
* Joey Tianyi Zhou, Hao Zhang, Di Jin, Xi Peng, "Dual Adversarial Neural Transfer for Sequence Labeling", IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI), 2019.

If you feel this project helpful to your research, please cite the following paper
```bibtex
@inproceedings{zhou2019dual_c,
 title = {Dual Adversarial Neural Transfer for Low-Resource Named Entity Recognition},
    author = {Zhou, Joey Tianyi and Zhang, Hao and Jin, Di and Zhu, Hongyuan and Fang, Meng and Goh, Rick Siow Mong and Kwok, Kenneth},
    booktitle = {ACL},
    year = {2019},
    pages = {3461--3471}
}
```
and 
```bibtex
@ARTICLE{8778733, 
author={Zhou, Joey Tianyi and Zhang, Hao and Jin, Di and Peng, Xi}, 
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
title={Dual Adversarial Transfer for Sequence Labeling}, 
year={2019}, 
volume={}, 
number={}, 
pages={1-1}, 
doi={10.1109/TPAMI.2019.2931569}, 
ISSN={0162-8828}, 
month={},}
```

This project includes all the variants of DATNet (i.e., base model, DATNet-F, DATNet-P) implementations reported in the paper. 

## Requirement
* numpy 1.10+
* matplotlib 1.8+
* tensorflow 1.6+
* python 3.6
* scikit-learn 1.16+
 
## Usage
To train a DATNet model, run:
```bash
$ python3 train_datnetf_model.py --task conll2003_ner \  # indicate the dataset for training
                        --use_gpu true \  # use GPU
                        --gpu_idx 0 \  # specify which GPU is utilized                    
                        ...
```
For more details, please ref. `train_datnetf_model.py` or `train_datnetp_model.py`.
