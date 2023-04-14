# SPEC: Summary Preference Decomposition for Low-Resource Abstractive Summarization


<p align="center">
  <img src="https://github.com/YiSyuanChen/SPEC/blob/main/spec_ipl.png" width="276" height="400">            <img src="https://github.com/YiSyuanChen/SPEC/blob/main/spec_iipl.png" width="276" height="400">
</p>

[[Paper](https://arxiv.org/abs/2303.14011)]
[[Datasets](https://drive.google.com/file/d/1J8fkKGB8CFG0EBWOHIW1IlFriDmLTOo2/view?usp=sharing)]
[[Models](https://drive.google.com/file/d/162sGfo6GXQGU4OIRnG4rPTdBb6CRkH80/view?usp=sharing)]


# Introduction

Original PyTorch implementation for TASLP 2022 Paper "SPEC: Summary Preference Decomposition for Low-Resource Abstractive Summarization
" by Yi-Syuan Chen, Yun-Zhu Song, and Hong-Han Shuai.


# Instructions
## Dataset
The datasets are organized in CSV format for individual splits, where each row denotes a single datum. The input and output column names may differ across datasets, which will be handle within the codes (_src/data/dataset_mappings.py_). The CSV files also include the preference values for each datum, which are obtained by runing the script in _src/get_preference/_. To generate the representative preference, please also refer to the script in _src/get_preference/_.


You may check the formats and prepare your datasets accordingly to run on our codes.

We provide part of the datasets that can be directly used for exploring. After downloading from [[Datasets](https://drive.google.com/file/d/1Sok3fXqk4vG8bmjvULGurGRQeEGxiZzG/view?usp=sharing)], unzip the compressed file and place the _datasets/_ and _analysis_/ folders under _SPEC/_. The _datasets/_ folder contains the CSV files while the _analysis/_ folder contains the representative preferences required for runing the codes.

## Run the Codes
We provide the scripts for training, adaptation, and testing of the SPEC-IPL and SPEC-IIPL. Please run them with _./scripts/<script_name>.sh_ under _src/_ folder. 

We provide part of the models that can be directly used for exploring. After downloading from [[Models](https://drive.google.com/file/d/1Sok3fXqk4vG8bmjvULGurGRQeEGxiZzG/view?usp=sharing)], unzip the compressed file and place the _results/_ folder under _SPEC/_.


# Citation
```
@ARTICLE{9992078,
    author={Chen, Yi-Syuan and Song, Yun-Zhu and Shuai, Hong-Han},
    journal={IEEE/ACM Transactions on Audio, Speech, and Language Processing}, 
    title={SPEC: Summary Preference Decomposition for Low-Resource Abstractive Summarization}, 
    year={2023},
    volume={31},
    number={},
    pages={603-618},
    doi={10.1109/TASLP.2022.3230539}
}
```
