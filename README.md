# SAPar

This is the implementation of [Constituency Parsing with Span Attention](https://www.aclweb.org/anthology/2020.findings-emnlp.153/) at Findings of EMNLP2020.

Please contact us at `yhtian@uw.edu` or `cuhksz.nlp@gmail.com` if you have any questions.

## Citation

If you use or extend our work, please cite our paper at Findings of EMNLP-2020.

```
@inproceedings{tian-etal-2020-improving,
    title = "Improving Constituency Parsing with Span Attention",
    author = "Tian, Yuanhe and Song, Yan and Xia, Fei and Zhang, Tong",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    pages = "1691--1703",
}
```

## Prerequisites
* `python 3.6`
* `pytorch 1.1`

Install python dependencies by running:

`
pip install -r requirements.txt
`

`EVALB` and `EVALB_SPMRL` contain the code to evaluate the parsing results for English and other languages. Before running evaluation, you need to go to the `EVALB` (for English) or `EVALB_SPMRL` (for other languages) and run `make`.


## Downloading BERT, ZEN, XLNet and Our Pre-trained Models

In our paper, we use [BERT](https://www.aclweb.org/anthology/N19-1423/), [ZEN](https://arxiv.org/abs/1911.00720), and [XLNet](https://arxiv.org/pdf/1906.08237.pdf) as the encoder.

For BERT, please download pre-trained BERT model from [Google](https://github.com/google-research/bert) and convert the model from the TensorFlow version to PyTorch version. 
* For Arabic, we use MulBERT-Base, Multilingual Cased.
* For Chinese, we use BERT-Base, Chinese;
* For English, we use BERT-Large, Cased and BERT-Large, Uncased.

For ZEN, you can download the pre-trained model from [here](https://github.com/sinovation/ZEN).

For XLNet, you can download the pre-trained model from [here](https://github.com/zihangdai/xlnet).

For our pre-trained model, you can download them from [Baidu Wangpan](https://pan.baidu.com/s/1iSUcfRHccrgGmc2GEsDDBw) (passcode: 2o1n) or [Google Drive](https://drive.google.com/drive/folders/1-wINl7lLtlT0WEX88NPwyBHZOr4yKnCK?usp=sharing).

## Run on Sample Data

To train a model on a small dataset, run:

`
./run.sh
`


## Datasets

We use datasets in three languages: Arabic, Chinese, and English.
 
* Arabic: we use ATB2.0 part 1-3 ([LDC2003T06](https://catalog.ldc.upenn.edu/LDC2003T06), [LDC2004T02](https://catalog.ldc.upenn.edu/LDC2004T02), and [LDC2005T20](https://catalog.ldc.upenn.edu/LDC2005T20)).
* Chinese: we use CTB5 ([LDC2005T01](https://catalog.ldc.upenn.edu/LDC2005T01)).
* English: we use PTB ([LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42)). 

To preprocess the data, please go to `data_processing` directory and follow the [instruction](./data_processing) to process the data. You need to obtain the official datasets yourself before running our code.

Ideally, all data will appear in `./data` directory. The data with gold POS tags are located in folders whose name is the same as the dataset name (i.e., ATB, CTB, and PTB); the data with predicted POS tags are located in folders whose name has a "_POS" suffix (i.e., ATB_POS, CTB_POS, and PTB_POS).



## Training, Testing, and Predicting

You can find the command lines to train and test models on a specific dataset in `run.sh`.


## To-do List

* Regular maintenance.

You can leave comments in the `Issues` section, if you want us to implement any functions.

You can check our updates at [updates.md](./updates.md).

