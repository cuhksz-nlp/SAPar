# SAPar

This is the implementation of [Constituency Parsing with Span Attention](https://www.aclweb.org/anthology/) at Findings of EMNLP2020.

This implementation is based on [Berkeley Neural Parser](https://github.com/nikitakit/self-attentive-parser).

You can e-mail Yuanhe Tian at `yhtian@uw.edu` or Guimin Chen at `cuhksz.nlp@gmail.com`, if you have any questions.

## Citation

If you use or extend our work, please cite our paper at EMNLP-2020.

```
@inproceedings{tian-etal-2020-constituency,
    title = "Constituency Parsing with Span Attention",
    author = "Tian, Yuanhe and Song, Yan and Xia, Fei and Zhang, Tong",
    booktitle = "Findings of the 2020 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2020",
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

For our pre-trained model, we will release it soon.

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

We will release the code to pre-process the data soon.


## Training, Testing, and Predicting

You can find the command lines to train and test models on a specific dataset in `run.sh`.


## To-do List

* Release the code to pre-process the data.
* Release the pre-trained model for CCG supertagging.
* Regular maintenance.

You can leave comments in the `Issues` section, if you want us to implement any functions.

You can check our updates at [updates.md](./updates.md).

