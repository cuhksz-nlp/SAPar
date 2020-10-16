## Data Processing

We use datasets in three languages: Arabic, Chinese, and English.
 
* Arabic: we use ATB2.0 part 1-3 ([LDC2003T06](https://catalog.ldc.upenn.edu/LDC2003T06), [LDC2004T02](https://catalog.ldc.upenn.edu/LDC2004T02), and [LDC2005T20](https://catalog.ldc.upenn.edu/LDC2005T20)).
* Chinese: we use CTB5 ([LDC2005T01](https://catalog.ldc.upenn.edu/LDC2005T01)).
* English: we use PTB ([LDC99T42](https://catalog.ldc.upenn.edu/LDC99T42)). 

You need obtain the official data yourself and then put the data under the current directory.

#### ATB

Put `LDC03T06`, `LDC04T02`, and `LDC05T20` under the current directory and run `./get_atb.sh`. You will see the processed data with gold POS tag in `../data/ATB` and the data with predicted POS tags in `../data/ATB_POS`.

#### CTB

Put `LDC05T01` under the current directory and run `./get_ctb.sh`. You will see the processed data with gold POS tag in `../data/CTB` and the data with predicted POS tags in `../data/CTB_POS`.

#### PTB

Put `LDC99T42` under the current directory and run `./get_ptb.sh`. You will see the processed data with gold POS tag in `../data/PTB` and the data with predicted POS tags in `../data/PTB_POS`.
