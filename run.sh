# prepare evaluation
cd EVALB
make
cd ..

cd EVALB_SPMRL
make
cd ..

# on sample data
# train xlnet+CatSA without POS tags
python SAPar_main.py train --use-xlnet --no-bert-do-lower-case --xlnet-model=/path/to/XLNet_large_cased --evalb-dir=./EVALB --train-path=./sample_data/train.mrg --dev-path=./sample_data/dev.mrg --test-path=./sample_data/test.mrg --model-path-base models/test --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15

# train xlnet+CatSA with POS tags
python SAPar_main.py train --use-xlnet --no-bert-do-lower-case --xlnet-model=/path/to/XLNet_large_cased --use-tags --evalb-dir=./EVALB --train-path=./sample_data/train.mrg --dev-path=./sample_data/dev.mrg --test-path=./sample_data/test.mrg --model-path-base models/test --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15

# train bert+CatSA without POS tags
python SAPar_main.py train --use-bert --no-bert-do-lower-case --bert-model=/path/to/bert_large_cased --evalb-dir=./EVALB --train-path=./sample_data/train.mrg --dev-path=./sample_data/dev.mrg --test-path=./sample_data/test.mrg --model-path-base models/test --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15

# train bert+CatSA without POS tags
python SAPar_main.py train --use-bert --no-bert-do-lower-case --bert-model=/path/to/bert_large_cased --use-tags --evalb-dir=./EVALB --train-path=./sample_data/train.mrg --dev-path=./sample_data/dev.mrg --test-path=./sample_data/test.mrg --model-path-base models/test --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15


# On ATB
# train bert+CatSA on ATB without POS tags
python SAPar_main.py train --use-bert --no-bert-do-lower-case --bert-model=/path/to/multi_base_cased --evalb-dir=./EVALB_SPMRL --train-path=./data/ATB/train.mrg --dev-path=./data/ATB/dev.mrg --test-path=./data/ATB/test.mrg --model-path-base models/SAPar.ATB.BERT --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1500 --ngram-type=pmi --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=200 --max-len-dev=200 --patients=15

# train bert+CatSA on ATB with predicted POS tags
python SAPar_main.py train --use-bert --use-tags --no-bert-do-lower-case --bert-model=/path/to/multi_base_cased --evalb-dir=./EVALB_SPMRL --train-path=./data/ATB_POS/train.mrg --dev-path=./data/ATB_POS/dev.mrg --test-path=./data/ATB_POS/test.mrg --model-path-base models/SAPar.ATB.BERT.POS --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1500 --ngram-type=pmi --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=200 --max-len-dev=200 --patients=15

# On CTB
# train bert+CatSA on CTB without POS tags
python SAPar_main.py train --use-bert --no-bert-do-lower-case --bert-model=/path/to/bert_base_chinese1 --evalb-dir=./EVALB_SPMRL --train-path=./data/CTB/train.mrg --dev-path=./data/CTB/dev.mrg --test-path=./data/CTB/test.mrg --model-path-base models/SAPar.CTB.BERT --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-type=pmi --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15

# train bert+CatSA on CTB with predicted POS tags
python SAPar_main.py train --use-bert --use-tags --no-bert-do-lower-case --bert-model=/path/to/bert_base_chinese1 --evalb-dir=./EVALB_SPMRL --train-path=./data/CTB_POS/train.mrg --dev-path=./data/CTB_POS/dev.mrg --test-path=./data/CTB_POS/test.mrg --model-path-base models/SAPar.CTB.BERT.POS --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-type=pmi --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15

# train zen+CatSA on CTB without POS tags
python SAPar_main.py train --use-zen --no-bert-do-lower-case --zen-model=/path/to/ZEN_pretrain_base_v0.1.0 --evalb-dir=./EVALB_SPMRL --train-path=./data/CTB/train.mrg --dev-path=./data/CTB/dev.mrg --test-path=./data/CTB/test.mrg --model-path-base models/SAPar.CTB.ZEN --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-type=pmi --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15

# train zen+CatSA on CTB with predicted POS tags
python SAPar_main.py train --use-zen --use-tags --no-bert-do-lower-case --zen-model=/path/to/ZEN_pretrain_base_v0.1.0 --evalb-dir=./EVALB_SPMRL --train-path=./data/CTB_POS/train.mrg --dev-path=./data/CTB_POS/dev.mrg --test-path=./data/CTB_POS/test.mrg --model-path-base models/SAPar.CTB.ZEN.POS --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-type=pmi --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15

# on PTB
# train bert+CatSA without POS tags
python SAPar_main.py train --use-bert --no-bert-do-lower-case --bert-model=/path/to/bert_large_cased --evalb-dir=./EVALB --train-path=./data/PTB/train.mrg --dev-path=./data/PTB/dev.mrg --test-path=./data/PTB/test.mrg --model-path-base models/SAPar.PTB.BERT --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15

# train bert+CatSA without POS tags
python SAPar_main.py train --use-bert --no-bert-do-lower-case --bert-model=/path/to/bert_large_cased --use-tags --evalb-dir=./EVALB --train-path=./data/PTB_POS/train.mrg --dev-path=./data/PTB_POS/dev.mrg --test-path=./data/PTB_POS/test.mrg --model-path-base models/SAPar.PTB.BERT.POS --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15

# train xlnet+CatSA without POS tags
python SAPar_main.py train --use-xlnet --no-bert-do-lower-case --xlnet-model=/path/to/XLNet_large_cased --evalb-dir=./EVALB --train-path=./data/PTB/train.mrg --dev-path=./data/PTB/dev.mrg --test-path=./data/PTB/test.mrg --model-path-base models/SAPar.PTB.XLNet --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15

# train xlnet+CatSA with POS tags
python SAPar_main.py train --use-xlnet --no-bert-do-lower-case --xlnet-model=/path/to/XLNet_large_cased --use-tags --evalb-dir=./EVALB --train-path=./data/PTB_POS/train.mrg --dev-path=./data/PTB_POS/dev.mrg --test-path=./data/PTB_POS/test.mrg --model-path-base models/SAPar.PTB.XLNet.POS --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15


# test English
python SAPar_main.py test --model-path-base=./path/to/the/model.pt --evalb-dir=./EVALB --test-path=./path/to/test.mrg --eval-batch-size=16

# test Arabic and Chinese
python SAPar_main.py test --model-path-base=./path/to/the/model.pt --evalb-dir=./EVALB_SPMRL --test-path=./path/to/test.mrg --eval-batch-size=16


# predict
# coming soon
