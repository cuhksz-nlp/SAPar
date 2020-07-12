# prepare evaluation
cd EVALB
make
cd ..

# train xlnet+CatSA without POS tag
python main.py train --use-xlnet --no-bert-do-lower-case --xlnet-model=/path/to/XLNet_large_cased --evalb-dir=./EVALB --train-path=./tmp_data/train.mrg --dev-path=./tmp_data/dev.mrg --model-path-base models/test --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15

# train xlnet+CatSA with POS tag
python main.py train --use-xlnet --no-bert-do-lower-case --xlnet-model=/path/to/XLNet_large_cased --use-tags --evalb-dir=./EVALB --train-path=./tmp_data/train.mrg --dev-path=./tmp_data/dev.mrg --model-path-base models/test --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15

# train bert+CatSA without POS tag
python main.py train --use-bert --no-bert-do-lower-case --bert-model=/path/to/bert_large_cased --evalb-dir=./EVALB --train-path=./tmp_data/train.mrg --dev-path=./tmp_data/dev.mrg --model-path-base models/test --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15

# train bert+CatSA without POS tag
python main.py train --use-bert --no-bert-do-lower-case --bert-model=/path/to/bert_large_cased --use-tags --evalb-dir=./EVALB --train-path=./tmp_data/train.mrg --dev-path=./tmp_data/dev.mrg --model-path-base models/test --num-layers 3 --learning-rate 1e-5 --batch-size 16 --eval-batch-size 16 --subbatch-max-tokens 1200 --ngram-threshold=0 --ngram-freq-threshold=2 --ngram=5 --max-len-train=300 --max-len-dev=300 --patients=15
