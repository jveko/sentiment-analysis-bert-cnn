run with

```bash
python main.py --model=bert_cnn --pretrained_model google-bert/bert-base-multilingual-uncased  --save-model --report
```

```bash
python main.py --model=bert_cnn --pretrained_model google-bert/bert-base-multilingual-uncased --dataset datasets/tweets_labeled_lexicon_cleaned.json --save-model --report 
```

```bash
python main.py --model=bert_cnn --pretrained_model google-bert/bert-base-multilingual-uncased --dataset datasets/tweets_labeled_lexicon_cleaned.json --save-model --report --neutral
```

```bash
python main.py --models bert_gru bert_cnn bert_lstm  --pretrained_model google-bert/bert-base-multilingual-cased --dataset datasets/tweets_ppkm_clean.json --save-model --report --neutral
```