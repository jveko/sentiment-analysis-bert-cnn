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