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
python main.py --models bert_cnn bert_lstm bert_gru  --pretrained_model google-bert/bert-base-multilingual-cased --dataset datasets/tweets_ppkm_clean.json --save-model --report --neutral
```
```bash
python main.py --models bert_cnn bert_lstm bert_gru  --pretrained_model google-bert/bert-base-multilingual-cased --dataset datasets/tweets_ppkm_clean.json --report
```

```bash
python main.py --models bert bert_cnn bert_lstm bert_gru  --pretrained_model indolem/indobert-base-uncased --dataset datasets/tweets_labeled_balanced.json --report
```

```bash
accelerate launch --config_file ./config.yaml  main.py --models bert bert_cnn bert_lstm bert_gru  --pretrained_model indolem/indobert-base-uncased --dataset datasets/tweets_distilbert_labeled_balanced.json
```
