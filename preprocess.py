import string

import pandas as pd
import re
import emoji


def load_and_clean_data(path, target_names):
    def clean_text(text):
        # Remove emojis
        text = emoji.replace_emoji(text)
        # Remove newlines
        text = re.sub(r'\n', ' ', text)
        # Remove tabs
        text = re.sub(r'\t', ' ', text)
        # # Remove multiple periods with a single one
        # text = re.sub(r'\.{2,}', '.', text)
        # # Remove multiple exclamation marks with a single one
        # text = re.sub(r'\!{2,}', '!', text)
        # Replace multiple spaces with a single space
        text = ' '.join(text.split())
        return text

    def preprocess(text):
        new_text = []
        for t in text.split(" "):
            t = '@user' if t.startswith('@') and len(t) > 1 else t
            t = 'http' if t.startswith('http') else t
            new_text.append(t)
        return " ".join(new_text)

    def convert_sentiment(sentiment):
        sentiment = sentiment.lower()
        if len(target_names) == 3:
            if sentiment == 'positive':
                return 2
            elif sentiment == 'neutral':
                return 1
            elif sentiment == 'negative':
                return 0
        else:
            if sentiment == 'positive':
                return 1
            elif sentiment == 'negative':
                return 0

    df = pd.read_json(path)
    df['sentiment_numeric'] = df['sentiment'].apply(convert_sentiment)
    df['raw_content'] = df['raw_content'].apply(preprocess)
    df['raw_content'] = df['raw_content'].apply(clean_text)

    return df
