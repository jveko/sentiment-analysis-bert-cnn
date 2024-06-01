import string

import pandas as pd
import re
import emoji


def load_and_clean_data(path):
    def clean_text(text):
        # Remove emojis
        text = emoji.replace_emoji(text)
        # # Remove URLs
        # text = re.sub(r'https?://\S+', ' ', text)
        text = re.sub('$URL', ' ', text)
        # Normalize ellipses
        text = re.sub(r'\.{2,}', ' . ', text)
        # Handle repeated punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        # Ensure spacing around punctuation
        text = re.sub(f'([{string.punctuation}])', r' \1 ', text)
        text = re.sub(r'\s{2,}', ' ', text)
        # Remove punctuation (.,?)
        # text = re.sub(r'[.,?]', ' ', text)
        # Remove quotes ('"“)
        # text = re.sub(r'[\'\"“]', ' ', text)
        # Remove mentions
        text = re.sub(r'@[A-Za-z0-9_]+', ' ', text)
        # Remove hashtags
        text = re.sub(r'#[A-Za-z0-9_]+', ' ', text)
        # Remove newlines
        text = re.sub(r'\n', ' ', text)
        # Remove tabs
        text = re.sub(r'\t', ' ', text)
        # Replace multiple spaces with a single space
        text = ' '.join(text.split())
        # Convert text to lowercase
        text = text.lower()
        return text

    def has_url(text):
        match = re.search(r'https?://\S+', text)
        if match:
            return True
        return False

    def convert_sentiment(sentiment):
        sentiment = sentiment.lower()
        if sentiment == 'neutral':
            return 2
        elif sentiment == 'positive':
            return 1
        elif sentiment == 'negative':
            return 0

    # Define a function to split hashtags
    def split_hashtag(hashtag):
        words = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', hashtag)
        return ' '.join(words)

    df = pd.read_json(path)
    df['sentiment_numeric'] = df['sentiment'].apply(convert_sentiment)
    # df['has_url'] = df['raw_content'].apply(has_url)
    # df.drop(df[df['has_url']].index, inplace=True)
    # df['raw_content'] = df['raw_content'].apply(clean_text)
    df.drop_duplicates(subset='raw_content')

    return df
