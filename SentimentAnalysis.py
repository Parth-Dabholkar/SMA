import pandas as pd
import numpy as np
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"datasets/datafile2.csv")

def get_sentiment(text):
    blob = TextBlob(str(text))
    return blob.sentiment.polarity

df['polarity'] = df['text'].apply(get_sentiment)

def sentiment_score(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'
    
df['sentiment'] = df['polarity'].apply(sentiment_score)
count = df['sentiment'].value_counts().to_dict()

x = list(count.keys())
y = list(count.values())

plt.figure(figsize=(8,5))
sns.barplot(x=x, y=y, palette="mako")
plt.show()