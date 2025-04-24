import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob

df = pd.read_csv("datasets/SMA_2_data.csv")

entities=["iphone" , "redmi" , "redmi note", "samsung", "galaxy" , "note"]

df = df.dropna(subset=['Tweet_Text'])

def get_setiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def sentiment_score(polarity):
    if polarity > 0 : return 'Positive'
    elif polarity < 0 : return 'Negative'
    else: return 'Neutral'

df['polarity'] = df['Tweet_Text'].apply(get_setiment)
df['sentiments'] = df['polarity'].apply(sentiment_score)

def find_brand(text):
    for brand in entities:
        if brand.lower() in text.lower():
            return brand
    return "Other"

df['brand'] = df['Tweet_Text'].apply(find_brand)

import seaborn as sns

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='brand', hue='sentiments')
plt.title('Sentiment Count per Brand')
plt.xlabel('Brand')
plt.ylabel('Count')
plt.legend(title='Sentiment')
plt.tight_layout()
plt.show()

sentiment_counts = df['sentiments'].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Overall Sentiment Distribution')
plt.axis('equal')
plt.show()
