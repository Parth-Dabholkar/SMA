import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import seaborn as sns

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def get_score(polarity):
    if polarity > 0 : return 'Positive'
    elif polarity < 0 : return 'Negative'
    else: return 'Neutral'

df = pd.read_csv('datasets/SMA 1-3 Dataset.csv')

df['sentiment'] = df['Content'].apply(get_sentiment)
df['polarity'] = df['sentiment'].apply(get_score)

type_count = df['Content_Type'].value_counts().to_dict()

x_val = []
[x_val.append(i) for i in df['Content_Type'] if i not in x_val]
print(x_val)
y_val = list(type_count.values())
print(y_val)

plt.figure(figsize=(10, 5))
sns.barplot(x=x_val, y=y_val, palette='mako')
plt.show()

df['engagement_rate'] = (df['Likes'] + df['Shares'] + df['Comments'])/df['Followers']

plt.figure(figsize=(10, 5))
sns.barplot(data=df, x='Content_Type', y='engagement_rate', palette='mako')
plt.show()

plt.figure(figsize=(10, 5))
sns.countplot(data=df, x='Content_Type', hue='polarity')
plt.show()