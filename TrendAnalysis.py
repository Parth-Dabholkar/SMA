import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud

df = pd.read_csv('datasets/youtube_womens_safety_full_data(in).csv')
print(df.info())

new_df = df.dropna()
print(new_df.shape)

tfidf = TfidfVectorizer(max_features=10, stop_words='english')
tfidf_matrix = tfidf.fit_transform(new_df['title'])

top_10_words = tfidf.get_feature_names_out()
list_of_top_10_words = list(tfidf.get_feature_names_out())
print(f"Most trending words in dataset are: {top_10_words}")
print(list_of_top_10_words)

tfidf_scores = tfidf_matrix.toarray().sum(axis=0)

wordcloud_string = ' '.join(list_of_top_10_words)
print(wordcloud_string)

wordcloud = WordCloud(width=800, height=400, colormap="viridis", background_color="white").generate(wordcloud_string)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.tight_layout()
plt.title('Most Trending Words')
plt.show()

plt.figure(figsize=(8,5))
sns.barplot(x=top_10_words, y=tfidf_scores, palette="mako")
plt.xlabel('Words')
plt.ylabel('TF-IDF Scores')
plt.title('Trending Words in dataset', fontsize=16)
plt.tight_layout()
plt.show()

new_df['published_at'] = pd.to_datetime(new_df['published_at'])
new_df['Year'] = new_df['published_at'].dt.year
new_df['Month'] = new_df['published_at'].dt.month
new_df['Day'] = new_df['published_at'].dt.day

monthly_engagement = new_df.groupby(['Day','Month'])['likes'].sum()

plt.figure(figsize=(8,5))
monthly_engagement.plot(kind='line', marker='o', color='green')
plt.xlabel('Months in Number')
plt.ylabel('Like Count')
plt.tight_layout()
plt.show()

avg_monthly_engagement = new_df.groupby('Month')['likes'].mean()
plt.figure(figsize=(8,5))
avg_monthly_engagement.plot(kind='bar', color='skyblue')
plt.xlabel('Months in Number')
plt.ylabel('Average Like Count')
plt.tight_layout()
plt.show()