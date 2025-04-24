#Libraries import kro
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

df = pd.read_csv(r"datasets/youtube_womens_safety_full_data(in).csv")
df.head()

new_df = df.dropna()
new_df.shape

#TF-IDF wapro for keyword extraction
#Koi bhi text column pe TF-IDF vectorizer wapro
tfidf = TfidfVectorizer(max_features=10, stop_words='english')  # Limit to top 10 keywords
tfidf_matrix = tfidf.fit_transform(new_df['title'])

#Get the top words in list format from text columns
key_words_list_title = list(tfidf.get_feature_names_out())
print(f"Top words in title column: {key_words_list_title}")

#Combine all words in the title column to string for wordcloud
all_text = ' '.join(key_words_list_title)

# Create a WordCloud object
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(all_text)

# Display the WordCloud
plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title("Word Cloud of Cleaned YouTube Transcripts", fontsize=16)
plt.show()

#LDA WAPRO
#Apply LDA to find 3 topics
lda = LatentDirichletAllocation(n_components=3, random_state=42)
lda.fit(tfidf_matrix)

#Print top words for each topic
words = tfidf.get_feature_names_out()

for i, topic in enumerate(lda.components_):
    print(f"\nTopic #{i + 1}:")
    top_words = topic.argsort()[-10:][::-1]
    print(" ".join(words[i] for i in top_words))
    all_new_text = " ".join(words[i] for i in top_words)

wordcloud_2 = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(all_new_text)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud_2, interpolation='bilinear')
plt.title("Word Cloud of Topics from LDA", fontsize=16)
plt.show()

keywords = tfidf.get_feature_names_out()
scores = tfidf_matrix.toarray().sum(axis=0)

plt.figure(figsize=(8, 5))
sns.barplot(x=scores, y=keywords, palette='mako')
plt.title('Top Keywords in Video Titles')
plt.xlabel('TF-IDF Score')
plt.ylabel('Keywords')
plt.tight_layout()
plt.show()