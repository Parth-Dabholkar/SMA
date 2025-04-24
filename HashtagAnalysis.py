import re
from wordcloud import WordCloud
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def removeHashtags(text):
    return re.findall(r'#\w+', str(text))

df = pd.read_csv('datasets/datafile1.csv')

df['temp_hashtags'] = df['text'].apply(removeHashtags)
print(df['temp_hashtags'])

all_hashtags = [tag for tags in df['temp_hashtags'] for tag in tags]
print(all_hashtags)

all_hashtags_string = ' '.join(all_hashtags)
print(all_hashtags_string)

hashtag_df = pd.DataFrame(all_hashtags, columns=['hashtags'])
print(hashtag_df)
print(hashtag_df.value_counts())

hash_count = hashtag_df.value_counts().to_dict()
y_val = hash_count.values()
x_val = []
[x_val.append(i) for i in all_hashtags if i not in x_val]
print(x_val)

wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='plasma').generate(all_hashtags_string)

plt.figure(figsize=(10,5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,5))
sns.barplot(x=x_val, y=y_val, palette='mako')
plt.xlabel('Hashtag words')
plt.ylabel('Count')
plt.tight_layout()
plt.show()
