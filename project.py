import pandas as pd
import numpy as np
import re
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk

data = pd.read_csv("Projdataset.csv")
data.dropna(inplace=True)

def tokenize_and_stem(text):
    text = re.sub(r'[^\w\s]', '', text)  
    tokens = re.findall(r'\b\w+\b', text)  
    tokens = [word.lower() for word in tokens]  
    return tokens

data['Text_tokens'] = data['Text'].apply(tokenize_and_stem)
data['Text_Tag_tokens'] = data['Text_Tag'].apply(tokenize_and_stem)

word2vec_model = Word2Vec(sentences=data['Text_tokens'].tolist() + data['Text_Tag_tokens'].tolist(), 
                          vector_size=100, window=5, min_count=1, workers=4)

words_text = " ".join([word for word_list in data["Text_tokens"] for word in word_list])
word_cloud_text = WordCloud(width=1200, height=600).generate(words_text)
plt.figure(figsize=(20, 10))
plt.imshow(word_cloud_text)
plt.title("Words - Text")
plt.axis("off")
plt.show()

words_tags = " ".join([word for word_list in data["Text_Tag_tokens"] for word in word_list])
word_cloud_tags = WordCloud(width=1200, height=600).generate(words_tags)
plt.figure(figsize=(20, 10))
plt.imshow(word_cloud_tags)
plt.title("Words - Text_Tag")
plt.axis("off")
plt.show()

X = data.drop(columns=['Labels'])  
y = data['Labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
X = vectorizer.fit_transform(data['Text_tokens'])

word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

stop_words = set(stopwords.words('english'))
filtered_words = [word for word in word_freq.columns if word not in stop_words]

word_freq_filtered = word_freq[filtered_words]

word_freq_sum = word_freq_filtered.sum(axis=0).sort_values(ascending=False)[:20]
plt.figure(figsize=(10, 6))
sns.barplot(x=word_freq_sum.values, y=word_freq_sum.index, palette='inferno')
plt.title('Top 20 Most Frequent Words (Excluding Stopwords)')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()

print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)
