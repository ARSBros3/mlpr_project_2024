import pandas as pd
import numpy as np
import re
import gensim
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import nltk
import keras
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

nltk.download('wordnet') #for lemmatizer

data = pd.read_csv("Projdataset.csv")
data.dropna(inplace=True)

#processing data so that the tags are now a list of tags, ready for further preprocessing
data['Text_Tag'] = data['Text_Tag'].str.replace('_', ' ')
data['Text_Tag'] = data['Text_Tag'].str.split(',')

stop_words = set(stopwords.words('english'))

def tokenize_and_stem(text):
    text = re.sub(r'[^\w\s]', '', text)   #remove non-alphanumeric characters
    text = text.lower() #lowercase
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words] #tokenise and remove stopwords
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens] #lemmatise
    return tokens

data['Text_tokens'] = data['Text'].apply(tokenize_and_stem)
data['Text_Tag_tokens'] = data['Text_Tag'].apply(lambda tags: [tokenize_and_stem(tag) for tag in tags])

words_text = " ".join([word for word_list in data["Text_tokens"] for word in word_list])
word_cloud_text = WordCloud(width=1200, height=600).generate(words_text)
plt.figure(figsize=(20, 10))
plt.imshow(word_cloud_text)
plt.title("Words - Text")
plt.axis("off")
plt.show()

words_tags = " ".join([tag for tag_list in data["Text_Tag_tokens"] for each_tag in tag_list for tag in each_tag])
word_cloud_tags = WordCloud(width=1200, height=600).generate(words_tags)
plt.figure(figsize=(20, 10))
plt.imshow(word_cloud_tags)
plt.title("Words - Text_Tag")
plt.axis("off")
plt.show()

vec_size = 100 #we will embed words into a 100-dimensional vector space.

embedding_model = Word2Vec(sentences=data['Text_tokens'], vector_size=vec_size, window=5, min_count=1, workers=4)

def embed_words(text_tokens, embedding_model): #embedding words using word2vec
    embedded_words = []
    for tokens in text_tokens:
        embedded_tokens = []
        for token in tokens:
            if token in embedding_model.wv:
                embedded_tokens.append(embedding_model.wv[token])
        embedded_words.append(embedded_tokens)
    return embedded_words

data["embedded_text"] = embed_words(data['Text_tokens'], embedding_model)

text_token_lengths = data['Text_tokens'].apply(len)
plt.figure(figsize=(10, 6))
sns.scatterplot(text_token_lengths.value_counts())
plt.title('Distribution of Text Token Lengths')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(data['embedded_text'], data['Labels']) #train test split

vectorizer = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
X = vectorizer.fit_transform(data['Text_tokens'])

word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

word_freq_sum = word_freq.sum(axis=0).sort_values(ascending=False)[:20]
plt.figure(figsize=(10, 6))
sns.barplot(x=word_freq_sum.values, y=word_freq_sum.index, palette='inferno')
plt.title('Top 20 Most Frequent Words (Excluding Stopwords)')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()

print("Test shape:", x_test.shape, y_test.shape)
print("Train set shape:", x_train.shape, y_train.shape)

