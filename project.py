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
from nltk.corpus import brown
import keras
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns

corpus = brown.sents() #corpus for word2vec model, captures sentence level relations

data = pd.read_csv("Projdataset.csv")
data.dropna(inplace=True)

#processing data so that the tags are now a list of tags, ready for further preprocessing
data['Text_Tag'] = data['Text_Tag'].str.replace('_', ' ')
data['Text_Tag'] = data['Text_Tag'].str.split(',')

stop_words = set(stopwords.words('english'))

def preprocess_labels(label): #since this is multi-class classification, we need to one-hot encode the labels.
    result = np.array([0, 0, 0, 0, 0])
    result[label-1] = 1
    return result.T #transpose to get a row vector- needed that way for the ml model

data['Labels'] = data['Labels'].apply(preprocess_labels)

def tokenize_and_stem(text):
    text = re.sub(r'[^\w\s]', '', text)   #remove non-alphanumeric characters
    text = text.lower() #lowercase
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in stop_words] #tokenise and remove stopwords
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens] #lemmatise
    return tokens

data['Text_tokens'] = data['Text'].apply(tokenize_and_stem) #apply tokenisation and stemming
data['Text_Tag_tokens'] = data['Text_Tag'].apply(lambda tags: [tokenize_and_stem(tag) for tag in tags])

words_text = " ".join([word for word_list in data["Text_tokens"] for word in word_list]) #word cloud for text
word_cloud_text = WordCloud(width=1200, height=600).generate(words_text)
plt.figure(figsize=(20, 10))
plt.imshow(word_cloud_text)
plt.title("Words - Text")
plt.axis("off")
plt.show()

words_tags = " ".join([tag for tag_list in data["Text_Tag_tokens"] for each_tag in tag_list for tag in each_tag]) #word cloud for tags
word_cloud_tags = WordCloud(width=1200, height=600).generate(words_tags)
plt.figure(figsize=(20, 10))
plt.imshow(word_cloud_tags)
plt.title("Words - Text_Tag")
plt.axis("off")
plt.show()

#graph to show the most common words
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

text_token_lengths = data['Text_tokens'].apply(len) #graph to show distribution of token lengths
plt.figure(figsize=(10, 6))
sns.scatterplot(text_token_lengths.value_counts())
plt.title('Distribution of Text Token Lengths')
plt.xlabel('Token Length')
plt.ylabel('Frequency')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(data['Text_tokens'], data['Labels']) #train test split

vec_size = 100 #we will embed words into a 100-dimensional vector space.

embedding_model = Word2Vec(sentences=corpus, vector_size=vec_size, window=5, min_count=1, workers=4)
embedding_model.build_vocab(x_train, update=True)

def embed_and_pad_words(text_tokens, embedding_model):
    max_length = text_token_lengths.value_counts().index.max() + 1 #pad sentences till this length.
    padding = np.zeros((1,vec_size))
    embedded_sentences = np.empty((1,0))
    for sentence in text_tokens:
        embedded_words = np.empty((1,0))
        for word in sentence:
            if word in embedding_model.wv:
                embedded_words = np.append(embedded_words, np.column_stack(embedding_model.wv[word]), axis=1) # embed the word using the embedding model
            else:
                embedded_words = np.append(embedded_words, np.zeros((1,vec_size)), axis=1) # if the word is not in the embedding model, use zero vector
        for i in range(max_length - len(embedded_words)+1):
            embedded_words = np.concatenate((embedded_words, padding), axis=1) # pad the words with zero vectors
        embedded_sentences = np.append(embedded_sentences, embedded_words, axis=1) # stack the padded words as column vectors
    return embedded_sentences # stack the embedded sentences as column vectors

x_train = embed_and_pad_words(x_train, embedding_model) #embedding and padding the sentences
print("Train set shape:", x_train.shape, y_train.shape)

x_test = embed_and_pad_words(x_test, embedding_model)
print("Test shape:", x_test.shape, y_test.shape)

#the ml model needs input and output to be a list.
y_train = y_train.tolist().T
y_test = y_test.tolist().T

model = keras.Sequential()
#model.add(keras.layers.Input(shape=()))