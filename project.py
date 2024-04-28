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
word_freq_sum = word_freq.sum(axis=0).sort_values(ascending=False)[:20]
plt.figure(figsize=(10, 6))
sns.barplot(x=word_freq_sum.values, y=word_freq_sum.index)
plt.title('Top 20 Most Frequent Words')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()
print("Train set shape:", X_train.shape, y_train.shape)
print("Test set shape:", X_test.shape, y_test.shape)

'''
print("Training Data")
print(training_data.isnull().sum())
print("\n")

print("Test Data")
print(test_data.isnull().sum())

training_data.dropna(subset=["Text_Tag"], inplace=True)
training_data["Text"] = training_data["Text"].astype(str)
training_data["Text_Tag"] = training_data["Text_Tag"].astype(str)

test_data["Text"] = test_data["Text"].astype(str)
test_data["Text_Tag"] = test_data["Text_Tag"].astype(str)
training_data["Text_Tag"] = training_data["Text_Tag"].apply(lambda x:x.split(','))

test_data["Text_Tag"] = test_data["Text_Tag"].apply(lambda x:x.split(','))
print("Training Data")
print(training_data.describe())
print(training_data.head())
print("\n")

print("Test Data")
print(test_data.describe())
print(test_data.head())
sns.histplot(data=training_data, x="Labels")
plt.show()
print(training_data[training_data["Labels"]==4]["Text"])
list_of_tags_train = [tag for tag_list in training_data["Text_Tag"] for tag in tag_list]
tags = " ".join(list_of_tags_train)
word_cloud_tags_train = WordCloud(width=1200,height=600).generate(tags)
plt.figure(figsize=(20,10))
plt.imshow(word_cloud_tags_train)
plt.title("Word Cloud for Tags- Training Dataset")
plt.axis("off")
plt.show()
list_of_tags_test = [tag for tag_list in test_data["Text_Tag"] for tag in tag_list]
tags = " ".join(list_of_tags_train)
word_cloud_tags_test = WordCloud(width=1200,height=600).generate(tags)
plt.figure(figsize=(20,10))
plt.imshow(word_cloud_tags_test)
plt.title("Word Cloud for Tags- Training Dataset")
plt.axis("off")
plt.show()
def porter_stemming(word):
    
    # a simple rules-based stemming algorithm.

    if word.endswith('sses'):
        word = word[:-2]
    elif word.endswith('ies'):
        word = word[:-2]
    elif word.endswith('ss'):
        word = word
    elif word.endswith('s'):
        word = word[:-1]

    if re.search(r'[aeiou].*[^aeiou]ed$', word):
        word = re.sub(r'ed$', '', word)
    elif re.search(r'[aeiou].*[^aeiou]ing$', word):
        word = re.sub(r'ing$', '', word)

    if re.search(r'[aeiou].*[^aeiou]y$', word):
        word = re.sub(r'(y|Y)$', 'i', word)

    if re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](ational|ation|ator)$', word):
        word = re.sub(r'(ational|ation|ator)$', 'ate', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](tional|tion|tor)$', word):
        word = re.sub(r'(tional|tion|tor)$', 'tion', word)

    if re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](enci|ence)$', word):
        word = re.sub(r'(enci|ence)$', 'ence', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](anci|ance)$', word):
        word = re.sub(r'(anci|ance)$', 'ance', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](izer|ization)$', word):
        word = re.sub(r'(izer|ization)$', 'ize', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](abli|able)$', word):
        word = re.sub(r'(abli|able)$', 'able', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](alli|al|ally)$', word):
        word = re.sub(r'(alli|al|ally)$', 'al', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](ently|ent)$', word):
        word = re.sub(r'(ently|ent)$', 'ent', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](ment)$', word):
        word = re.sub(r'(ment)$', 'ment', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](ement)$', word):
        word = re.sub(r'(ement)$', 'ement', word)
    
    if re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](ion)$', word):
        result = re.sub(r'(ion)$', '', word)
        if re.search(r'[^aeiou]*[aeiou]+.*[^aeiou]s$', result) or re.search(r'[^aeiou]*[aeiou]+.*[^aeiou]t$', result):
            word = result
    
    if re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](ate)$', word):
        word = re.sub(r'(ate)$', '', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](alize)$', word):
        word = re.sub(r'(alize)$', 'al', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](ify)$', word):
        word = re.sub(r'(ify)$', '', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](ic)$', word):
        word = re.sub(r'(ic)$', '', word)
    
    if re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](ance)$', word):
        word = re.sub(r'(ance)$', '', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](ence)$', word):
        word = re.sub(r'(ence)$', '', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](able)$', word):
        word = re.sub(r'(able)$', '', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](ment)$', word):
        word = re.sub(r'(ment)$', '', word)
    elif re.search(r'[^aeiou]*[aeiou]+.*[^aeiou](ion)$', word):
        result = re.sub(r'(ion)$', '', word)
        if re.search(r'[^aeiou]*[aeiou]+.*[^aeiou]t$', result):
            word = result
    
    if re.search(r'[^aeiou]*[aeiou]+.*[^aeiou]e$', word):
        result = re.sub(r'e$', '', word)
        if re.search(r'[^aeiou]*[aeiou]+.*[^aeiou]ll$', result):
            word = result
    
    return word
def simple_tokeniser_and_stemmer(text):
    text = re.sub(r'[^\w\s]', '', text) # removing spl chars
    tokens = re.findall(r'\b\w+\b', text) # finding all "words"
    tokens = [word.lower() for word in tokens] #lowercase
    tokens = [porter_stemming(word) for word in tokens]
    return tokens

#apply stemming for text and tags.
training_data["Text_tokens"] = training_data["Text"].apply(simple_tokeniser_and_stemmer)
training_data['Text_Tag_tokens'] = training_data['Text_Tag'].apply(lambda x: [porter_stemming(word) for word in x])

test_data["Text_tokens"] = test_data["Text"].apply(simple_tokeniser_and_stemmer)
test_data['Text_Tag_tokens'] = test_data['Text_Tag'].apply(lambda x: [porter_stemming(word) for word in x])
list_of_words_token_train = [word for word_list in training_data["Text_tokens"] for word in word_list]
words_train = " ".join(list_of_words_token_train)
word_cloud_words_train = WordCloud(width=1200,height=600).generate(words_train)
plt.figure(figsize=(20,10))
plt.imshow(word_cloud_words_train)
plt.title("Words after Stemming- Training Dataset")
plt.axis("off")
plt.show()
list_of_words_test = [word for word_list in test_data["Text_tokens"] for word in word_list]
words = " ".join(list_of_words_test)
word_cloud_words_test = WordCloud(width=1200,height=600).generate(words)
plt.figure(figsize=(20,10))
plt.imshow(word_cloud_words_test)
plt.title("Words after Stemming- Test Dataset")
plt.axis("off")
plt.show()
list_of_tags_token_train = [tag for tag_list in training_data["Text_Tag_tokens"] for tag in tag_list]
tags_token_train = " ".join(list_of_tags_token_train)
word_cloud_tags_token_train = WordCloud(width=1200,height=600).generate(tags_token_train)
plt.figure(figsize=(20,10))
plt.imshow(word_cloud_tags_token_train)
plt.title("Tags after Stemming- Training Dataset")
plt.axis("off")
plt.show()
list_of_tags_token_test = [tag for tag_list in test_data["Text_Tag_tokens"] for tag in tag_list]
tags_token_test = " ".join(list_of_tags_token_test)
word_cloud_tags_token_test = WordCloud(width=1200,height=600).generate(tags_token_test)
plt.figure(figsize=(20,10))
plt.imshow(word_cloud_tags_token_test)
plt.title("Tags after Stemming- Test Dataset")
plt.axis("off")
plt.show()
'''