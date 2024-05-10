#module for dataset importing.
from datasets import load_dataset

#basic libraries
import pandas as pd
import numpy as np
import re

#modules for text preprocessing
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences

#wordnet lemmatizer
from nltk.stem import WordNetLemmatizer

#stopwords (we will remove these from the text data)
from nltk.corpus import stopwords

#sentence transformers for the pretrained embedding model
from sentence_transformers import SentenceTransformer

#the main callable functions for other python files.
#to use these functions, type: from dataset_handling import <function_name>
#returns, in order: X_TRAIN(_VIZ), X_VAL(_VIZ), X_TEST(_VIZ), Y_TRAIN(_VIZ), Y_VAL(_VIZ), Y_TEST(_VIZ)

stopwords = set(stopwords.words('english')) #set up stopwords

def get_viz_data():
    return do_basics()

def get_embedding_data():
    return do_embedding()

#below are the function declarations.

def do_basics():
    #function to do basic preprocessing, up to the point of embedding.
    #returns data ready for visualisation.

    liar = load_dataset('liar') #load the liar dataset from the datasets library. same as what we had from kaggle.

    #conveniently pre-split into train, test, and val sets! we'll just use the statement and label columns.
    train = liar['train'].to_pandas()[['label', 'statement']]
    test = liar['test'].to_pandas()[['label', 'statement']]
    val = liar['validation'].to_pandas()[['label', 'statement']]

    train.dropna(subset='statement', inplace=True) #drop null values
    test.dropna(subset='statement', inplace=True)
    val.dropna(subset='statement', inplace=True)
    
    #drop duplicates
    train.drop_duplicates(subset='statement', keep='first', inplace=True)
    test.drop_duplicates(subset='statement', keep='first', inplace=True)
    val.drop_duplicates(subset='statement', keep='first', inplace=True)

    train['statement'] = train['statement'].apply(tokenise_lemmatize)
    test['statement'] = test['statement'].apply(tokenise_lemmatize)
    val['statement'] = val['statement'].apply(tokenise_lemmatize)
    
    #storing these variables.
    #to use them in other notebooks, type: from dataset_handling import <variable_name>

    X_TRAIN_VIZ = train['statement']
    X_VAL_VIZ = val['statement']
    X_TEST_VIZ = test['statement']
    
    Y_TRAIN_VIZ = train['label']
    Y_VAL_VIZ = val['label']
    Y_TEST_VIZ = test['label']

    X_TRAIN_VIZ = X_TRAIN_VIZ.astype(str)
    X_VAL_VIZ = X_VAL_VIZ.astype(str)
    X_TEST_VIZ = X_TEST_VIZ.astype(str)

    return X_TRAIN_VIZ, X_VAL_VIZ, X_TEST_VIZ, Y_TRAIN_VIZ, Y_VAL_VIZ, Y_TEST_VIZ

def do_embedding():
    #function to do basic preprocessing, and then embedding, padding, etc.

    X_TRAIN, X_VAL, X_TEST, Y_TRAIN, Y_VAL, Y_TEST = do_basics() #first do the basics
    
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2') #the embedding model- going with MiniLM for now coz it's a popular choice
    
    #does embedding and padding from sentences conveniently!
    X_TRAIN = embedding_model.encode(X_TRAIN.values, show_progress_bar=True, convert_to_numpy=True, output_value='token_embeddings')
    X_VAL = embedding_model.encode(X_VAL.values, show_progress_bar=True, convert_to_numpy=True, output_value='token_embeddings')
    X_TEST = embedding_model.encode(X_TEST.values, show_progress_bar=True, convert_to_numpy=True, output_value='token_embeddings')
    
    #pad it just in case lol
    X_TRAIN = pad_sequences(X_TRAIN, padding='post', dtype='float32')
    X_VAL = pad_sequences(X_VAL, padding='post', dtype='float32')
    X_TEST = pad_sequences(X_TEST, padding='post', dtype='float32')
    
    #one-hot encode the labels
    Y_TRAIN = to_categorical(Y_TRAIN)
    Y_VAL = to_categorical(Y_VAL)
    Y_TEST = to_categorical(Y_TEST)

    return X_TRAIN, X_VAL, X_TEST, Y_TRAIN, Y_VAL, Y_TEST

def tokenise_lemmatize(text):
    
    text = re.sub(r'[^\w\s]', '', text) #remove punctuation
    text = text.lower() #lowercase

    tokens = text.split() #tokenise

    tokens = [token for token in tokens if token not in stopwords] #remove stopwords
    
    lemmatizer = WordNetLemmatizer()
    tokens = " ".join(lemmatizer.lemmatize(token) for token in tokens) #lemmatise
    return tokens