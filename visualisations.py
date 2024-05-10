#importing data for visualisation
from dataset_handling import get_viz_data

#basic libraries for data handling and visualisation
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

#count vectorizer for one of the visualisations
from sklearn.feature_extraction.text import CountVectorizer

#word cloud for visualization
from wordcloud import WordCloud as wc

#loading the data
X_TRAIN_VIZ, X_VAL_VIZ, X_TEST_VIZ, Y_TRAIN_VIZ, Y_VAL_VIZ, Y_TEST_VIZ = get_viz_data()

sns.histplot(Y_TRAIN_VIZ, color='blue', label='Training set')
sns.histplot(Y_VAL_VIZ, color='green', label='Validation set')
sns.histplot(Y_TEST_VIZ, color='red', label='Test set')
plt.legend()
plt.title('Distribution of labels')
plt.show()

words_text = " ".join([word for word in X_TRAIN_VIZ]) + " " + " ".join([word for word in X_VAL_VIZ]) + " " + " ".join([word for word in X_TEST_VIZ]) # combine all text data

word_cloud_text = wc(width=1200, height=600).generate(words_text)
plt.figure(figsize=(20, 10))
plt.imshow(word_cloud_text)
plt.title("Common Words in the Dataset")
plt.axis("off")
plt.show()

#graph to show the most common words

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(X_TRAIN_VIZ)

word_freq = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
word_freq_sum = word_freq.sum(axis=0).sort_values(ascending=False)[:20]

plt.figure(figsize=(10, 6))
sns.barplot(x=word_freq_sum.values, y=word_freq_sum.index, palette='inferno')
plt.title('Top 20 Most Frequent Words in the Training Set (excluding stopwords)')
plt.xlabel('Frequency')
plt.ylabel('Word')
plt.show()