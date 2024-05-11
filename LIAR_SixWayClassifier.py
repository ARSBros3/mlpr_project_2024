#import embedded data
from dataset_handling import get_embedding_data

#basics
import numpy as np
import matplotlib.pyplot as plt

#keras functions for the model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Masking, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Embedding
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

#metrics for model evaluation
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#load the data
X_TRAIN, X_VAL, X_TEST, Y_TRAIN, Y_VAL, Y_TEST = get_embedding_data()

model = Sequential()

model.add(Input(shape=(X_TRAIN.shape[1],X_TRAIN.shape[2]))) #input layer

model.add(Masking(mask_value=0.0)) #masking layer- to ignore the padding

model.add(Bidirectional(LSTM(192, kernel_regularizer='l2', dropout=0.5, recurrent_dropout=0.4))) #bidirectional LSTM layer, dropout and regularisation to prevent overfitting

model.add(Dense(64, activation='tanh')) #feature selection layer- could be a positive or negative sentiment, so went with tanh

model.add(BatchNormalization()) #speed!

model.add(Dropout(0.5)) #dropout to prevent overfitting

model.add(Dense(32, activation='tanh')) #same logic, narrowing down the features

model.add(BatchNormalization()) #speed!

model.add(Dropout(0.5)) #dropout to prevent overfitting

model.add(Dense(6, activation='softmax')) #output layer- 6 classes, so softmax activation to ensure the output adds up to 1 (like probabilities)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) #went with popular choices for the optimizer and loss function, got it from the papers

print(model.summary()) #always helps to print the summary.

model.fit(X_TRAIN, Y_TRAIN, validation_data=(X_VAL, Y_VAL), epochs=50, callbacks=[ReduceLROnPlateau(monitor='val_accuracy', patience=2, factor=0.1, verbose=1), EarlyStopping(monitor='val_accuracy', patience=10, verbose=1)])

'''
fit for 50 epochs. validation set provided.
two interesting callback functions-
ReduceLROnPlateau reduces the learning rate by <factor> if the <monitor> metric doesn't improve after <patience> number of epochs
EarlyStopping stops the training if the <monitor> metric doesn't improve after <patience> number of epochs.
'''

y_pred = model.predict(X_TEST)
y_pred = np.argmax(y_pred, axis=1) #extracts the prediction from the one-hot encoded output
y_true = np.argmax(Y_TEST, axis=1)

print(classification_report(y_true, y_pred)) #print general stats

ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred)).plot() #plot confusion matrix
plt.show()