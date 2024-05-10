from tensorflow.keras.layers import Bidirectional, LSTM, SimpleRNN

def BiLSTM(units, regularizer, return_sequences=False):
    return Bidirectional(LSTM(units, kernel_regularizer=regularizer, dropout=0.5, recurrent_dropout=0.4, return_sequences=return_sequences))

def RNN(units, regularizer, return_sequences=False):
    return SimpleRNN(units, kernel_regularizer=regularizer, dropout=0.5, recurrent_dropout=0.4, return_sequences=return_sequences)