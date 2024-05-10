#import embedded data
from dataset_handling import get_embedding_data

from models import BiLSTM, RNN

#basics
import numpy as np
import matplotlib.pyplot as plt

#keras functions for the model building
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Masking, Bidirectional, LSTM, Dense, Dropout, BatchNormalization, Embedding
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import activations, regularizers, optimizers, losses, metrics

#metrics for model evaluation
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

#the one and only class you will ever need for all model building, and evaluation.
class LIAR_SixWayClassifier:
    def __init__(self, data_shape, **build_kwargs):

        self.input_dim = data_shape #required parameter. pass the shape of first element of data.

        self.n_classes = 6

        self.model = self.build_model(**build_kwargs)

    def build_model(self, **build_kwargs):
        
        self.build_kwargs = build_kwargs

        self.available_algorithms = ['bi-lstm', 'rnn']

        #kwarg for the type of algorithm to use in the model. default is 'bi-lstm'.
        self.algorithm = build_kwargs.get('algorithm', 'bi-lstm')
        if self.algorithm not in self.available_algorithms:
            raise ValueError(f"{self.algorithm} is either not a valid algorithm or not available yet. Choose from {self.available_algorithms}.")
            
        #kwarg for the number of initial features. for example, if this is 64, the first feature selection layer will have 64 nodes.
        self.initial_features = build_kwargs.get('initial_features', 64)
        if self.initial_features < 0 or not isinstance(self.initial_features, int):
            raise ValueError('initial_features must be a positive integer.')
        
        #kwarg for the activation function to use in the feature selection layer. default is tanh.
        self.activation_fn = build_kwargs.get('activation_fn', 'tanh')
        if activations.get(self.activation_fn) is None:
            raise ValueError(f"{self.activation_fn} is not a valid Keras activation function")
        
        self.regularizer = build_kwargs.get('regularizer', 'l2')
        if regularizers.get(self.regularizer) is None:
            raise ValueError(f"{self.regularizer} is not a valid Keras regularizer.")
        
        #kwarg for the number of feature selection layers. default is 2.
        self.feature_layers = build_kwargs.get('feature_layers', 2)
        if self.feature_layers < 0:
            raise ValueError('feature_layers must be a positive integer.')
        
        #kwarg for the number of units in the feature extraction layer (for now, BiLSTM). default is 96.
        self.units = build_kwargs.get('units', 96)
        if self.units < 0 or not isinstance(self.units, int):
            raise ValueError('units must be a positive integer.')
        
        #kwarg for whether or not to add dropout layers. prevents overfitting at the cost of information loss. default is False.
        self.dropout_layers = build_kwargs.get('dropout_layers', False)
        if not isinstance(self.dropout_layers, bool):
            raise ValueError('dropout_layers must be a boolean.')
        
        #in case the user specifies dropout_rate without dropout_layers being True.
        if 'dropout_layers' not in build_kwargs and 'dropout_rate' in build_kwargs:
            raise ValueError('dropout_rate cannot be specified without dropout_layers being True.')

        #if dropout_layers is True, this kwarg is for the dropout rate. default is 0.5.
        if self.dropout_layers:
            self.dropout_rate = build_kwargs.get('dropout_rate', 0.5)
            if self.dropout_rate < 0 or self.dropout_rate > 1:
                raise ValueError('dropout_rate must be a float between 0 and 1.')

        model = Sequential()

        model.add(Input(shape=(int(dim) for dim in self.input_dim)))

        model.add(Masking(mask_value=0.0))

        if self.algorithm == 'bi-lstm':
            model.add(BiLSTM(self.units, self.regularizer, False))
        elif self.algorithm == 'rnn':
            model.add(RNN(self.units, self.regularizer, False))

        for i in range(1, self.feature_layers+1):
            model.add(Dense(int(self.initial_features/(2**i)), activation=self.activation_fn))
            if self.dropout_layers:
                model.add(Dropout(self.dropout_rate))
            model.add(BatchNormalization())

        model.add(Dense(self.n_classes, activation='softmax'))

        print(model.summary())

        return model
    
    def compile_model(self, **compile_kwargs):

        self.compile_kwargs = compile_kwargs

        #kwarg for the optimizer to use. default is adam.
        self.optimizer = compile_kwargs.get('optimizer', 'adam')
        if optimizers.get(self.optimizer) is None:
            raise ValueError(f"{self.optimizer} is not a valid Keras optimizer.")
        
        #kwarg for the loss function to use. default is categorical_crossentropy.
        self.loss = compile_kwargs.get('loss', 'categorical_crossentropy')
        if losses.get(self.loss) is None:
            raise ValueError(f"{self.loss} is not a valid Keras loss function.")
        
        #kwarg for the metrics to use. default is accuracy.
        self.metrics = compile_kwargs.get('metrics', ['accuracy'])
        for metric in self.metrics:
            if metrics.get(metric) is None:
                raise ValueError(f"{metric} is not a valid Keras metric.")

        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)

    def train_model(self, X_train, y_train, X_val, y_val, **train_kwargs):

        self.train_kwargs = train_kwargs

        #kwarg for the batch size. default is 64.
        self.batch_size = train_kwargs.get('batch_size', 64)
        if self.batch_size < 0 or not isinstance(self.batch_size, int):
            raise ValueError('batch_size must be a positive integer.')
        
        #kwarg for the number of epochs. default is 100.
        self.epochs = train_kwargs.get('epochs', 100)
        if self.epochs < 0 or not isinstance(self.epochs, int):
            raise ValueError('epochs must be a positive integer.')
        
        #to store callbacks, if applicable.
        self.callbacks = []

        #kwarg for whether or not to reduce the learning rate if . default is True.
        self.reduce_lr = train_kwargs.get('reduce_lr', True)
        if not isinstance(self.reduce_lr, bool):
            raise ValueError('reduce_lr must be a boolean.')
        if self.reduce_lr:
            self.callbacks.append(ReduceLROnPlateau(monitor='val_accuracy', patience=5, factor=0.1, verbose=1))      

        #kwarg for whether or not to use early stopping. default is True.
        self.early_stopping = train_kwargs.get('early_stopping', True)
        if not isinstance(self.early_stopping, bool):
            raise ValueError('early_stopping must be a boolean.')
        if self.early_stopping:
            self.callbacks.append(EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, restore_best_weights=True))
        
        self.model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=self.batch_size, epochs=self.epochs, callbacks=self.callbacks)

    def test_model(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test = np.argmax(y_test, axis=1)

        print(classification_report(y_test, y_pred))
        ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred)).plot()
        plt.show()


if __name__ == '__main__':
    X_TRAIN, X_VAL, X_TEST, Y_TRAIN, Y_VAL, Y_TEST= get_embedding_data()
    model = LIAR_SixWayClassifier((X_TRAIN.shape[1], X_TRAIN.shape[2]))
    model.compile_model()
    model.train_model(X_TRAIN, Y_TRAIN, X_VAL, Y_VAL)
    model.test_model(X_TEST, Y_TEST)