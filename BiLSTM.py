from tensorflow import keras
from keras import layers
import keras_tuner as kt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import classification_report

class BiLSTM():
    
    def __init__(self, max_words, max_len) -> None:
        self.max_words = max_words
        self.max_len = max_len
        # self.model = keras.models.Sequential([
        #     layers.Embedding(max_words, 32, input_length=max_len),
        #     layers.Bidirectional(layers.LSTM(16)),
        #     layers.Dense(512, activation='relu', kernel_regularizer='l1'),
        #     layers.BatchNormalization(),
        #     layers.Dropout(0.5),
        #     layers.Dense(2, activation='softmax')
        # ])
        
        # self.model.compile(loss='categorical_crossentropy',
        #             optimizer='adam',
        #             metrics=['accuracy'])
        
        self.es = EarlyStopping(patience=5,
                   monitor = 'val_accuracy',
                   restore_best_weights = True)
 
        self.lr = ReduceLROnPlateau(patience = 5,
                            monitor = 'val_loss',
                            factor = 0.5,
                            verbose = 0)
        
    def model_builder(self, hp):
        model = keras.Sequential()
        embedding_size = hp.Int('embedding_size', min_value = 16, max_value = 64, step = 16)
        model.add(layers.Embedding(self.max_words, embedding_size,input_length=self.max_len))
        lstm_units = hp.Int('lstm_units', min_value = 16, max_value = 64, step = 16)
        model.add(layers.Bidirectional(layers.LSTM(lstm_units)))
        dense_units = hp.Int('dense_units', min_value = 265, max_value = 1024, step = 128)
        model.add(layers.Dense(dense_units, activation='relu', kernel_regularizer='l1'))
        model.add(layers.BatchNormalization())
        dropout_amount = hp.Choice('dropout_amount',values=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        model.add(layers.Dropout(dropout_amount))
        model.add(layers.Dense(2, activation='softmax'))

        model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model

    def tune(self, training_data, training_labels, validation_data, validation_labels, project_name):
        tuner = kt.Hyperband(self.model_builder, objective='val_accuracy', max_epochs=10, factor=3, directory='Hyperparameter_Tuning', project_name=project_name)
        stop_early = EarlyStopping(monitor='val_loss', patience=5)
        tuner.search(training_data, training_labels, validation_data=(validation_data, validation_labels), callbacks=[stop_early])
        self.best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        print(f"""
                The hyperparameter search is complete. The optimal embedding size is {self.best_hps.get('embedding_size')},
                the optimal units in the BiLSTM is {self.best_hps.get('lstm_units')}
                the optimal number of units in the dense layer is {self.best_hps.get('dense_units')} 
                and the optimal dropout rate is {self.best_hps.get('dropout_amount')}.
                """)
        
        self.model = self.model_builder(self.best_hps)

    def plot_model(self):
        keras.utils.plot_model(
            self.model,
            show_shapes=True,
            show_dtype=True,
            show_layer_activations=True,
            to_file="BiLSTM.png"
        )
    
    def fit(self, training_data, training_labels, validation_data, validation_labels):
        self.history = self.model.fit(training_data, training_labels,
                    validation_data=(validation_data, validation_labels),
                    epochs=50,
                    verbose=1,
                    batch_size=32,
                    callbacks=[self.lr, self.es])
        return self.history
    
    def plot_history(self):
        history_df = pd.DataFrame(self.history.history)
        history_df.loc[:, ['loss', 'val_loss']].plot()
        history_df.loc[:, ['accuracy', 'val_accuracy']].plot()
        plt.show()
    
    def test(self, testing_data, testing_labels):
        testing_labels = np.where(testing_labels==True)[1]
        y_pred = self.model.predict(testing_data)
        y_pred_bool = np.argmax(y_pred, axis=1)
        print(classification_report(testing_labels, y_pred_bool))