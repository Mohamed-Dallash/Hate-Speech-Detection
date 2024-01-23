from tensorflow import keras
from keras import layers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import classification_report

class RNN():
    
    def __init__(self, max_words, max_len) -> None:
        self.model = keras.models.Sequential([
            layers.Embedding(max_words, 32, input_length=max_len),
            layers.SimpleRNN(16),
            layers.Dense(512, activation='relu', kernel_regularizer='l1'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(2, activation='softmax')
        ])
        
        self.model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])
        
        self.es = EarlyStopping(patience=5,
                   monitor = 'val_accuracy',
                   restore_best_weights = True)
 
        self.lr = ReduceLROnPlateau(patience = 2,
                            monitor = 'val_loss',
                            factor = 0.5,
                            verbose = 0)
        
    def plot_model(self):
        keras.utils.plot_model(
            self.model,
            show_shapes=True,
            show_dtype=True,
            show_layer_activations=True,
            to_file="RNN.png"
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