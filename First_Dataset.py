import pandas as pd
from Data_Preprocessing import *

class First_Dataset:
    def __init__(self) -> None:
        self.df = pd.read_csv('labeled_data.csv')
    
    def preprocess(self):
        self.df = self.df[self.df['class']!=1]
        convertLowerCase(self.df,"tweet")
        remove_punctuation(self.df,'tweet')
        remove_stopwords(self.df, 'tweet')
    
    def split(self, val_size = 0.1, test_size = 0.1):
        features = self.df['tweet']
        target = self.df['class']
        self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = split_data(features,target,val_size,test_size)
    
    def encode_labels(self):
        self.Y_train = one_hot_encode(self.Y_train)
        self.Y_val = one_hot_encode(self.Y_val)
        self.Y_test = one_hot_encode(self.Y_test)

    def embed_tokens(self, max_words, max_len):
        self.Training_pad, self.Validation_pad, self.Testing_pad = generate_token_embeddings(self.X_train,self.X_val,self.X_test,max_words,max_len)

    def getData(self):
        return self.Training_pad, self.Validation_pad, self.Testing_pad, self.Y_train, self.Y_val, self.Y_test