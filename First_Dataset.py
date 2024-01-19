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
        features = self.embedded_tweets
        target = self.encoded_labels
        self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = split_data(features,target,val_size,test_size)
    
    def encode_labels(self):
        self.encoded_labels = one_hot_encode(self.df['class'])

    def embed_tokens(self, max_words, max_len):
        self.embedded_tweets = generate_token_embeddings(self.df['tweet'], max_words, max_len)

    def getData(self):
        return self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test