import pandas as pd
from Data_Preprocessing import *

class Second_Dataset:
    def __init__(self) -> None:
        self.df = pd.read_csv('hasoc_english_dataset.tsv',sep="\t")
    
    def preprocess(self):
        remove_retweet_label(self.df,"text")
        remove_URL(self.df,"text")
        replaceAtUser(self.df,"text")
        remove_contractions(self.df,"text")
        convertLowerCase(self.df,"text")
        remove_emojis(self.df,"text")
        remove_punctuation(self.df,'text')
        remove_at_symbols(self.df,'text')
        remove_numbers(self.df,"text")
        remove_stopwords(self.df, 'text')
    
    def split(self, val_size = 0.1, test_size = 0.1):
        features = self.embedded_tweets
        target = self.encoded_labels
        self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = split_data(features,target,val_size,test_size)
    
    def encode_labels(self):
        self.encoded_labels = one_hot_encode(self.df['task_1'])

    def embed_tokens(self, max_words, max_len):
        self.embedded_tweets = generate_token_embeddings(self.df['text'], max_words, max_len)

    def getData(self):
        return self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test
    

    def split_2(self, val_size = 0.1, test_size = 0.1):
        features = self.df['text']
        target = self.df['task_1']
        self.X_train, self.X_val, self.X_test, self.Y_train, self.Y_val, self.Y_test = split_data(features,target,val_size,test_size)