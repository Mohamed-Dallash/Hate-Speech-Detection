import pandas as pd
import string
import warnings
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sb
# from sklearn.model_selection import train_test_split
 
# # Text Pre-processing libraries
# import nltk
# import string
# import warnings
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer
# from wordcloud import WordCloud

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')

def convertLowerCase(df : pd.DataFrame, col_name : str):
    df[col_name] = df[col_name].str.lower()

def remove_punctuation(df : pd.DataFrame, col_name : str):
    def punctuation_remover(text):
        temp = str.maketrans('', '', string.punctuation)
        return text.translate(temp)
    
    df[col_name] = df[col_name].apply(lambda text: punctuation_remover(text))

def remove_stopwords(df : pd.DataFrame, col_name : str):
    def stopwords_remover(text):
        stop_words = stopwords.words('english')

        imp_words = []

        # Storing the important words
        for word in str(text).split():

            if word not in stop_words:

                # Let's Lemmatize the word as well
                # before appending to the imp_words list.

                lemmatizer = WordNetLemmatizer()
                lemmatizer.lemmatize(word)

                imp_words.append(word)

        output = " ".join(imp_words)

        return output
    df[col_name] = df[col_name].apply(lambda text: stopwords_remover(text))

def plot_labels_distribution(df : pd.DataFrame, col_name : str):
    plt.pie(df[col_name].value_counts().values,
        labels = df[col_name].value_counts().index,
        autopct='%1.1f%%')
    plt.show()

def plot_word_cloud(df : pd.DataFrame, col_name : str, label : str):
    # Joining all the tweets to get the corpus
    email_corpus = " ".join(df[col_name])

    plt.figure(figsize = (10,10))

    # Forming the word cloud
    wc = WordCloud(max_words = 100,
                width = 2000,
                height = 1000,
                collocations = False).generate(email_corpus)

    # Plotting the wordcloud obtained above
    plt.title(f'WordCloud for {label} tweets.', fontsize = 15)
    plt.axis('off')
    plt.imshow(wc)
    plt.show()
    print()

def split_data(features, target, val_size = 0.1, test_size = 0.1):
    X_train, X_temp, Y_train, Y_temp = train_test_split(features,
                                                  target,
                                                  test_size=val_size+test_size,
                                                  random_state=22,
                                                  stratify=target)
    X_val, X_test, Y_val, Y_test = train_test_split(X_temp,
                                                  Y_temp,
                                                  test_size=test_size/(val_size+test_size),
                                                  random_state=22,
                                                  stratify=Y_temp)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test

def one_hot_encode(data):
    return pd.get_dummies(data)

def generate_token_embeddings(train,val,test, max_words = 5000, max_len = 50):

    token = Tokenizer(num_words=max_words,
                    lower=True,
                    split=' ')
    
    token.fit_on_texts(train)


    Training_seq = token.texts_to_sequences(train)
    Training_pad = pad_sequences(Training_seq,
                                maxlen=max_len,
                                padding='post',
                                truncating='post')
    

    Vaidation_seq = token.texts_to_sequences(val)
    Validation_pad = pad_sequences(Vaidation_seq,
                                maxlen=max_len,
                                padding='post',
                                truncating='post')

    Testing_seq = token.texts_to_sequences(test)
    Testing_pad = pad_sequences(Testing_seq,
                                maxlen=max_len,
                                padding='post',
                                truncating='post')
    
    return Training_pad, Validation_pad, Testing_pad