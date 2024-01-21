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
import re
import contractions
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('omw-1.4')
nltk.download('wordnet')
warnings.filterwarnings('ignore')

def convertLowerCase(df : pd.DataFrame, col_name : str):
    df[col_name] = df[col_name].str.lower()

def remove_punctuation(df : pd.DataFrame, col_name : str):
    punctuations = string.punctuation.replace('@','')
    def punctuation_remover(text):
        temp = str.maketrans('', '', punctuations)
        return text.translate(temp)
    
    df[col_name] = df[col_name].apply(lambda text: punctuation_remover(text))

def remove_at_symbols(df : pd.DataFrame, col_name : str):
    df[col_name] = df[col_name].apply(lambda text: re.sub('@(?!(user))','',str(text)))

def remove_numbers(df : pd.DataFrame, col_name : str):
    df[col_name] = df[col_name].apply(lambda text: ''.join([i for i in text if not i.isdigit()]))

def remove_retweet_label(df : pd.DataFrame, col_name : str):
    df[col_name] = df[col_name].apply(lambda text: re.sub('(RT @[\w_]+:?)','', str(text)))

def remove_URL(df : pd.DataFrame, col_name : str):
    df[col_name] = df[col_name].apply(lambda text: re.sub('((www\.[^\s]+)|(https?://[^\s]+))','',str(text)))

def replaceAtUser(df : pd.DataFrame, col_name : str):
    def atUser_replacer(text):
        text = re.sub('@[^\s]+:','',str(text))
        text = re.sub('@[^\s]+','@user',str(text))
        return text
    
    df[col_name] = df[col_name].apply(lambda text: atUser_replacer(text))

def remove_contractions(df : pd.DataFrame, col_name : str):

    df[col_name] = df[col_name].apply(lambda text: ' '.join([contractions.fix(word) for word in str(text).split()]))

def remove_emojis(df : pd.DataFrame, col_name : str):
    emoji_pattern = re.compile("["
          u"\U0001F600-\U0001F64F"  # emoticons
          u"\U0001F300-\U0001F5FF"  # symbols & pictographs
          u"\U0001F680-\U0001F6FF"  # transport & map symbols
          u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
          u"\U00002702-\U000027B0"
          u"\U000024C2-\U0001F251"
          u"\U0001f926-\U0001f937"
          u'\U00010000-\U0010ffff'
          u"\u200d"
          u"\u2640-\u2642"
          u"\u2600-\u2B55"
          u"\u23cf"
          u"\u23e9"
          u"\u231a"
          u"\u3030"
          u"\ufe0f"
                            "]+", flags=re.UNICODE)
    df[col_name] = df[col_name].apply(lambda text: emoji_pattern.sub(r'', str(text)))

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

def split_data(features, target, val_size = 0.15, test_size = 0.15):
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

def generate_token_embeddings(data, max_words = 5000, max_len = 50):

    token = Tokenizer(num_words=max_words,
                    lower=True,
                    split=' ',
                    )
    
    token.fit_on_texts(data)

    seq = token.texts_to_sequences(data)
    pad = pad_sequences(seq,
                        maxlen=max_len,
                        padding='post',
                        truncating='post')
    
    return pad


def convert_to_tfidf(X_train, X_val, X_test, max_features=5000):
    vectorizer = TfidfVectorizer(max_features=max_features)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_val_tfidf, X_test_tfidf