import pandas as pd
import os
import re
import nltk
import requests
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#matplotlib inline
from nltk.corpus import stopwords
nltk.download("stopwords")
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
warnings.filterwarnings('ignore')


def load_excel():

    books = pd.read_csv('./res/Preprocessed_data.csv')
    # books = pd.read_csv('C:/Users/junho park/Desktop/Python/Reco_Engine/res/Preprocessed_data.csv')

    df = books.copy()
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.drop(columns=['Unnamed: 0', 'location', 'isbn', 'img_s', 'img_m', 'city', 'age', 'state', 'Language', 'country',
                     'year_of_publication'], axis=1, inplace=True)  # remove useless cols

    df.drop(index=df[df['Category'] == '9'].index, inplace=True)  # remove 9 in category

    df.drop(index=df[df['rating'] == 0].index, inplace=True)  # remove 0 in rating

    # 문자열(_)제외하고 반복하면서 삭제
    df['Category'] = df['Category'].apply(lambda x: re.sub('[\W_]+', ' ', x).strip())

    df.head()

    return df


def load_parameter(df):

    book_title = ''

    return book_title