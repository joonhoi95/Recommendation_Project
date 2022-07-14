import pandas_profiling
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')



from PIL import Image as im
#from wordcloud import WordCloud,STOPWORDS
from IPython.core.display import Image
from colorama import Fore, Back, Style
y_ = Fore.YELLOW
r_ = Fore.RED
g_ = Fore.GREEN
b_ = Fore.BLUE
m_ = Fore.MAGENTA
sr_ = Style.RESET_ALL

u_cols = ['user_id', 'location', 'age']
users = pd.read_csv('C:/Users/junho park/Desktop/Python/Reco_Engine/res/BX-Users.csv', sep=';', names=u_cols, encoding='latin-1',low_memory=False)

#Books
i_cols = ['isbn', 'book_title' ,'book_author','year_of_publication', 'publisher', 'img_s', 'img_m', 'img_l']
items = pd.read_csv('C:/Users/junho park/Desktop/Python/Reco_Engine/res/BX_Books.csv', sep=';', names=i_cols, encoding='latin-1',low_memory=False)

#Ratings
r_cols = ['user_id', 'isbn', 'rating']
ratings = pd.read_csv('C:/Users/junho park/Desktop/Python/Reco_Engine/res/BX-Book-Ratings.csv', sep=';', names=r_cols, encoding='latin-1',low_memory=False)

#users.head(5)

users = users.drop(users.index[0])
items = items.drop(items.index[0])
ratings = ratings.drop(ratings.index[0])

users['age'] = users['age'].astype(float)
users['user_id'] = users['user_id'].astype(int)
ratings['user_id'] = ratings['user_id'].astype(int)
ratings['rating'] = ratings['rating'].astype(int)
items['year_of_publication'] = items['year_of_publication'].astype(int)

users.isnull().sum()

users['age'].describe()

users.loc[(users.age>99) | (users.age<5),'age'] = np.nan
users.age = users.age.fillna(users.age.mean())

ratings.isnull().sum()

items.loc[items.publisher.isnull(),:]

items.loc[items.isbn=='193169656X','publisher']='Mundania Press LLC'
items.loc[items.isbn=='1931696993','publisher']='Novelbooks Incorporated'

items.loc[items.publisher.isnull(),:]

items.loc[items.isbn=='9627982032','book_author']='Larissa Anne Downe'

df = pd.merge(users, ratings, on='user_id')
df = pd.merge(df, items, on='isbn')
df.head(5)

location = df.location.str.split(', ', n=2, expand=True)
location.columns=['city', 'state', 'country']

df['city'] = location['city']
df['state'] = location['state']
df['country'] = location['country']

plt.figure(figsize=(10,8))
sns.countplot(x='rating',data=df)
plt.title('Rating Distribution',size=20)
plt.show()

df_v=df[['rating']].copy()
df_v.dtypes
df_v = df_v[df_v.rating != 0]
plt.figure(figsize=(10,8))
sns.countplot(x='rating',data=df_v)
plt.title('Explicit Rating Distribution',size=20)
plt.show()

plt.figure(figsize=(10,8))
sns.distplot(df['age'],kde=False)
plt.xlabel('Age')
plt.ylabel('count')
plt.title('Age Distribution',size=20)
plt.show()

df_v=df[['year_of_publication']].copy()
df_v['year_of_publication'] = df_v['year_of_publication'].astype(int).astype(str)
df_v=df_v['year_of_publication'].value_counts().head(25).reset_index()
df_v.columns=['year','count']
df_v['year']='Year '+df_v['year']

plt.figure(figsize=(10,8))
sns.barplot(x='count',y='year',data=df_v)
plt.ylabel('Year Of Publication')
plt.yticks(size=12)
plt.title('Years of Publication',size=20)
plt.show()

def barplot(df,col,l):
    df_v=df[col].value_counts().head(25).reset_index()
    df_v.columns=[col,'count']

    plt.figure(figsize=(10,12))
    sns.barplot(x='count',y=col,data=df_v)
    plt.ylabel(l)
    plt.title(l,size=20)
    plt.show()


barplot(df,'book_title','Book Title')