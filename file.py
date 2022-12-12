#!/usr/bin/env python
# coding: utf-8

import simplejson as json
import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
import textblob
import matplotlib.pyplot as plt
import seaborn as sb
import scipy




from nltk.corpus import stopwords
from textblob import Word
from textblob import TextBlob
from pylab import rcParams
from scipy.stats import pearsonr
#from IPython import get_ipython



json_data = None
with open("yelpreview.json") as data_file:
    lines = json.load(data_file)
   
    

lines[0]['text']

review=[]

for x in range(110):
    review.append(lines[x]['text'])


df=pd.DataFrame(np.array(review),columns=['review'])
df

rating=[]

for x in range(110):
    rating.append(lines[x]['stars'])


df2=pd.DataFrame(np.array(rating),columns=['rating'])



df2
dataframe = pd.concat([df,df2], axis=1)


dataframe['word_count'] = dataframe['review'].apply(lambda x : len(x.split()))

dataframe['char_count'] = dataframe['review'].apply(lambda x :len(x))


def avg_word(review):
  words = review.split()
  return (sum(len(word) for word in words) / len(words))

# Calculate average words
dataframe['avg_word'] = dataframe['review'].apply(lambda x: avg_word(x))


dataframe

stop_words = stopwords.words('english')


stop_words
len(stop_words)

dataframe['stopWord_count']=dataframe['review'].apply(lambda x: len([word for word in x.split() if word.lower() in stop_words]))
dataframe['stopWord_rate']= dataframe['stopWord_count'] / dataframe ['word_count']



#Data_Cleaning
dataframe['lowerCase']=dataframe['review'].apply(lambda x: " ".join(word.lower() for word in x.split()))
dataframe['punctuation']=dataframe['lowerCase'].str.replace('[^\w\s]','')
dataframe['remove_stopwords']=dataframe['punctuation'].apply(lambda x: " ".join(word for word in x.split() if word not in stop_words))

dataframe['lemmatize']=dataframe['remove_stopwords'].apply(lambda x: " ".join(Word(word).lemmatize()for word in x.split()))
dataframe['polarity']=dataframe['lemmatize'].apply(lambda x: TextBlob(x).sentiment[0])
dataframe['subjectivity']=dataframe['lemmatize'].apply(lambda x: TextBlob(x).sentiment[1])


dataframe.describe()

print(dataframe['polarity'],dataframe['subjectivity'])


#Pearson Correlation parametric method

get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize']=5,4
sb.set_style('whitegrid')



#Pearson correlation

sb.pairplot(dataframe)


dataframe2 = dataframe[['polarity','subjectivity','rating']]


sb.pairplot(dataframe2)

polaritys=dataframe['polarity']
ratings=dataframe['rating']


pearsonr_coefficient,p_value = pearsonr(polaritys,ratings)
print (pearsonr_coefficient)


corr = dataframe2.corr()

corr


sb.heatmap(corr,xticklabels=corr.polarity.values,yticklabels=corr.rating.values)

dataframe2.head()

dataframe2.describe()
dataframe.head()





