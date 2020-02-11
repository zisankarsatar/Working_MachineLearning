# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 16:42:45 2019

@author: ZİŞAN
"""
import pandas as pd

#%% dataset eklenir
data=pd.read_csv(r"gender-classifier.csv", encoding="latin1")
data=pd.concat([data.gender,data.description],axis=1)

#%% eksik ve anlamsız verileri çıkartma
data.dropna(axis=0, inplace=True)#nan degerleri çıkartma
#gruplandırıken string ifadeyi sayısala çevrime
#female -> 1, male ->0
data.gender=[1 if each == "female" else 0 for each in data.gender]

#%% cleaning data
#regular expression RE
import re
 
first_description=data.description[4]
#a dan z ye ve A dan Z yekadar olan harfleri değiştirme geri 
#kalanını boşluğa dönüştür
description=re.sub("[^a-zA-Z]"," ",first_description)
#her harfin boyutunu küçült
description=description.lower()

#%% stopwords (irrelavent words) gereksiz kelimeler
import nltk #natural languafe tool kit
nltk.download("stopwords") #corpus diye klasöre indiriliyor
#from nltk.corpus import stopwords #corpusdan import ediliyor
#description=description.split()
#split yerine tokanizer kullanılıyor
description = nltk.word_tokenize(description)

#%%
#gereksiz kelimleri çıkart
description= [word for word in description if not word in set(stopwords.words("english"))]

#%%
#lemmatization loved->love kelime kökleri
import nltk as nlp
lemma=nlp.WordNetLemmatizer()
description=[lemma.lemmatize(word) for word in description]
#kelimeleri tekrar birleştirdi
description=" ".join(description)
#%%tüm datalar için yapma
description_list=[]
for description in data.description:
    description=re.sub("[^a-zA-Z]"," ",description)
    description=description.lower()
    description = nltk.word_tokenize(description)
    description= [word for word in description if not word in set(stopwords.words("english"))]
    lemma=nlp.WordNetLemmatizer()
    description=[lemma.lemmatize(word) for word in description]
    description=" ".join(description)
    description_list.append(description)

#%%bag of words

from sklearn.feature_extraction.text import CountVectorizer #bag of words yazdırmak için kullanılan vektör
max_features=5000

count_vectorizer=CountVectorizer(max_features=max_features) #parametre olarak stop_words="english" vb seyler alabilr

sparce_matrix=count_vectorizer.fit_transform(description_list).toarray() #x
    
print("en sık kullanılan {} kelimler: {}".format(max_features,count_vectorizer.get_feature_names()))

#%%
y= data.iloc[:,0].values #male or female
x=sparce_matrix
from sklearn.model_selection import train_test_split #train test split
x_train, x_test, y_train, y_test=train_test_split(x,y, test_size=0.1, random_state=42)

#%%navie bayes

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()
nb.fit(x_train,y_train)

#prediction
y_pred=nb.predict(x_test)
print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))











