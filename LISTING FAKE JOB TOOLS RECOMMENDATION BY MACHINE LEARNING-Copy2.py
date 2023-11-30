#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install wordcloud')


# In[2]:


get_ipython().system('pip install -U spacy')


# In[3]:


import re
import string
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.base import TransformerMixin
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from wordcloud import WordCloud
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English


# In[4]:


df=pd.read_csv('fake_job_postings.csv')


# In[5]:


df.head()


# In[6]:


df.shape


# In[7]:


df.isnull().sum()


# In[8]:


columns=['job_id', 'telecommuting', 'has_company_logo', 'has_questions', 'salary_range', 'employment_type']
for colu in columns:
 del df[colu]


# In[9]:


df.head()


# In[10]:


df.fillna('',inplace=True)


# In[11]:


plt.figure(figsize=(15,5))
sns.countplot(y='fraudulent', data=df)
plt.show()


# In[12]:


df.groupby('fraudulent')['fraudulent'].count()


# In[13]:


exp = dict(df.required_experience.value_counts())
del exp['']


# In[14]:


exp


# In[15]:


plt.figure(figsize=(10,5))
sns.set_theme (style="whitegrid")
plt.bar (exp.keys (), exp.values())
plt.title('No. of Jobs with Experience',size=20)
plt.xlabel('Experience', size=10)
plt.ylabel('No. of jobs', size=10)
plt.xticks (rotation=30)
plt.show()


# In[16]:


def split(location):
    l = location.split(',')
    return l[0]
df['country'] = df.location.apply(split)


# In[17]:


df.head()


# In[18]:


countr= dict(df.country.value_counts() [:14])
del countr['']
countr


# In[19]:


plt.figure(figsize=(8,6))
plt.title('Country-wise Job Posting', size=20)
plt.bar(countr.keys(), countr.values())
plt.ylabel('No. of jobs', size=10)
plt.xlabel ('Countries', size=10)


# In[20]:


edu = dict(df.required_education.value_counts()[:7])
del edu['']
edu


# In[21]:


plt.figure(figsize=(15,6))
plt.title('Job Postings basd on Education', size=20)
plt.bar (edu.keys(),edu.values())
plt.ylabel('No. of jobs', size=10)
plt.xlabel('Education', size=10)


# In[22]:


print (df[df.fraudulent==0].title.value_counts()[: 10])


# In[23]:


print (df[df.fraudulent==1].title.value_counts()[: 10])


# In[24]:


df['text']=df['title']+' '+df['company_profile']+' '+df['description']+' '+df['requirements']+' '+df['benefits']
del df['title']
del df['location']
del df['department']
del df['company_profile']
del df['description']
del df['requirements']
del df['benefits']
del df['required_experience']
del df['required_education']
del df['industry']
del df['function']
del df['country']


# In[25]:


df.head()


# In[26]:


fraudjobs_text = df[df.fraudulent==1].text
realjobs_text = df[df.fraudulent==0].text


# In[29]:


STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize = (16, 14))
wc = WordCloud(min_font_size = 3, max_words = 3000, width = 1600, height = 800, stopwords = STOPWORDS).generate(str(" ".join(fraudjobs_text)))
plt.imshow(wc,interpolation = 'bilinear')


# In[31]:


STOPWORDS = spacy.lang.en.stop_words.STOP_WORDS
plt.figure(figsize = (16, 14))
wc = WordCloud(min_font_size = 3, max_words = 3000, width = 1600, height = 800, stopwords = STOPWORDS).generate(str(" ".join(realjobs_text)))
plt.imshow(wc,interpolation = 'bilinear')


# In[32]:


get_ipython().system('pip install spacy && python -m spacy download en')


# In[39]:


punctuations = string.punctuation

nlp = spacy.load("en_core_web_sm")
stop_words = spacy.lang.en.stop_words.STOP_WORDS

parser = English()

def spacy_tokenizer (sentence):
    mytokens = parser (sentence)
    
    mytokens = [ word.lemma_.lower ().strip() if word.lemma_ != "-PRON-" else word.lower for word in mytokens ]

    mytockens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    return mytokens

class predictors (TransformerMixin):
    def transform(self, X, **transform_params):

        return [clean_text (text) for text in X]
    
    def fit (self, x, y=None, **fit_params):

        return self
    
    def get_params (self, deep=True):

        return {}

def clean_text(text):
    return text.strip().lower()


# In[41]:


df['text'] = df['text'].apply(clean_text)


# In[45]:


from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


cv = TfidfVectorizer(max_features=100)

x = cv.fit_transform(df['text'])

df1 = pd.DataFrame(x.toarray(), columns=cv.get_feature_names_out())

df.drop(["text"], axis=1, inplace=True)

main_df = pd.concat([df1, df], axis=1)


# In[46]:


main_df.head()


# In[47]:


from sklearn.model_selection import train_test_split

Y = main_df.iloc[:, -1]
X = main_df.iloc[:, :-1]

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)


# In[49]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier (n_jobs=3,oob_score=True, n_estimators=100, criterion="entropy")
model=rfc.fit(X_train,y_train)


# In[50]:


print (X_test)


# In[53]:


pred = rfc.predict(X_test)
score = accuracy_score(y_test, pred)
score


# In[54]:


print ("Classification Report\n")
print(classification_report (y_test, pred))
print("Confusion Matrix\n")
print(confusion_matrix(y_test, pred))


# In[ ]:




