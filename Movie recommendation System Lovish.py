#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import warnings


# In[4]:


warnings.filterwarnings('ignore')


# In[5]:


columns_name=['user_id','item_id','rating','timestamp']
df=pd.read_csv('u.data',sep="\t",names=columns_name)


# In[6]:


df.head()


# In[7]:


df.shape


# In[8]:


df['user_id']


# In[9]:


df['user_id'].nunique()


# In[10]:


df['item_id'].nunique()


# In[11]:


movies_title=pd.read_csv('u.item',sep="\|",header=None)


# In[12]:


movies_title.shape


# In[13]:


movies_titles=movies_title[[0,1]]
movies_titles.columns=["item_id","title"]
movies_titles.head()


# In[14]:


df=pd.merge(df,movies_titles,on="item_id")


# In[28]:


df


# In[15]:


df.tail()


# In[16]:


ratings=pd.DataFrame(df.groupby('title').mean()['rating'])


# In[17]:


ratings.head()


# In[19]:


ratings['num of ratings']=pd.DataFrame(df.groupby('title').count()['rating'])


# #recommendar system

# In[20]:


df.head()


# In[21]:


moviemat=df.pivot_table(index='user_id',columns='title',values='rating')


# In[22]:


moviemat.head()


# In[30]:


starwars_user_ratings=moviemat['Star Wars (1977)']


# In[31]:


starwars_user_ratings.head()


# In[32]:


similar_to_starwars=moviemat.corrwith(starwars_user_ratings)


# In[33]:


similar_to_starwars


# In[34]:


corr_starwars=pd.DataFrame(similar_to_starwars,columns=['correlation'])


# In[35]:


corr_starwars.dropna(inplace=True)


# In[37]:


corr_starwars


# In[38]:


corr_starwars.head()


# In[39]:


corr_starwars.sort_values('correlation',ascending=False).head(10)


# In[40]:


ratings


# In[41]:


corr_starwars=corr_starwars.join(ratings['num of ratings'])


# In[42]:


corr_starwars


# In[43]:


corr_starwars.head()


# In[45]:


corr_starwars[corr_starwars['num of ratings']>100].sort_values('correlation',ascending=False)


# In[46]:


def predict_movies(movie_name):
    movie_user_ratings=moviemat[movie_name]
    similar_to_movie=moviemat.corrwith(movie_user_ratings)
    corr_movie=pd.DataFrame(similar_to_movie,columns=['correlation'])
    corr_movie.dropna(inplace=True)
    corr_movie=corr_movie.join(ratings['num of ratings'])
    predictions=corr_movie[corr_movie['num of ratings']>100].sort_values('correlation',ascending=False)
    return predictions


# In[47]:


predict_my_movie=predict_movies("Titanic (1997)")


# In[48]:


predict_my_movie.head()


# In[ ]:




