#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


homelessness = pd.read_csv('2007-2016-Homelessnewss-USA.csv')


# In[3]:


homelessness.head()


# In[4]:


homelessness.info()


# In[5]:


print(homelessness.columns)
print('--')

print(homelessness.index)
print('--')


# In[6]:


homelessness['Count'] = homelessness['Count'].replace('[^\d.]','',regex=True).astype(int)


# In[7]:


homelessness.sort_values('State').head()


# In[8]:


homelessness['State'].unique()


# In[9]:


homelessness.sort_values(['State','Measures'], ascending=[True,False]).head()


# In[ ]:





# In[10]:


#homelessness['Measures'].unique()


# In[11]:


homelessness[homelessness['Count']<100]


# In[12]:


homelessness[homelessness['State']=='OH']


# In[13]:


homelessness=homelessness[homelessness['Count']>100]


# In[14]:


homelessness.head()


# In[15]:


happiness = pd.read_csv('output.csv')


# In[16]:


happiness.head()


# In[17]:


in_happiness = happiness['Region'].isin(['Western Europe','North America'])


# In[18]:


happiness[in_happiness]


# In[19]:


happiness.drop(['Explained by: Freedom to make life choices','Explained by: Generosity'],axis='columns')

happiness.info()
# In[20]:


wall= pd.read_csv('mrkt_data.csv')


# In[21]:


wall.head()


# In[22]:


ewallet = wall[wall['Payment']=='Ewallet']


# In[23]:


ewallet.head()


# In[24]:


wall['Gender'].value_counts()


# In[25]:


is_health_and_be = wall['Product line'].isin(['Health and beauty'])
wall[is_health_and_be]


# In[26]:


import seaborn as sns


# In[27]:


sns.relplot(x='Gender', y='gross income', data=wall, hue='Total')


# In[28]:


print(f"Meadian: {wall['Unit price'].mean():.2f}")


# In[29]:


print(f"Max : {wall['gross income'].max()}")


# In[30]:


def pct40(column):
    return column.quantile(0.4)

def pct30(column):
    return column.quantile(0.3)

print("-----------------")

print(f"3er quantile : {wall['gross income'].agg(pct30)} ")
print('------------')

print(wall['gross income'].agg([pct30,pct40]))


# In[31]:


wall.head(2)


# In[32]:


import numpy as np


# In[33]:


def iqr(column):
    return column.quantile(0.75) - column.quantile(0.25)

print(wall['Total'].agg(iqr))
print('------')


# In[34]:


print(wall[['Unit price','Total','gross income']].agg(iqr))


# # Counting 

# In[35]:


happ = pd.read_csv('output.csv')


# In[36]:


happ.head(2)


# In[37]:


happ.columns


# In[38]:


df1=happ[happ['year'] ==2015]


# In[39]:


df1.head(2)


# In[40]:


df1.drop(['Explained by: Log GDP per capita', 'Explained by: Social support',
       'Explained by: Healthy life expectancy',
       'Explained by: Freedom to make life choices',
       'Explained by: Generosity', 'Explained by: Perceptions of corruption',
       'Dystopia + residual', 'RANK', 'Happiness score', 'Whisker-high',
       'Whisker-low', 'Dystopia (1.83) + residual',
       'Explained by: GDP per capita'],axis='columns',inplace = True)


# In[41]:


df1.columns


# In[42]:


df1.head(2)


# In[43]:


df1.drop(['Freedom to make life choices', 'Perceptions of corruption',
       'Country name', 'Regional indicator', 'Ladder score',
       'Standard error of ladder score', 'upperwhisker', 'lowerwhisker',
       'Logged GDP per capita', 'Ladder score in Dystopia'],axis='columns',inplace=True)


# In[44]:


df1.head()


# In[45]:


df1.columns


# In[46]:


df1.drop(['Lower Confidence Interval',
       'Upper Confidence Interval', 'Happiness.Rank', 'Happiness.Score',
       'Whisker.high', 'Whisker.low', 'Economy..GDP.per.Capita.',
       'Health..Life.Expectancy.', 'Trust..Government.Corruption.',
       'Dystopia.Residual', 'Overall rank', 'Country or region', 'Score',
       'GDP per capita', 'Social support', 'Healthy life expectancy'], axis='columns',inplace=True)


# In[47]:


df1.head()


# In[48]:


happ_1 = df1
happ_1.head(2)


# In[49]:


happ_1.drop_duplicates(subset='Country')


# In[50]:


unique_happ_1=happ_1['Region'].value_counts()
unique_happ_1


# In[51]:


happ_uni_sort = happ_1['Region'].value_counts(sort=True)
happ_uni_sort


# In[52]:


happ_1['Region'].value_counts(normalize=True)
#If True then the object returned will contain the relative frequencies of the unique values(perc)


# In[53]:


happ_1.groupby('Region')['Health (Life Expectancy)'].mean()


# In[54]:


happ_1.groupby('Region')['Happiness Score'].agg([min,max,np.mean])


# In[55]:


happ_1.groupby(['Region','Country'])['Freedom'].mean()


# In[56]:


happ_1.pivot_table(values='Trust (Government Corruption)', index='Region')


# In[57]:


import matplotlib.pyplot as plt


# In[58]:


data = happ_1.pivot_table(values='Trust (Government Corruption)', index= 'Region', aggfunc=np.median)


# In[59]:


sns.lineplot(x='Region',y='Trust (Government Corruption)', data=data)
locs, labels = plt.xticks()
plt.setp(labels, rotation=90)
plt.show()


# In[60]:


happ_1.pivot_table(values='Happiness Score', index='Region', aggfunc=[np.mean, np.median])


# In[61]:


happ_1.pivot_table(values='Family',index='Region',columns='Country', fill_value=0, margins=True)


# # Slicing and Indexing

# In[62]:


wall.head()


# In[63]:


male=wall[wall['Gender'] == 'Male']


# In[64]:


male.head(2)


# In[65]:


male_ind = male.set_index('Payment')


# In[66]:


male_ind.head()


# In[67]:


male_ind.reset_index()


# In[68]:


male[male['Payment'].isin(['Cash'])]


# In[69]:


electron =male[male['Product line'].isin(['Electronic accessories'])]


# In[70]:


elect1 = electron.set_index('City')


# In[71]:


elect1.loc[['Naypyitaw','Yangon']]


# # 65

# In[ ]:




