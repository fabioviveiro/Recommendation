#!/usr/bin/env python
# coding: utf-8

# <h1><b>Libraries import</b></h1>

# In[231]:


import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# <h1><b>Load data</b></h1>

# In[7]:


df_order_items = pd.read_csv('DNC_order_items_dataset.csv',index_col='Unnamed: 0')
df_order_reviews = pd.read_csv('DNC_order_reviews_dataset.csv',index_col='Unnamed: 0')
df_orders = pd.read_csv('DNC_orders_dataset.csv',index_col='Unnamed: 0')
df_products = pd.read_csv('DNC_products_dataset.csv',index_col='Unnamed: 0')


# In[8]:


display(df_order_items.head().style)
print(f'shape is: {df_order_items.shape}')


# In[9]:


display(df_order_reviews.head().style)
print(f'shape is: {df_order_reviews.shape}')


# In[10]:


display(df_orders.head().style)
print(f'shape is: {df_orders.shape}')


# In[11]:


display(df_products.head().style)
print(f'shape is: {df_products.shape}')


# <h1><b>Data analysis</b></h1>

# In[60]:


def info_basicas(dataf):
    display(dataf.describe(include='all').transpose())
    print(f'\n{dataf.info()}')
    print(f'\nContagem de nulos(%):\n{(dataf.isnull().sum().sort_values(ascending=False)/dataf.shape[0])*100}')


# In[61]:


info_basicas(df_order_items)


# In[62]:


info_basicas(df_order_reviews)


# In[63]:


info_basicas(df_orders)


# In[64]:


print(f'Contagem de valores:\n{df_orders["order_status"].value_counts()}')


# In[65]:


info_basicas(df_products)


# In[66]:


print(f'Contagem de valores:\n{df_products["product_category_name"].value_counts()}')


# In[67]:


df_products.dropna(inplace=True)


# <h1><b>Recommendation model</b></h1>

# <h2><b>Merge Datasets</b></h2>

# In[69]:


df_int = df_order_items.merge(df_order_reviews,how='left',left_on='order_id',right_on='order_id')


# In[72]:


display(df_int.head().style)
print(f'shape is: {df_int.shape}')


# In[71]:


info_basicas(df_int)


# In[146]:


df_rec_avalicao = df_products.merge(df_int,how='left',left_on='product_id',right_on='product_id')
display(df_rec_avalicao.head().style)
print(f'shape is: {df_rec_avalicao.shape}')


# In[147]:


df_rec_avalicao.drop(columns=['product_name_lenght',
       'product_description_lenght', 'product_weight_g', 'product_length_cm',
       'product_height_cm', 'product_width_cm', 'order_id', 'order_item_id',
       'price', 'review_id'],inplace=True)


# In[148]:


info_basicas(df_rec_avalicao)


# In[149]:


df_rec_avalicao.dropna(inplace=True)


# In[151]:


df_rec_prod = df_products.merge(df_int,how='left',left_on='product_id',right_on='product_id')
display(df_rec_prod.head().style)
print(f'shape is: {df_rec_prod.shape}')


# In[152]:


info_basicas(df_rec_prod)


# In[153]:


df_rec_prod.dropna(inplace=True)


# <h2><b>Recommendation of the most purchased items by category</b></h2>

# This model can be used for the categories' first page when a user enter on the page the model will return the list of the most purchased products

# In[297]:


def start_pipeline(dataf):
    d = dataf.copy()
    return d

def product_count(d, n):
    x = pd.pivot_table(df_rec_avalicao,index=['product_category_name', 'product_id'],                        aggfunc={'product_id':pd.Series.value_counts})    .rename(columns={'product_id':'product_id_count'})     .sort_values(['product_category_name','product_id_count'], ascending = False)    .reset_index()
    y = x.groupby(by='product_category_name').head(n).reset_index(drop=True)
    return y

def dicio(y):
    categories = top_products.product_category_name.unique()
    dict_categories = {category: top_products[top_products.product_category_name==category]                       .product_id.unique().tolist() for category in categories}
    return dict_categories


# In[298]:


top_products = df_rec_avalicao     .pipe(start_pipeline)     .pipe(product_count, n=5)     .pipe(dicio)


# In[299]:


top_products


# <h2><b>Recommendation of product similarity</b></h2>

# This model should be used on shopping cart's page to recommend similar products for the user

# In[161]:


df_rec_prod.drop(columns=['order_id', 'order_item_id','review_id'],inplace=True)


# In[221]:


valores = ['product_name_lenght','product_description_lenght', 'product_weight_g',
           'product_length_cm','product_height_cm', 'product_width_cm', 'price', 'review_score']
grupo = ['product_id','product_category_name']


# In[225]:


df_agrupado = df_rec_prod.groupby(by=grupo, as_index=False)[valores].mean()
df_agrupado


# In[229]:


df_agrupado = pd.get_dummies(df_agrupado, columns=['product_category_name'])


# In[234]:


df_agrupado = df_agrupado.set_index('product_id')


# In[235]:


df_agrupado


# In[236]:


cos_prod = cosine_similarity(df_agrupado)
cos_prod


# In[264]:


prod_semelhantes = df_agrupado.index[3]

print(f'O produto escolhido é: {prod_semelhantes}')

prod_id = df_agrupado.index.tolist().index(prod_semelhantes)
print(f'Produto: {prod_semelhantes}, tem índice: {prod_id}')

similares_10 = np.argsort(-cos_prod[prod_id])[1:10]

for i in zip(df_agrupado.index[similares_10],cos_prod[prod_id][similares_10]):
  print(f'Produto {i[0]} tem similaridade {i[1]:2f} com produto {prod_semelhantes}')

