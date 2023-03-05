#!/usr/bin/env python
# coding: utf-8

# In[221]:


import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings("ignore")


# In[223]:


df1 = pd.read_csv("E:\ML Projects\Zamoto EDA\zomato-1.csv")
df2 = pd.read_csv("E:\ML Projects\Zamoto EDA\zomato-2.csv")
df3 = pd.read_csv("E:\ML Projects\Zamoto EDA\zomato-3.csv")


# In[224]:


df1.shape


# In[225]:


df2.shape


# In[226]:


df3.shape


# In[227]:


# Here we are mearging first 2 files

data = df1.merge(df2, how='outer')
data.shape


# # DataSet Information

# In[228]:


# Finally, the files are joined

df = data.merge(df3, how = 'outer')
df.shape


# In[229]:


df.head()


# In[230]:


df.tail()


# In[231]:


df.columns


# In[232]:


df.info()


# In[233]:


df.isnull().mean()*100


# By using df.isnull() function we ge to know that how many null values are present in each column.

# # Data Cleaning

# ### Droping the unnecessary columns:-

# Unnecessary column are those columns which are not required for analysis.

# In[234]:


df.drop(['dish_liked','url','address','phone','menu_item','reviews_list'],axis=1,inplace=True)


# as url , address , phone, menu items, dish liked and review list are not importanat columns so i drop the columns

# In[235]:


df.info()


# ### Renaming columns

# In[236]:


df.rename(columns={'name':'restaurants','book_table':'booking','listed_in(city)':'city','rate':'rating','approx_cost(for two people)':'cost','listed_in(type)':'categorie'},inplace=True)


# In[237]:


df.head(1)


# ### Finding out the duplicate rows:-

# Helps to find the same repeated rows.
# Then removing the duplicate rows for understanding the data.

# In[238]:


#checking duplicate values

df.duplicated().sum()


# In[239]:


#displaying duplicate rows

duplicate = df[df.duplicated()]
 
print("Duplicate Rows :")
 
# Print the resultant Dataframe
duplicate.head(5)


# In[240]:


# dropping the duplicate rows

df.drop_duplicates(keep=False, inplace=True)


# In[241]:


df.shape


# ### Checking unique values

# In[242]:


# displaying no of unique values in each columns:

print("Unique values in each columns are :")
df.nunique()


# ### Cleaning individual columns:

# #### Column:- rating

# As we can see that rate contains the values in the numerator and denominator form. so here I used the regex pattern to convert all those values into some meaningful numeric value

# In[243]:


df['rating']=df['rating'].str.replace(r'\/\d+','')


# In[244]:


# Checking unique values:

df['rating'].unique()


# In[245]:


def rate(value):
    if value=='NEW' or value=='-' or value==0:
        return np.nan
    else:
        return float(value)
    
df['rating']=df['rating'].apply(rate)
df['rating'].head()


#     1.Rating column contains some string values as well as some dash values. so I repalce all those values with null value.
#     2.As the datatype of the 'rate' is object so i convert the datatype to float.

# In[246]:


df['rating'].unique()


# In[247]:


#Counting the NaN values in rating column

df['rating'].isnull().sum()


# In[248]:


# Filling the null values with the mean value

df['rating'] = df['rating'].fillna(df['rating'].mean())


# In[249]:


#Counting the NaN values in rating column

df['rating'].isnull().sum()


# #### Column:- location 

# In[250]:


# Checking unique values:

df['location'].unique()


# In[251]:


#Counting the NaN values

df['location'].isnull().sum()


# In[252]:


# Filling the null values with the value og city column:

df['location'].fillna(df['city'], inplace=True)


# In[253]:


#Counting the NaN values

df['location'].isnull().sum()


# #### Column :- votes 

# In[254]:


# Checking unique values:

df['votes'].unique()


# In[255]:


# Replacing 0 to null values

df['votes']=df['votes'].replace(0,np.nan)


# In[256]:


#Counting the NaN values

df['votes'].isnull().sum()


# In[257]:


# Filling the null values with the mean

df['votes'] = df['votes'].fillna(df['votes'].mean())


# In[258]:


#Counting the NaN values

df['votes'].isnull().sum()


# #### Column :- Cost 

# In[259]:


df['cost'].unique()


# As we can see that in the 'cost' column their are some comma separated values so, we will convert it in normal format using one regex pattern to convert the comma with the space

# In[260]:


# changing the text format
df['cost']=df['cost'].str.replace(r"\,","")

# converting the datatype by using astype function

df['cost']=df['cost'].astype(float)


# In[261]:


# Checking unique values:

df['cost'].unique()


# In[262]:


#Counting the NaN values

df['cost'].isnull().sum()


# In[263]:


# Filling the null values with the mean

df['cost'] = df['cost'].fillna(df['cost'].mean())


# In[264]:


#Counting the NaN values

df['cost'].isnull().sum()


# #### Column :- booking 

# In[265]:


# Checking unique values:

df['booking'].unique()


# In[266]:


#Counting the NaN values

df['booking'].isnull().sum()


# ####  Column :- online order

# In[267]:


# Checking unique values:

df['online_order'].unique()


# In[268]:


#Counting the NaN values

df['online_order'].isnull().sum()


# #### Column :- rest type 

# In[269]:


# Checking unique values:

df['rest_type'].unique()


# In[270]:


#Counting the NaN values

df['rest_type'].isnull().sum()


# In[271]:


# Filling the null values with the backward fill 

df['rest_type'].fillna( method ='bfill', inplace = True)


# In[272]:


#Counting the NaN values

df['rest_type'].isnull().sum()


# #### Column :- categories 

# In[273]:


# Checking unique values:

df['categorie'].unique()


# In[274]:


#Counting the NaN values

df['categorie'].isnull().sum()


# #### Column :- cuisines 

# In[275]:


# Checking unique values:

df['cuisines'].unique()


# In[276]:


#Counting the NaN values

df['cuisines'].isnull().sum()


# In[277]:


# Filling the null values with the Unknown  

df['cuisines'].fillna('Unknown',inplace= True)


# In[278]:


#Counting the NaN values

df['cuisines'].isnull().sum()


# ####  Column :- restaurants

# In[279]:


# Checking unique values:

df['restaurants'].unique()


# In[280]:


# Correcting the name format using the regex function

df['restaurants']=df['restaurants'].str.replace('[Ãx][^A-Za-z]+','',regex=True)


# In[281]:


#Counting the NaN values

df['restaurants'].isnull().sum()


# In[282]:


df.groupby('restaurants').count().head(10)


# #### Checking Null values after cleaning individual columns: 

# In[283]:


df.isnull().sum()


# In[284]:


df.info()


# In[285]:


df.shape


# In[286]:


# df.to_csv('./clean_zomato.csv')


# # Data Visualization

# 1.Restaurants delivering Online or not

# In[287]:


sns.countplot(x ='online_order', data=df)
plt.title('Online delivery')
plt.figure(figsize=(5,5))


#     by observing the graph we can say that around 30000 restaurants accepting the online order from the zomato and 20000 restaurants not accepting the online order from the zomato

# 2.Restaurants allowing table booking or not

# In[288]:


sns.countplot(x='booking', data=df)
plt.title('Table Booking')
plt.figure(figsize=(5,5))


#     from the graph we observe that more than 40000 restaurants not accepting the table order and only less than 10000 restaurants accepting the booking table order

# 3.Best City 

# In[289]:


a=df.groupby('city')['votes'].mean()
sorted_a=a.sort_values(ascending=False).head(5)

plt.figure(figsize=[10,6])
sns.barplot(y=sorted_a.index,x=sorted_a.values)


#     From the graph we observe that indiranagar is the best city according to the votes and after that Old Airport Road and so on

# In[290]:


a=df.groupby('city')['rating'].mean()
sorted_a=a.sort_values(ascending=False).head(20)

plt.figure(figsize=[10,6])
sns.barplot(y=sorted_a.index,x=sorted_a.values)


# 4.Best Location

# In[291]:


a=df.groupby('location')['rating'].mean()#.agg({'rate':'mean'})
sorted_a=a.sort_values(ascending=False).head(5)
plt.figure(figsize=[10,5])
sns.barplot(y=sorted_a.index,x=sorted_a.values)


#     From the graph we observe that lavelle road is the best location according to the rate and after that st.Marks Road and so on

# In[292]:


b=df.groupby('location')['votes'].median()
sorted_b=b.sort_values(ascending=False).head(5)

plt.figure(figsize=[10,5])
sns.barplot(x=sorted_b.index,y=sorted_b.values)


#     from the above graph we observe that Church street is the best location according to the votes

# 5.Total no of resturent in specific area

# In[293]:


plt.figure(figsize = (20, 15))
s = sns.countplot(y = df['city'].sort_values(ascending=False),data = df)


# In[294]:


print("Top 10 City with maximum no of resturent :", df['city'].nunique())
df['city'].value_counts().nlargest(10).to_frame().style.background_gradient(cmap='copper').set_precision(2)


#     1.BTM has highest no (around 3300) of restaurants and Karamangala 7th Block has second highest no (around 2932) of restaurants in Bangalore.
#     2.New BEL Road has least no (around 750) of restaurants in Bangalore.

# 5.Restaurant Type

# In[295]:


plt.figure(figsize=[20,20])
sns.countplot(y='rest_type', data=df)


#     from the graph we observe that more than 18500 are the Quick bytes restaurants, and around 10000 are the casual dinning restaurants and so on

# In[296]:


print("Top 10 resturent types are:")
df['rest_type'].value_counts().nlargest(10).to_frame().style.background_gradient(cmap='copper').set_precision(2)


#     from the graph we observe that more than 19000 are the Quick bytes restaurants, and around 10000 are the casual dinning restaurants and so on

# 6.Cost of Restaurant

# In[297]:


df['cost']=df['cost'].astype(int)


# As the 'cost' datatype is float so by using .astype() i changed the datatype of 'cost ' from float into int datatype

# In[298]:


plt.figure(figsize=[15,5])
c=df.groupby('restaurants')['cost'].mean().sort_values(ascending=False).head(15)
sns.barplot(y=c.index,x=c.values)


# In[299]:


print("Top 10 resturent with higest cost are:")
df.groupby('restaurants')['cost'].mean().sort_values(ascending=False).nlargest(10).to_frame().style.background_gradient(cmap='copper').set_precision(2)


#     'Le Cirque Signature- The Leela Palace ' having the largest cost of 6000, followed by Royal Afghan - ITC Windsor and so on

# In[300]:


print("Top 10 resturent with lowest cost are:")
df.groupby('restaurants')['cost'].mean().sort_values(ascending=False).nsmallest(10).to_frame().style.background_gradient(cmap='copper').set_precision(2)


# In[301]:


plt.figure(figsize=[20,10])
c=df.groupby('restaurants')['cost'].mean().sort_values(ascending=True).head(15)
sns.barplot(y=c.index,x=c.values)


#     'Srinidhi Sagar Deluxe' have the lowest cost of 40

# 7.Rating

# In[302]:


plt.figure(figsize=[10,5])
sns.distplot(df['rating'])


#     from the graph we observe that most of the restaurant types having rate in between more than 3 and less than 5 while maximum no of resturent have average rating of 3.7

# In[303]:


print("Top 10 resturent with Higest Rating are:")
df.groupby('restaurants')['rating'].mean().sort_values(ascending=False).nlargest(10).to_frame().style.background_gradient(cmap='copper').set_precision(2)


#     From this we can see that SantSpa Cuisine has higest rating and followed by Brewing Compant with rating of 4.9

# In[304]:


print("Top 10 resturent with lowest Rating are:")
df.groupby('restaurants')['rating'].mean().sort_values(ascending=False).nsmallest(10).to_frame().style.background_gradient(cmap='copper').set_precision(2)


#     From this we can see that Alibi-Maya has lowest rating of 1.8

# 8.Most famous restaurant chains in Bengaluru

# In[305]:


df[df['rating']>=4.5].head(3)


# I have taken the dataframe whose rate is greater than 4.5

# In[306]:


d=df[df['rating']>=4.5].groupby('restaurants')['votes'].sum()
sorted_d=d.sort_values(ascending=False).head(5)
sorted_d.values
plt.figure(figsize=[15,5])
sns.barplot(x=sorted_d.index,y=sorted_d.values)


#     From the graph we observe that Truffles is the most famous restaurant chains in bengaluru and after that hammered is there and so on

# 9.Type of services

# In[307]:


plt.figure(figsize=[15,5])
sns.countplot(x='categorie',data=df)


#     Delivery types of services are around 25000 , Dine-out are greater than 15000 and so on

# 10.relation between restaurant type and ratings

# In[308]:


df1=df.groupby('categorie')['rating'].mean().sort_values(ascending=False).head(10)


# In[309]:


df1.plot(kind='bar')


#     Drinks and Nightlife and Pubs and bars having the higher rating , and delivery having low rating

# 11.Top 15 Best Liked cuisines in Bangalore

# In[310]:


df['cuisines'].value_counts()[0:10]


# In[311]:


plt.figure(figsize=(18, 10))
sns.countplot(y='cuisines', data=df, order=df['cuisines'].value_counts().head(15).index)
plt.title('Top 10 Products which have the highest count', fontsize=16)
plt.xlabel('Count', fontsize=12)
plt.ylabel('Product Name', fontsize=12)

plt.show()


#     from the above graph we can see that North Indian is most liked food in Bangalore
