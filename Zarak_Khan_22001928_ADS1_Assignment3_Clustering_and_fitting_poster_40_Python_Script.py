#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pcakges
import numpy as np                                      
import pandas as pd 
get_ipython().system('pip install -q pycountry')
import pycountry                                        
import statsmodels.api as sm                                
import seaborn as sns                                   
import matplotlib.pyplot as plt                         
from sklearn.preprocessing import MinMaxScaler          
from sklearn.cluster import KMeans                      
from scipy.optimize import curve_fit                  


# In[ ]:


min_max_scaler=MinMaxScaler()  


# In[ ]:


#reading the data
#first 4 rows of the data contain some metadata so we skipped that rows
df = pd.read_csv("climate_indicators.csv", skiprows=4)

# Drop columns that are not required 
df = df.iloc[:, :-1]

# Create a dataset with County Name column name
country_df = df.set_index(["Country Name", "Indicator Name"])
country_df.drop(["Country Code", "Indicator Code"], axis=1, inplace=True)

# Taking transpose of the countries_Df
country_df= country_df.T


# In[ ]:


indicators = [
'CO2 emissions (kt)',
'Urban population growth (annual %)',
'Cereal yield (kg per hectare)',
'Arable land (% of land area)',
'Forest area (sq. km)',
]


# In[ ]:


# Filter the dataset for the required indicators
extracted_results = df[df["Indicator Name"].isin(indicators)]
# Extracting data for only countries we are interested in
countries_ls = [country.name for country in list(pycountry.countries)]
extracted_resutls = extracted_results[extracted_results["Country Name"].isin(countries_ls)]


# In[ ]:


#missing values imputation
data = extracted_resutls.fillna(method='ffill').fillna(method='bfill')
pivot_table = data.pivot_table(index='Country Name', columns='Indicator Name', values='2020')


# In[ ]:


plt.figure(figsize=(15,5))
sns.lineplot(pivot_table['Arable land (% of land area)'].sort_values(ascending=False)[0:10])
plt.title('Countries Standing by Arable land (% of land area)')


# In[ ]:


sns.displot(np.log(pivot_table['Forest area (sq. km)']))
plt.title('Distribution of Forest Area in Sq)')  #this data is plotted data log transformation


# In[ ]:


ub_growth=pivot_table['Urban population growth (annual %)'].sort_values(ascending=False)[0:10]
plt.figure(figsize=(15,5))
plt.title('Top 10 Countries with Highest Urbanization in 2020')
plt.bar(x=ub_growth.index,height=ub_growth.values)


# In[ ]:


sns.displot(pivot_table['CO2 emissions (kt)'])
plt.title('Distribution of CO2 for 2020')


# In[ ]:


# data Normalization
scaled_data = min_max_scaler.fit_transform(pivot_table.values)
min=np.min(pivot_table.values)
max=np.max(pivot_table.values)


# In[ ]:


# number of clusters
num_clusters = 3


# In[ ]:


#KMeans Clustering
kmeans = KMeans(n_clusters=num_clusters, random_state=22)
cluster_labels = kmeans.fit_predict(scaled_data)


# In[ ]:


# Add the cluster labels to the dataset
pivot_table["Cluster"] = cluster_labels
pivot_table.groupby("Cluster").mean()
labels = indicators
