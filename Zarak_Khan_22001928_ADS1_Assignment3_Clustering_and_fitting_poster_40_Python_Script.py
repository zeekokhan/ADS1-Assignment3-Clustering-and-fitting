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