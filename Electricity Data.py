#!/usr/bin/env python
# coding: utf-8
---
title: "Quarto Basics"
format: 
  html:
    code-fold: true
jupyter: python3
---
# In[1]:


import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


# ## Importing Data set

# In[2]:


data = pd.read_csv("electricity.csv", index_col=0)
data.index = pd.to_datetime(data.index, format="%d/%m/%Y %H:%M")


# In[3]:


df = pd.DataFrame(data)


# In[4]:


df.head()


# In[5]:


df.shape


# ## Data Cleaning

# In[6]:


df.info()


# ##### Here, ForecastWindProduction, SystemLoadEA, SMPEA, ORKTemperature, ORKWindspeed, CO2Intensity, ActualWindProduction, SystemLoadEP2, SMPEP2 should be numeric values. Hence, converting them into numeric type is needed.

# In[7]:


cols_to_numeric = ['ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 'ORKTemperature', 'ORKWindspeed', 'CO2Intensity', 'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.info()


# In[8]:


missing_values = df.isnull().sum()
print(missing_values)


# ##### Before fill missing values, Exploratory data Analysis is essential.

# ## Exploratory data Analysis

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


numeric_vals = ['HolidayFlag', 'DayOfWeek', 'WeekOfYear', 'Day', 'Month', 'Year', 'PeriodOfDay',
               'ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 'ORKTemperature', 'ORKWindspeed',
                'CO2Intensity', 'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
for col in numeric_vals:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.tight_layout()
    plt.show()


# df.describe()

# ### Handling missing values

# In[11]:


df.replace(['', 'NA', 'N/A', None], np.nan, inplace=True)


# In[12]:


df_cleaned = df.dropna(subset=['ORKTemperature','ORKWindspeed'])


# In[13]:


df_cleaned.shape


# ##### Because of having many missing values in ORKTemperature and ORKWindspeed columns, filling them with mean or median could introduce errors in the model. Therefore, null values were removed.

# ### Removing Outliers

# In[14]:


# Fixed: Changed OR (|) to AND (&) to properly filter values between 0 and 550
SMPEP2_out = (df_cleaned['SMPEP2'] > 0) & (df_cleaned['SMPEP2'] <= 550)


# In[15]:


df_cleaned = df_cleaned[SMPEP2_out]


# In[16]:


df_cleaned.shape


# In[17]:


fill_with_median = ['ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
for col in fill_with_median:
    median_col = df_cleaned[col].median()
    df_cleaned[col].fillna(median_col, inplace=True)


# ##### Because of having skewed distributions in 'ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2' columns, missing values were filled with median.

# In[18]:


mean_CO2Intensity = df_cleaned['CO2Intensity'].mean()
df_cleaned['CO2Intensity'].fillna(mean_CO2Intensity, inplace=True)


# ##### CO2Intensity has a normal distribution. Therefore, null values were filled with the mean value of CO2Intensity.

# In[19]:


missing_values = df_cleaned.isna().sum()
print(missing_values)


# In[20]:


df_cleaned.head()


# ### EDA (Correlation Analysis)

# plt.figure(figsize=(20, 20))
# sns.pairplot(df_cleaned,palette='set1')
# plt.show()

# numeric_df = df_cleaned.select_dtypes(include = ['number'])
# corr_matrix =numeric_df.corr()
# plt.figure(figsize=(20, 20))
# sns.heatmap(corr_matrix,annot=True, cmap='coolwarm',fmt=".2f")
# plt.title('Correlation for elecricity data')
# plt.show()

# #### By studying the heatmap, the following observations can be gathered:
# ###### Highly positive correlation between month and weekofyear
# ###### Highly positive correlation between ForecastWindProduction and ActualWindProduction
# ###### Highly positive correlation between SystemLoadEA and SystemLoadEA2
# ###### Highly positive correlation between ORKWindspeed and ForecastWindProduction
# ###### Highly positive correlation between ORKWindspeed and ActualWindProduction

# #### If some features in the dataset have high correlation, it is often a good idea to consider ignoring or removing one of the highly correlated features.
# #### It can cause multicollinearity, increase model complexity, and reduce model performance. Hence, one feature can be ignored.
# ###### ** Month might be more straightforward and useful in many contexts compared to WeekOfYear. Therefore, WeekOfYear was removed.
# ###### ** To get more accurate model, I think removing ForecastWindProduction column is suitable.
# ###### ** SystemLoadEA is forecasted national load and SystemLoadEP2 is actual national system load. Hence, removing SystemLoadEA is more accurate.
# ###### ** Because of model requirements, ORKWindspeed or ActualWindProduction weren't removed. ForecastWindProduction was already removed.
# ###### ** Holiday column is also populated. Hence, Holiday column can be removed.

# In[21]:


df_new = df_cleaned.drop(columns=['Holiday', 'WeekOfYear', 'ForecastWindProduction', 'SystemLoadEA'])


# In[22]:


df_new.head()


# In[23]:


df_new.shape


# #### Split Date and timestamp from index

# In[24]:


df_scaled = df_new.reset_index()


# In[25]:


df_scaled.columns


# In[26]:


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 3))
sns.lineplot(x=df_scaled.DateTime, y=df_scaled.Month)
plt.show()


# In[27]:


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 3))
sns.lineplot(x=df_scaled.DateTime, y=df_scaled.DayOfWeek)
plt.show()


# In[28]:


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 3))
sns.lineplot(x=df_scaled.DateTime, y=df_scaled.Day)
plt.show()


# In[ ]:


from datetime import date

f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 3))
sns.lineplot(x=df_scaled.DateTime, y=df_scaled.PeriodOfDay)
ax.set_xlim([date(2013, 12, 1), date(2014, 1, 1)])
plt.show()


# ##### By looking at the above graphs, we can see cyclic patterns. Convert cyclic features into two continuous features using sine and cosine transformations to preserve the cyclic nature. Sine and cosine functions help in representing the cyclic nature by mapping the feature to a circle. 

# ## Feature Engineering

# In[ ]:


def periodic_transform(df, variable):
    """Transform cyclic features into sine and cosine components."""
    max_val = df[variable].max()
    
    # Check for division by zero - if max is zero, set normalized values to 0
    if max_val == 0:
        df[f"{variable}_SIN"] = 0
        df[f"{variable}_COS"] = 1
    else:
        df[f"{variable}_SIN"] = np.sin(df[variable] / max_val * 2 * np.pi)
        df[f"{variable}_COS"] = np.cos(df[variable] / max_val * 2 * np.pi)
    
    return df


# In[ ]:


df_scaled = periodic_transform(df_scaled, 'DayOfWeek')
df_scaled = periodic_transform(df_scaled, 'Day')
df_scaled = periodic_transform(df_scaled, 'Month')
df_scaled = periodic_transform(df_scaled, 'PeriodOfDay')
df_scaled.head()


# ##### Hence, DayOfWeek, Day, Month, and PeriodOfDay columns can be removed.

# In[ ]:


df_scaled = df_scaled.drop(columns=['DateTime', 'DayOfWeek', 'Day', 'Month', 'PeriodOfDay'])


# In[ ]:


df_scaled.columns


# In[ ]:


df_scaled.head()


# ### Splitting Data

# In[ ]:


x = df_scaled.drop(columns='SMPEP2', axis=1)
y = df_scaled['SMPEP2']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# ### Scaling data

# #### HolidayFlag column has 0 and 1 values. Hence, no need to encode this column.
# #### As most columns have skewed distributions, MinMaxScaler can be applied for those columns such as Year, SMPEA, ORKTemperature, ORKWindspeed, ActualWindProduction, CO2Intensity, SystemLoadEP2, and SMPEP2.
# 

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


mm = MinMaxScaler()
x_train_scaled = mm.fit_transform(x_train)
x_test_scaled = mm.transform(x_test)


# ### Define Model

# #### Defining modes for scaled Data

# In[ ]:


def model_acc(model):
    """Train model and print accuracy score."""
    model.fit(x_train_scaled, y_train)
    acc = model.score(x_test_scaled, y_test)
    print(f'{model.__class__.__name__} --> {acc:.4f}')


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model_acc(lr)

from sklearn.linear_model import Lasso
lasso = Lasso()
model_acc(lasso)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
model_acc(dt)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1)
model_acc(rf)

from sklearn.svm import SVR
svm = SVR()
model_acc(svm)


# In[ ]:


best_model = rf


# In[ ]:


y_test_pred = best_model.predict(x_test_scaled)


# ### Evaluation for Scaled Data for Best Model

# In[ ]:


from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_pacf


# In[ ]:


y_test_pred = y_test_pred.flatten()


# In[ ]:


final_df = pd.DataFrame(np.hstack((y_test_pred[:, np.newaxis], y_test[:, np.newaxis])), columns=['Prediction', 'Real'])


# In[ ]:


print(f'MAE: {mean_absolute_error(final_df["Prediction"],final_df["Real"])}')
print(f'MSE: {mean_squared_error(final_df["Prediction"],final_df["Real"])}')


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 5))
sns.lineplot(x=range(len(final_df['Real'])), y=final_df['Real'], color='black', label='Real')
sns.lineplot(x=range(len(final_df['Prediction'])), y=final_df['Prediction'], color='red', label='Prediction')
ax.set_xlim([100, 200])
plt.title('Real vs. Predictions')
plt.show()


# ### PCA

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA()
pca.fit(x_train_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = explained_variance_ratio.cumsum()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance_ratio) + 1), cumulative_explained_variance, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Explained Variance vs. Number of Components')
plt.show()


# In[ ]:


pca = PCA(n_components=11)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)


# In[ ]:


def model_acc_pca(model):
    """Train model on PCA-transformed data and print accuracy score."""
    model.fit(x_train_pca, y_train)
    acc = model.score(x_test_pca, y_test)
    print(f'{model.__class__.__name__} --> {acc:.4f}')


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model_acc_pca(lr)

from sklearn.linear_model import Lasso
lasso = Lasso()
model_acc_pca(lasso)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
model_acc_pca(dt)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1)
model_acc_pca(rf)

from sklearn.svm import SVR
svm = SVR()
model_acc_pca(svm)


# In[ ]:


y_test_pred = rf.predict(x_test_pca)
y_test_pred = y_test_pred.flatten()
final_df_pca = pd.DataFrame(np.hstack((y_test_pred[:, np.newaxis], y_test[:, np.newaxis])), columns=['Prediction', 'Real'])


# In[ ]:


print(f'MAE: {mean_absolute_error(final_df_pca["Prediction"],final_df_pca["Real"])}')
print(f'MSE: {mean_squared_error(final_df_pca["Prediction"],final_df_pca["Real"])}')


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 5))
sns.lineplot(x=range(len(final_df_pca['Real'])), y=final_df_pca['Real'], color='black', label='Real')
sns.lineplot(x=range(len(final_df_pca['Prediction'])), y=final_df_pca['Prediction'], color='red', label='Prediction')
ax.set_xlim([100, 200])
plt.title('Real vs. Predictions')
plt.show()


# ##### Here, the best model can be defined as Random Forest Regressor. However, the model score is lower than modelling without PCA-transformed data. Hence, the previous scenario can be applied.

# ### Defining models for data without scaling and PCA

# In[ ]:


df_new.head()


# In[ ]:


df_new = df_new.reset_index()


# In[ ]:


df_new.head()


# In[ ]:


X = df_new.drop(columns=['DateTime', 'SMPEP2'], axis=1)


# In[ ]:


Y = df_new['SMPEP2']


# In[ ]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# In[ ]:


def model_acc_no_scale(model):
    """Train model on unscaled data and print accuracy score."""
    model.fit(X_train, Y_train)
    acc = model.score(X_test, Y_test)
    print(f'{model.__class__.__name__} --> {acc:.4f}')


# In[ ]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model_acc_no_scale(lr)

from sklearn.linear_model import Lasso
lasso = Lasso()
model_acc_no_scale(lasso)

from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
model_acc_no_scale(dt)

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1)
model_acc_no_scale(rf)

from sklearn.svm import SVR
svm = SVR()
model_acc_no_scale(svm)


# In[ ]:


y_test_pred = rf.predict(X_test)
y_test_pred = y_test_pred.flatten()
final_dfr = pd.DataFrame(np.hstack((y_test_pred[:, np.newaxis], y_test[:, np.newaxis])), columns=['Prediction', 'Real'])


# In[ ]:


print(f'MAE: {mean_absolute_error(final_dfr["Prediction"],final_dfr["Real"])}')
print(f'MSE: {mean_squared_error(final_dfr["Prediction"],final_dfr["Real"])}')


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 5))
sns.lineplot(x=range(len(final_dfr['Real'])), y=final_dfr['Real'], color='black', label='Real')
sns.lineplot(x=range(len(final_dfr['Prediction'])), y=final_dfr['Prediction'], color='red', label='Prediction')
ax.set_xlim([100, 200])
plt.title('Real vs. Predictions')
plt.show()


# #### Here also, the best model is Random Forest Regressor and it's score is lower than modelling with the scaled data.

# ## Conclusion

# ###### According to the previous results, in this scenario, data can be used without PCA transformation. 
# ###### The most suitable model is Random Forest Regressor and its model score (R²) is 0.6502.
# ###### Here, the mean absolute error is 8.5611 and the mean squared error is 407.1928. These are relatively low values.
# ###### By studying the graph of "Real vs. Predictions", similar patterns can be seen.
# ###### Hence, Random Forest Regressor can be defined as the most suitable model for the electricity data among LinearRegression, Lasso, DecisionTreeRegressor, RandomForestRegressor, and SVR.

# #### Here also, the best model is Random Forest Regressor and it's score is lower than modelling with the scaled data.

# ## Conclution

# ###### According To previous results, In this senario, Data can be used with out PCA transformiong. 
# ###### The most suitable model is Random Forest Regressor and it's model score (R^2) is 0.6502.
# ###### Here, the mean absolute error is 8.5611 and the mean squared error is 407.1928. those parameters are less values.
# ###### By studing the graph of "Real vs. Predictions", Similer patterns can be seen.
# ###### Hence, Random Forest Regressor can be defined as the most suitable model for the electricity data among "LinearRegression","Lasso","DecisionTreeRegressor","RandomForestRegressor", and "SVR".