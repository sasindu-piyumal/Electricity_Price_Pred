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


data = pd.read_csv("Z:\\Sasindu\\Data set\\electricity.csv", index_col=0, parse_dates=[0])


# In[3]:


df = pd.DataFrame(data)


# In[4]:


df.head()


# In[5]:


df.shape


# ## Data Cleaning

# In[6]:


df.info()


# ##### Here, ForecastWindProduction, SystemLoadEA, SMPEA, ORKTemperature, ORKWindspeed, CO2Intensity, ActualWindProduction, SystemLoadEP2, SMPEP2 should be numeric values. Hance, Converting them into numeric type is needed.

# In[7]:


cols_to_numeric = ['ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 'ORKTemperature', 'ORKWindspeed', 'CO2Intensity', 'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
for col in cols_to_numeric:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.info()


# In[8]:


missing_values =df.isnull().sum()
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


# ##### Because of having more missing values in ORKTemperature, ORKWindspeed columns, If they filled with mean or median, it can be error for model. So Null values were removed.

# ### Removing Outliers

# In[14]:


SMPEP2_out = (df_cleaned['SMPEP2'] > 0) | (df_cleaned['SMPEP2'] <= 550)


# In[15]:


df_cleaned = df_cleaned[SMPEP2_out]


# In[16]:


df_cleaned.shape


# In[17]:


fill_with_median = ['ForecastWindProduction','SystemLoadEA','SMPEA','ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
for col in fill_with_median:
    median_col = df_cleaned[col].median()
    df_cleaned[col].fillna(median_col,inplace=True)


# ##### Because of having skewed distribution on 'ForecastWindProduction','SystemLoadEA','SMPEA','ActualWindProduction', 'SystemLoadEP2', 'SMPEP2' columns. Missing values were filled with median.

# In[18]:


mean_CO2Intensity = df_cleaned['CO2Intensity'].mean()
df_cleaned['CO2Intensity'].fillna(mean_CO2Intensity,inplace=True)


# ##### CO2Intensity has normal dustribution. So null values were filled with mean value of the CO2Intensity.

# In[19]:


missing_values =df_cleaned.isna().sum()
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

# #### By studing Heatmap, Following observations can be gathered.
# ###### Highly positive correlation between month and weekofyear
# ###### Highly positive correlation between ForecastWindProduction and ActualWindProduction
# ###### Highly positive correlation between SystemLoadEA and SystemLoadEA2
# ###### Highly positive correlation between ORKWindspeed and ForecastWindProduction
# ###### Highly positive correlation between ORKWindspeed and ActualWindProduction

# ####  If some features in dataset have high correlation, it is often a good idea to consider ignoring or removing one of the highly correlated features.
# #### It can be cause to multicolinearity, increase model complexity, reduse model performence. Hance, one feature can be ignored.
# ###### ** Month might be more straightforward and useful in many contexts compared to WeekOfYear. So ,WeekOfYear was removed.
# ###### ** To get more accurate model, I think removing ForecastWindProduction column is suitable.
# ###### ** SystemLoadEA is forecasted national load and SystemLoadEA2 is actual national system load. hance, Removing SystemLoadEA is more accurate.
# ###### ** Because of model requament, ORKWindspeed or ActualWindProduction weren't removed. ForecastWindProduction has already removed.
# ###### ** Holiday column is also populated. Hence, Holiday column can be removed.

# In[21]:


df_new = df_cleaned.drop(columns=['Holiday','WeekOfYear','ForecastWindProduction','SystemLoadEA'])


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


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,3))
sns.lineplot(x=df_scaled.DateTime, y = df_scaled.Month)
plt.show()


# In[27]:


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,3))
sns.lineplot(x=df_scaled.DateTime, y = df_scaled.DayOfWeek)
plt.show()


# In[28]:


f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,3))
sns.lineplot(x=df_scaled.DateTime, y = df_scaled.Day)
plt.show()


# In[ ]:


from datetime import date

f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20,3))
sns.lineplot(x=df_scaled.DateTime, y = df_scaled.PeriodOfDay)
ax.set_xlim([date(2013,12,1),date(2014,1,1)])
plt.show()


# ##### By looking above graphs, We can see cyclic nature.Convert cyclic features into two continuous features using sine and cosine transformations to preserve the cyclic nature.Sine and Cosine functions help in representing the cyclic nature by mapping the feature to a circle. 

# ## Feature Engineering

# In[ ]:


def periodic_transform(df,variable):
    df_scaled[f"{variable}_SIN"] = np.sin(df_scaled[variable] / df_scaled[variable].max()*2*np.pi)
    df_scaled[f"{variable}_COS"] = np.cos(df_scaled[variable] / df_scaled[variable].max()*2*np.pi)
    return df_scaled


# In[ ]:


df_scaled = periodic_transform(df_scaled, 'DayOfWeek')
df_scaled = periodic_transform(df_scaled, 'Day')
df_scaled = periodic_transform(df_scaled, 'Month')
df_scaled = periodic_transform(df_scaled, 'PeriodOfDay')
df_scaled.head()


# ##### Hence, DayOfWeek, Day, Month, and PeriodOfDay  columns can be removed.

# In[ ]:


df_scaled = df_scaled.drop(columns=['DateTime','DayOfWeek','Day','Month','PeriodOfDay'])


# In[ ]:


df_scaled.columns


# In[ ]:


df_scaled.head()


# ### Splitting Data

# In[ ]:


x = df_scaled.drop(columns = 'SMPEP2', axis = 1)
y = df_scaled['SMPEP2']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=42)


# ### Scaling data

# #### HolidayFlag column 0 and 1 values. Hence, no need to encord this column.
# #### As most of columns having skewed distribution, minmaxscaler canbe applied for those columns such as Year,SMPEA,ORKTemperature,ORKWindspeed,ActualWindProduction,CO2Intensity,SystemLoadEP2,and SMPEP2
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
    model.fit(x_train_scaled,y_train)
    acc = model.score(x_test_scaled,y_test)
    print(str(model)+'-->'+str(acc))


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
rf = RandomForestRegressor()
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
sns.lineplot(x=range(len(final_df['Real'])) ,y=final_df['Real'],color='black',label='Real')
sns.lineplot(x=range(len(final_df['Prediction'])),y=final_df['Prediction'],color='red',label='Prediction')
ax.set_xlim([100,200])
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


pca = PCA(n_components= 11)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)


# In[ ]:


def model_acc_pca(model):
    model.fit(x_train_pca,y_train)
    acc = model.score(x_test_pca,y_test)
    print(str(model)+'-->'+str(acc))


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
rf = RandomForestRegressor()
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
sns.lineplot(x=range(len(final_df_pca['Real'])) ,y=final_df_pca['Real'],color='black',label='Real')
sns.lineplot(x=range(len(final_df_pca['Prediction'])),y=final_df_pca['Prediction'],color='red',label='Prediction')
ax.set_xlim([100,200])
plt.title('Real vs. Predictions')
plt.show()


# ##### Here, The best model can be defined as Random Forest Regressor. But here, model score is lower than modelling without PCA transformed Data. Hence, Previous senario can be applied.

# ### Defining modes for Data Without scaling and PCA

# In[ ]:


df_new.head()


# In[ ]:


df_new = df_new.reset_index()


# In[ ]:


df_new.head()


# In[ ]:


X = df_new.drop(columns = ['DateTime','SMPEP2'],axis=1)


# In[ ]:


Y = df_new['SMPEP2']


# In[ ]:


X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)


# In[ ]:


def model_acc_no_scale(model):
    model.fit(X_train,Y_train)
    acc = model.score(X_test,Y_test)
    print(str(model)+'-->'+str(acc))


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
rf = RandomForestRegressor()
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
sns.lineplot(x=range(len(final_dfr['Real'])) ,y=final_dfr['Real'],color='black',label='Real')
sns.lineplot(x=range(len(final_dfr['Prediction'])),y=final_dfr['Prediction'],color='red',label='Prediction')
ax.set_xlim([100,200])
plt.title('Real vs. Predictions')
plt.show()


# #### Here also, the best model is Random Forest Regressor and it's score is lower than modelling with the scaled data.

# ## Conclution

# ###### According To previous results, In this senario, Data can be used with out PCA transformiong. 
# ###### The most suitable model is Random Forest Regressor and it's model score (R^2) is 0.6502.
# ###### Here, the mean absolute error is 8.5611 and the mean squared error is 407.1928. those parameters are less values.
# ###### By studing the graph of "Real vs. Predictions", Similer patterns can be seen.
# ###### Hence, Random Forest Regressor can be defined as the most suitable model for the electricity data among "LinearRegression","Lasso","DecisionTreeRegressor","RandomForestRegressor", and "SVR".

# ## Time Series Walk-Forward Validation

# ### Implementation of walk-forward validation to simulate real-world prediction scenarios

# In[ ]:


# Re-load and prepare data with temporal ordering preserved
data_wf = pd.read_csv("electricity.csv", index_col=0, parse_dates=[0])
df_wf = pd.DataFrame(data_wf)

# Apply same preprocessing as before
cols_to_numeric = ['ForecastWindProduction', 'SystemLoadEA', 'SMPEA', 'ORKTemperature', 
                   'ORKWindspeed', 'CO2Intensity', 'ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
for col in cols_to_numeric:
    df_wf[col] = pd.to_numeric(df_wf[col], errors='coerce')

# Handle missing values
df_wf.replace(['', 'NA', 'N/A', None], np.nan, inplace=True)
df_wf = df_wf.dropna(subset=['ORKTemperature','ORKWindspeed'])

# Remove outliers for SMPEP2
SMPEP2_out = (df_wf['SMPEP2'] > 0) | (df_wf['SMPEP2'] <= 550)
df_wf = df_wf[SMPEP2_out]

# Fill remaining missing values
fill_with_median = ['ForecastWindProduction','SystemLoadEA','SMPEA','ActualWindProduction', 'SystemLoadEP2', 'SMPEP2']
for col in fill_with_median:
    median_col = df_wf[col].median()
    df_wf[col].fillna(median_col,inplace=True)

mean_CO2Intensity = df_wf['CO2Intensity'].mean()
df_wf['CO2Intensity'].fillna(mean_CO2Intensity,inplace=True)

# Drop same columns as before
df_wf = df_wf.drop(columns=['Holiday','WeekOfYear','ForecastWindProduction','SystemLoadEA'])

# Sort by datetime to ensure temporal ordering
df_wf = df_wf.sort_index()

# Reset index to have DateTime as a column
df_wf = df_wf.reset_index()


# In[ ]:


# Apply same feature engineering for cyclic features
def periodic_transform_wf(df, variable):
    df[f"{variable}_SIN"] = np.sin(df[variable] / df[variable].max()*2*np.pi)
    df[f"{variable}_COS"] = np.cos(df[variable] / df[variable].max()*2*np.pi)
    return df

df_wf = periodic_transform_wf(df_wf, 'DayOfWeek')
df_wf = periodic_transform_wf(df_wf, 'Day')
df_wf = periodic_transform_wf(df_wf, 'Month')
df_wf = periodic_transform_wf(df_wf, 'PeriodOfDay')


# In[ ]:


# Define walk-forward validation function
def walk_forward_validation(df, min_train_days=30, test_days=1, horizons=[2, 12, 48]):
    """
    Perform walk-forward validation with expanding windows
    
    Parameters:
    - df: DataFrame with features and target
    - min_train_days: Minimum training days required (default 30)
    - test_days: Number of days to predict in each step (default 1)
    - horizons: List of prediction horizons in 30-min intervals [1hr, 6hr, 24hr]
    
    Returns:
    - results: Dictionary containing predictions and metrics for each horizon
    """
    
    # Calculate periods
    periods_per_day = 48  # 30-minute intervals
    min_train_periods = min_train_days * periods_per_day
    test_periods = test_days * periods_per_day
    
    # Initialize results storage
    results = {
        'timestamps': [],
        'actual': [],
        'predictions': {h: [] for h in horizons},
        'metrics': {h: {'r2': [], 'mae': [], 'mse': [], 'window_start': [], 'window_end': []} for h in horizons}
    }
    
    # Prepare features and target
    feature_cols = [col for col in df.columns if col not in ['DateTime', 'SMPEP2']]
    X = df[feature_cols].values
    y = df['SMPEP2'].values
    timestamps = df['DateTime'].values
    
    # Initialize scaler
    scaler = MinMaxScaler()
    
    # Walk-forward validation loop
    for i in range(min_train_periods, len(df) - max(horizons)):
        # Expanding window for training
        X_train = X[:i]
        y_train = y[:i]
        
        # Fit scaler on training data
        X_train_scaled = scaler.fit_transform(X_train)
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions for different horizons
        for horizon in horizons:
            if i + horizon < len(df):
                # Get test data for this horizon
                X_test = X[i + horizon - 1:i + horizon]
                y_actual = y[i + horizon - 1:i + horizon]
                
                # Scale test data
                X_test_scaled = scaler.transform(X_test)
                
                # Predict
                y_pred = model.predict(X_test_scaled)
                
                # Store results
                results['predictions'][horizon].append(y_pred[0])
                
                # Calculate metrics every test_periods (daily)
                if i % test_periods == 0:
                    # Get all predictions and actuals for current window
                    recent_preds = results['predictions'][horizon][-test_periods:]
                    recent_actuals = y[i:i+test_periods]
                    
                    if len(recent_preds) == test_periods:
                        # Calculate metrics
                        from sklearn.metrics import r2_score
                        r2 = r2_score(recent_actuals, recent_preds)
                        mae = mean_absolute_error(recent_actuals, recent_preds)
                        mse = mean_squared_error(recent_actuals, recent_preds)
                        
                        # Store metrics
                        results['metrics'][horizon]['r2'].append(r2)
                        results['metrics'][horizon]['mae'].append(mae)
                        results['metrics'][horizon]['mse'].append(mse)
                        results['metrics'][horizon]['window_start'].append(timestamps[i-min_train_periods])
                        results['metrics'][horizon]['window_end'].append(timestamps[i])
        
        # Store timestamp and actual value
        if i == min_train_periods:
            results['timestamps'] = timestamps[i:]
            results['actual'] = y[i:]
        
        # Print progress every 30 days
        if (i - min_train_periods) % (30 * periods_per_day) == 0:
            days_processed = (i - min_train_periods) // periods_per_day
            print(f"Processed {days_processed} days of walk-forward validation...")
    
    return results


# In[ ]:


# Run walk-forward validation
print("Starting walk-forward validation...")
wf_results = walk_forward_validation(df_wf, min_train_days=30, test_days=1)
print("Walk-forward validation completed!")


# In[ ]:


# Calculate performance by time of day and day of week
def analyze_temporal_performance(df, results, horizons=[2, 12, 48]):
    """Analyze performance by time of day and day of week"""
    
    temporal_analysis = {}
    
    for horizon in horizons:
        predictions = results['predictions'][horizon]
        actual = results['actual'][:len(predictions)]
        timestamps = pd.to_datetime(results['timestamps'][:len(predictions)])
        
        # Create DataFrame for analysis
        analysis_df = pd.DataFrame({
            'timestamp': timestamps,
            'actual': actual,
            'prediction': predictions,
            'error': np.abs(np.array(actual) - np.array(predictions)),
            'hour': timestamps.hour,
            'day_of_week': timestamps.dayofweek,
            'month': timestamps.month,
            'quarter': timestamps.quarter
        })
        
        # Performance by hour of day
        hourly_perf = analysis_df.groupby('hour').agg({
            'error': ['mean', 'std'],
            'actual': 'count'
        }).round(2)
        
        # Performance by day of week
        daily_perf = analysis_df.groupby('day_of_week').agg({
            'error': ['mean', 'std'],
            'actual': 'count'
        }).round(2)
        
        # Performance by month
        monthly_perf = analysis_df.groupby('month').agg({
            'error': ['mean', 'std'],
            'actual': 'count'
        }).round(2)
        
        # Performance by quarter
        quarterly_perf = analysis_df.groupby('quarter').agg({
            'error': ['mean', 'std'],
            'actual': 'count'
        }).round(2)
        
        temporal_analysis[horizon] = {
            'hourly': hourly_perf,
            'daily': daily_perf,
            'monthly': monthly_perf,
            'quarterly': quarterly_perf,
            'full_df': analysis_df
        }
    
    return temporal_analysis


# In[ ]:


# Perform temporal analysis
temporal_perf = analyze_temporal_performance(df_wf, wf_results)
print("Temporal performance analysis completed!")


# In[ ]:


# Create visualization functions
def plot_metric_evolution(results, metric='mae', save_dir='plots/temporal_validation/'):
    """Plot evolution of metrics over time with confidence bands"""
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    horizons = [2, 12, 48]
    horizon_labels = ['1 Hour Ahead', '6 Hours Ahead', '24 Hours Ahead']
    
    for idx, (horizon, label) in enumerate(zip(horizons, horizon_labels)):
        ax = axes[idx]
        
        # Get metric values
        metric_values = results['metrics'][horizon][metric]
        timestamps = pd.to_datetime(results['metrics'][horizon]['window_end'])
        
        if len(metric_values) > 0:
            # Calculate rolling statistics
            window_size = 30  # 30-day rolling window
            metric_series = pd.Series(metric_values, index=timestamps)
            rolling_mean = metric_series.rolling(window=window_size, min_periods=1).mean()
            rolling_std = metric_series.rolling(window=window_size, min_periods=1).std()
            
            # Plot
            ax.plot(timestamps, metric_values, alpha=0.3, label='Daily Values')
            ax.plot(rolling_mean.index, rolling_mean.values, linewidth=2, label='30-Day Moving Average')
            
            # Confidence bands (±2 std)
            ax.fill_between(rolling_mean.index,
                           rolling_mean - 2*rolling_std,
                           rolling_mean + 2*rolling_std,
                           alpha=0.2, label='95% Confidence Band')
            
            ax.set_title(f'{metric.upper()} Evolution - {label}', fontsize=14)
            ax.set_xlabel('Date')
            ax.set_ylabel(metric.upper())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
    plt.tight_layout()
    plt.savefig(f'{save_dir}metric_evolution_{metric}.png', dpi=300, bbox_inches='tight')
    plt.show()


# In[ ]:


# Plot metric evolution for all metrics
for metric in ['mae', 'mse', 'r2']:
    plot_metric_evolution(wf_results, metric=metric)


# In[ ]:


def plot_temporal_patterns(temporal_perf, save_dir='plots/temporal_validation/'):
    """Plot performance patterns by time of day and day of week"""
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    horizons = [2, 12, 48]
    horizon_labels = ['1 Hour Ahead', '6 Hours Ahead', '24 Hours Ahead']
    
    # Time of day analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (horizon, label) in enumerate(zip(horizons, horizon_labels)):
        ax = axes[idx]
        hourly_errors = temporal_perf[horizon]['hourly']['error']['mean']
        hourly_std = temporal_perf[horizon]['hourly']['error']['std']
        
        hours = hourly_errors.index
        ax.bar(hours, hourly_errors.values, yerr=hourly_std.values, capsize=5, alpha=0.7)
        ax.set_title(f'MAE by Hour of Day - {label}')
        ax.set_xlabel('Hour')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_xticks(range(0, 24, 4))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}performance_by_hour.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Day of week analysis
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    
    for idx, (horizon, label) in enumerate(zip(horizons, horizon_labels)):
        ax = axes[idx]
        daily_errors = temporal_perf[horizon]['daily']['error']['mean']
        daily_std = temporal_perf[horizon]['daily']['error']['std']
        
        ax.bar(range(7), daily_errors.values, yerr=daily_std.values, capsize=5, alpha=0.7)
        ax.set_title(f'MAE by Day of Week - {label}')
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_xticks(range(7))
        ax.set_xticklabels(days)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}performance_by_day_of_week.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Seasonal analysis
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, (horizon, label) in enumerate(zip(horizons, horizon_labels)):
        # Monthly performance
        ax = axes[0, idx]
        monthly_errors = temporal_perf[horizon]['monthly']['error']['mean']
        monthly_std = temporal_perf[horizon]['monthly']['error']['std']
        
        months = monthly_errors.index
        ax.bar(months, monthly_errors.values, yerr=monthly_std.values, capsize=5, alpha=0.7)
        ax.set_title(f'MAE by Month - {label}')
        ax.set_xlabel('Month')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_xticks(range(1, 13))
        ax.grid(True, alpha=0.3)
        
        # Quarterly performance
        ax = axes[1, idx]
        quarterly_errors = temporal_perf[horizon]['quarterly']['error']['mean']
        quarterly_std = temporal_perf[horizon]['quarterly']['error']['std']
        
        quarters = quarterly_errors.index
        ax.bar(quarters, quarterly_errors.values, yerr=quarterly_std.values, capsize=5, alpha=0.7)
        ax.set_title(f'MAE by Quarter - {label}')
        ax.set_xlabel('Quarter')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_xticks(range(1, 5))
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}seasonal_performance.png', dpi=300, bbox_inches='tight')
    plt.show()


# In[ ]:


# Plot temporal patterns
plot_temporal_patterns(temporal_perf)


# In[ ]:


def identify_poor_performance_periods(results, temporal_perf, threshold_percentile=80):
    """Identify periods with poor performance and potential causes"""
    
    poor_performance = {}
    
    for horizon in [2, 12, 48]:
        df = temporal_perf[horizon]['full_df']
        
        # Calculate error threshold (80th percentile)
        error_threshold = df['error'].quantile(threshold_percentile/100)
        
        # Find periods with high errors
        high_error_mask = df['error'] > error_threshold
        high_error_periods = df[high_error_mask]
        
        # Analyze patterns in poor performance
        poor_perf_analysis = {
            'threshold': error_threshold,
            'count': len(high_error_periods),
            'percentage': len(high_error_periods) / len(df) * 100,
            'by_hour': high_error_periods.groupby('hour').size(),
            'by_day': high_error_periods.groupby('day_of_week').size(),
            'by_month': high_error_periods.groupby('month').size(),
            'top_errors': high_error_periods.nlargest(10, 'error')[['timestamp', 'actual', 'prediction', 'error']]
        }
        
        poor_performance[horizon] = poor_perf_analysis
        
        # Print summary
        horizon_label = {2: '1 Hour', 12: '6 Hours', 48: '24 Hours'}[horizon]
        print(f"\nPoor Performance Analysis - {horizon_label} Ahead:")
        print(f"  Error threshold (80th percentile): {error_threshold:.2f}")
        print(f"  Number of high-error periods: {poor_perf_analysis['count']} ({poor_perf_analysis['percentage']:.1f}%)")
        print(f"  Most common hours: {poor_perf_analysis['by_hour'].nlargest(3).to_dict()}")
        print(f"  Most common days: {poor_perf_analysis['by_day'].nlargest(3).to_dict()}")
        print(f"  Most common months: {poor_perf_analysis['by_month'].nlargest(3).to_dict()}")
    
    return poor_performance


# In[ ]:


# Identify poor performance periods
poor_perf = identify_poor_performance_periods(wf_results, temporal_perf)


# In[ ]:


def compare_with_standard_split(df_wf, wf_results):
    """Compare walk-forward validation results with standard train/test split"""
    
    # Prepare data for standard split
    feature_cols = [col for col in df_wf.columns if col not in ['DateTime', 'SMPEP2']]
    X = df_wf[feature_cols].values
    y = df_wf['SMPEP2'].values
    
    # Standard 80/20 split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled, y_train)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    from sklearn.metrics import r2_score
    standard_metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'mse': mean_squared_error(y_test, y_pred),
        'sample_size': len(y_test)
    }
    
    # Calculate walk-forward average metrics
    wf_avg_metrics = {}
    for horizon in [2, 12, 48]:
        if len(wf_results['metrics'][horizon]['r2']) > 0:
            wf_avg_metrics[horizon] = {
                'r2': np.mean(wf_results['metrics'][horizon]['r2']),
                'mae': np.mean(wf_results['metrics'][horizon]['mae']),
                'mse': np.mean(wf_results['metrics'][horizon]['mse']),
                'r2_std': np.std(wf_results['metrics'][horizon]['r2']),
                'mae_std': np.std(wf_results['metrics'][horizon]['mae']),
                'mse_std': np.std(wf_results['metrics'][horizon]['mse'])
            }
    
    # Print comparison
    print("\n=== Comparison: Walk-Forward vs Standard Train/Test Split ===\n")
    print("Standard Train/Test Split (80/20):")
    print(f"  R² Score: {standard_metrics['r2']:.4f}")
    print(f"  MAE: {standard_metrics['mae']:.2f}")
    print(f"  MSE: {standard_metrics['mse']:.2f}")
    print(f"  Test Size: {standard_metrics['sample_size']} samples\n")
    
    print("Walk-Forward Validation (Average ± Std):")
    for horizon in [2, 12, 48]:
        horizon_label = {2: '1 Hour', 12: '6 Hours', 48: '24 Hours'}[horizon]
        if horizon in wf_avg_metrics:
            m = wf_avg_metrics[horizon]
            print(f"\n  {horizon_label} Ahead:")
            print(f"    R² Score: {m['r2']:.4f} ± {m['r2_std']:.4f}")
            print(f"    MAE: {m['mae']:.2f} ± {m['mae_std']:.2f}")
            print(f"    MSE: {m['mse']:.2f} ± {m['mse_std']:.2f}")
            
            # Calculate performance variance
            variance_pct = (m['mae_std'] / m['mae']) * 100
            print(f"    Performance Variance: {variance_pct:.1f}%")
    
    return standard_metrics, wf_avg_metrics


# In[ ]:


# Compare with standard split
standard_metrics, wf_metrics = compare_with_standard_split(df_wf, wf_results)


# In[ ]:


# Save walk-forward results to CSV
def save_wf_results(wf_results, temporal_perf, standard_metrics, wf_metrics):
    """Save walk-forward validation results"""
    
    # Prepare summary DataFrame
    summary_data = []
    
    # Add standard split results
    summary_data.append({
        'method': 'Standard_Split',
        'horizon': 'N/A',
        'r2_mean': standard_metrics['r2'],
        'mae_mean': standard_metrics['mae'],
        'mse_mean': standard_metrics['mse'],
        'r2_std': 0,
        'mae_std': 0,
        'mse_std': 0,
        'performance_variance': 0
    })
    
    # Add walk-forward results
    for horizon in [2, 12, 48]:
        if horizon in wf_metrics:
            m = wf_metrics[horizon]
            horizon_label = {2: '1_hour', 12: '6_hours', 48: '24_hours'}[horizon]
            summary_data.append({
                'method': 'Walk_Forward',
                'horizon': horizon_label,
                'r2_mean': m['r2'],
                'mae_mean': m['mae'],
                'mse_mean': m['mse'],
                'r2_std': m['r2_std'],
                'mae_std': m['mae_std'],
                'mse_std': m['mse_std'],
                'performance_variance': (m['mae_std'] / m['mae']) * 100
            })
    
    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('walk_forward_results.csv', index=False)
    
    print("\nResults saved to walk_forward_results.csv")
    
    # Also save detailed metrics time series
    for horizon in [2, 12, 48]:
        horizon_label = {2: '1_hour', 12: '6_hours', 48: '24_hours'}[horizon]
        if len(wf_results['metrics'][horizon]['r2']) > 0:
            detailed_metrics = pd.DataFrame({
                'window_end': wf_results['metrics'][horizon]['window_end'],
                'r2': wf_results['metrics'][horizon]['r2'],
                'mae': wf_results['metrics'][horizon]['mae'],
                'mse': wf_results['metrics'][horizon]['mse']
            })
            detailed_metrics.to_csv(f'walk_forward_metrics_{horizon_label}.csv', index=False)
    
    print("Detailed metric time series saved to walk_forward_metrics_*.csv files")


# In[ ]:


# Save all results
save_wf_results(wf_results, temporal_perf, standard_metrics, wf_metrics)


# In[ ]:


# Create final summary visualization
def create_summary_visualization(wf_results, standard_metrics, wf_metrics, save_dir='plots/temporal_validation/'):
    """Create a comprehensive summary visualization"""
    
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Metric comparison plot
    ax1 = plt.subplot(2, 3, 1)
    metrics_names = ['R²', 'MAE', 'MSE']
    standard_values = [standard_metrics['r2'], standard_metrics['mae'], standard_metrics['mse']]
    
    x = np.arange(len(metrics_names))
    width = 0.2
    
    ax1.bar(x - 1.5*width, standard_values, width, label='Standard Split', alpha=0.8)
    
    for i, horizon in enumerate([2, 12, 48]):
        if horizon in wf_metrics:
            values = [wf_metrics[horizon]['r2'], wf_metrics[horizon]['mae'], wf_metrics[horizon]['mse']]
            ax1.bar(x + (i-0.5)*width, values, width, 
                   label=f'WF {[1,6,24][i]}h', alpha=0.8)
    
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Value')
    ax1.set_title('Model Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics_names)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance stability over time
    ax2 = plt.subplot(2, 3, (2, 3))
    for horizon, label in zip([2, 12, 48], ['1h', '6h', '24h']):
        if len(wf_results['metrics'][horizon]['mae']) > 0:
            timestamps = pd.to_datetime(wf_results['metrics'][horizon]['window_end'])
            mae_values = wf_results['metrics'][horizon]['mae']
            ax2.plot(timestamps, mae_values, label=label, alpha=0.7)
    
    ax2.set_xlabel('Date')
    ax2.set_ylabel('MAE')
    ax2.set_title('MAE Evolution Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Performance variance comparison
    ax3 = plt.subplot(2, 3, 4)
    horizons_labels = ['1 Hour', '6 Hours', '24 Hours']
    variances = []
    
    for horizon in [2, 12, 48]:
        if horizon in wf_metrics:
            variance = (wf_metrics[horizon]['mae_std'] / wf_metrics[horizon]['mae']) * 100
            variances.append(variance)
    
    ax3.bar(horizons_labels, variances, alpha=0.7)
    ax3.axhline(y=20, color='r', linestyle='--', label='20% Threshold')
    ax3.set_xlabel('Prediction Horizon')
    ax3.set_ylabel('Performance Variance (%)')
    ax3.set_title('Model Stability Assessment')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. R² distribution
    ax4 = plt.subplot(2, 3, 5)
    for horizon, label in zip([2, 12, 48], ['1h', '6h', '24h']):
        if len(wf_results['metrics'][horizon]['r2']) > 0:
            r2_values = wf_results['metrics'][horizon]['r2']
            ax4.hist(r2_values, bins=30, alpha=0.5, label=label, density=True)
    
    ax4.set_xlabel('R² Score')
    ax4.set_ylabel('Density')
    ax4.set_title('R² Score Distribution')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. Summary statistics table
    ax5 = plt.subplot(2, 3, 6)
    ax5.axis('tight')
    ax5.axis('off')
    
    # Create summary table
    table_data = [['Metric', 'Standard', '1h WF', '6h WF', '24h WF']]
    
    # R² row
    row = ['R²', f"{standard_metrics['r2']:.3f}"]
    for h in [2, 12, 48]:
        if h in wf_metrics:
            row.append(f"{wf_metrics[h]['r2']:.3f}±{wf_metrics[h]['r2_std']:.3f}")
    table_data.append(row)
    
    # MAE row
    row = ['MAE', f"{standard_metrics['mae']:.1f}"]
    for h in [2, 12, 48]:
        if h in wf_metrics:
            row.append(f"{wf_metrics[h]['mae']:.1f}±{wf_metrics[h]['mae_std']:.1f}")
    table_data.append(row)
    
    # Variance row
    row = ['Var%', 'N/A']
    for h in [2, 12, 48]:
        if h in wf_metrics:
            var = (wf_metrics[h]['mae_std'] / wf_metrics[h]['mae']) * 100
            row.append(f"{var:.1f}%")
    table_data.append(row)
    
    table = ax5.table(cellText=table_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax5.set_title('Summary Statistics', pad=20)
    
    plt.suptitle('Walk-Forward Validation Summary Report', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'{save_dir}walk_forward_summary.png', dpi=300, bbox_inches='tight')
    plt.show()


# In[ ]:


# Create summary visualization
create_summary_visualization(wf_results, standard_metrics, wf_metrics)

print("\n=== Walk-Forward Validation Complete ===")
print("\nKey Findings:")
print("1. Walk-forward validation implemented with expanding windows (minimum 30 days)")
print("2. Multi-horizon predictions created for 1, 6, and 24 hours ahead")
print("3. Performance metrics tracked over time showing temporal patterns")
print("4. Model stability assessed - check performance variance against 20% threshold")
print("5. Periods of poor performance identified and analyzed")
print("6. Results compared with standard train/test split approach")
print("\nAll results saved to CSV files and visualizations saved to plots/temporal_validation/")
