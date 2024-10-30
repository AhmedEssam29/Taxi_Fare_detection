# Taxi Fare Detection 

This project is aimed at predicting taxi fares in New York City by leveraging a machine learning model. We’ll begin with data loading and sampling, then move through data cleaning, outlier removal, and feature engineering, and finally explore potential model selection.

## Here’s a summary of the primary steps:


### 1. Data Preparation and Importing Libraries

**Objective:** Load the dataset and reduce memory usage.

**Libraries:** Basic libraries include pandas and numpy for data handling, and matplotlib/seaborn for visualization aesthetics.

**Sampling:** To optimize performance, we’re working with a sample size of 10% from the dataset, using a function to skip rows randomly.
### 2. Data Cleaning
**Drop NA Values:** First, check for any missing values and remove rows containing them.

**Filter Rows:** Adjust the training dataset by filtering out unrealistic values (like extreme fare amounts or out-of-bound geographical coordinates). This helps remove noise that could impact the model's accuracy.
### 3. Feature Engineering
**Distance Calculation:** Use the haversine formula to calculate the distance between pickup and dropoff points based on latitude and longitude.

**Datetime Extraction:** Extract important time-based features from any datetime columns, such as the hour of the day, day of the week, and month.

**Landmarks Proximity (Optional):** An additional feature could include calculating proximity to popular NYC landmarks (e.g., JFK, Times Square) for potential impact on fare prediction.

# Code Implementation Highlights
#### Here's a streamlined version of your code to give an organized structure:


```bash

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from datetime import timedelta

# Set aesthetics and display options
pd.set_option("display.max_columns", None)
sns.set_style('darkgrid')
random.seed(123)

# Sampling and Data Import
sample_size = 0.1
dtypes = {'fare_amount': 'float32', 'pickup_longitude': 'float32', 'pickup_latitude': 'float32', 
          'dropoff_longitude': 'float32', 'passenger_count': 'float32'}

train = pd.read_csv(
    "E:/taxi_fare/new-york-city-taxi-fare-prediction/train.csv", 
    skiprows=lambda i: i > 0 and random.random() > sample_size, dtype=dtypes
)
train.dropna(inplace=True)

# Define helper functions
def remove_outliers(df):
    conditions = (
        (df['fare_amount'] >= 1) & (df['fare_amount'] <= 500) &
        (df['pickup_longitude'] >= -75) & (df['pickup_longitude'] <= -72) &
        (df['dropoff_longitude'] >= -75) & (df['dropoff_longitude'] <= -72) &
        (df['pickup_latitude'] >= 40) & (df['pickup_latitude'] <= 42) &
        (df['dropoff_latitude'] >= 40) & (df['dropoff_latitude'] <= 42) &
        (df['passenger_count'] >= 1) & (df['passenger_count'] <= 6)
    )
    return df[conditions]

train = remove_outliers(train)

# Haversine distance function
def haversine_np(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    return 6367 * 2 * np.arcsin(np.sqrt(a))

train['trip_distance'] = haversine_np(
    train['pickup_longitude'], train['pickup_latitude'],
    train['dropoff_longitude'], train['dropoff_latitude']
)

# Datetime extraction function
def extract_from_date(df):
    for col in df.columns:
        if 'date' in col:
            df[col] = pd.to_datetime(df[col]) - timedelta(hours=4)
            df[col] = df[col].dt.tz_localize(None)
            df['year'] = df[col].dt.year
            df['month'] = df[col].dt.month
            df['day'] = df[col].dt.day
            df['day_of_week'] = df[col].dt.dayofweek
            df['week_of_year'] = df[col].dt.isocalendar().week.astype(int)
            df['hour'] = df[col].dt.hour

extract_from_date(train)

# Descriptive statistics
print(train.describe())
```

### 4. Further Steps for Model Training
**Data Split:** Use train-test split for model validation.

**Model Selection:** Algorithms like Linear Regression, Random Forest, or Gradient Boosting are typically effective for fare prediction.

**Hyperparameter Tuning:** Use GridSearchCV to refine model parameters for improved accuracy.

**Evaluation:** Evaluate the model’s performance using RMSE or MAE.



