# FitFood sales analysis and predictions

**Semester Project for Decision Systems 2020/2021 Course**

The goal of the competition is to create an efficient model for predicting whether the total 14-days sales of a particular product, offered by the Fitfood company at one of their FitBoxy locations in Poland, will exceed four pieces

## Task description

Provided data describe a short-term sales history of products at various point of sales (PoS). The target attribute will_it_sell tells if in the following 14 day period the total sales of a given product at a particular location will be at least 4 pcs. 

The data is very similar to the one from the second graded task, with the difference that the sets in this challenge do not contain any random probes (which were deliberately added to the data from the second graded task for evaluation purposes). 

The data tables are provided as two CSV files with the ';' separator sign. They can be downloaded after the registration for the challenge. Both files (training and test sets) have exactly the same format but all the values from the will_it_sell column in the test set are missing.

The evaluation metric will be AUC. During the challenge, your solutions will be evaluated on a small fraction of the test set, and your best preliminary AUC score will be displayed on the public Leaderboard. 

The submission format: the solutions need to be submitted as text files with predictions. The file should have exactly the same number of rows as the test data table. In each row, it should contain exactly one real number expressing the likeliness that the correct target value for the corresponding test set instance is 1.

## Solution

#### Author of the solution: Krzysztof Piesiewicz

#### Contents of the notebook:
  - [Loading the data](#Loading-the-data)
  - [Preliminary analysis and preprocessing](#Preliminary-analysis-and-preprocessing)
    - [Size attribute](#Size-attribute)
    - [Storage temperature attribute](#Storage-temperature-attribute)
    - [Generalizing some discrete attributes](#Generalizing-some-discrete-attributes)
    - [Preprocessing the data](#Preprocessing-the-data)
  - [Spliting data for training and validation](#Spliting-data-for-training-and-validation)
  - [Chalange test set preparation and answer saver](#Chalange-test-set-preparation-and-answer-saver)
  - [Features selection](#Features-selection)
    - [Analysis of correlations with target value](#Analysis-of-correlations-with-target-value)
    - [Features importance with random forrest](#Features-importance-with-random-forrest)
  - [Training on all the features](#Training-on-all-the-features)
    - [More samples in a leaf](#More-samples-in-a-leaf)
    - [Less samples in a leaf](#Less-samples-in-a-leaf)
    - [Training each estimator on more samples](#Training-each-estimator-on-more-samples)
  - [Fast comparison of feature selections with the same classifiers](#Fast-comparison-of-feature-selections-with-the-same-classifiers)
    - [The most important features with no information about locations](#The-most-important-features-with-no-information-about-locations)
    - [The most correlated with target value](#The-most-correlated-with-target-value)
  - [Training on the 85 most important features](#Training-on-the-85-most-important-features)
  - [Let's consider time dependecies problem](#Let's-consider-time-dependecies-problem)
    - [Benchmark of ExtraTreesClassifiers with season features and without](#Benchmark-of-ExtraTreesClassifiers-with-season-features-and-without)
  - [ExtraTreesClassifier and GradientBoosting with no season features trained on all data](#ExtraTreesClassifier-and-GradientBoosting-with-no-season-features-trained-on-all-data)
    - [ExtraTreesClassifier](#ExtraTreesClassifier)
    - [GradientBoosting](#GradientBoosting)
  - [Combining multiple answers into a final one](#Combining-multiple-answers-into-a-final-one)
  - [Summary - lessons learned](#Summary-lessons-learned)

## Loading the data


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 150)
```


```python
train_data = pd.read_csv("FitFood_competition_data_training.csv", sep=";")
```

## Preliminary analysis and preprocessing


```python
df = train_data
attrs = set(df.keys())
df.describe(include='all')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>will_it_sell</th>
      <th>pos_id</th>
      <th>product_id_unified</th>
      <th>company_id</th>
      <th>category_id</th>
      <th>category_name</th>
      <th>product_name</th>
      <th>partner_product</th>
      <th>address_city</th>
      <th>diet</th>
      <th>size</th>
      <th>cooking_time</th>
      <th>cooking_mv</th>
      <th>cooking_ov</th>
      <th>storage_temp</th>
      <th>vat</th>
      <th>bialko_100</th>
      <th>weglow_100</th>
      <th>cukry_calk</th>
      <th>tluszcz_nasyc_calk</th>
      <th>energia_calk</th>
      <th>energia_100</th>
      <th>tluszcz_nasyc_100</th>
      <th>blonnik_100</th>
      <th>tluszcz_calk</th>
      <th>bialko_calk</th>
      <th>sol_100</th>
      <th>weglow_calk</th>
      <th>blonnik_calk</th>
      <th>tluszcz_100</th>
      <th>cukry_100</th>
      <th>sol_calk</th>
      <th>weekday</th>
      <th>quarter</th>
      <th>month</th>
      <th>week</th>
      <th>qty_lag1</th>
      <th>qty_lag2</th>
      <th>qty_lag3</th>
      <th>qty_lag4</th>
      <th>qty_lag5</th>
      <th>qty_lag6</th>
      <th>qty_lag7</th>
      <th>qty_lag8</th>
      <th>qty_lag9</th>
      <th>qty_lag10</th>
      <th>qty_lag11</th>
      <th>qty_lag12</th>
      <th>qty_lag13</th>
      <th>qty_lag14</th>
      <th>meanLastPeriod_lag1</th>
      <th>meanLastPeriod_lag2</th>
      <th>meanLastPeriod_lag3</th>
      <th>meanLastPeriod_lag4</th>
      <th>meanLastPeriod_lag5</th>
      <th>meanLastPeriod_lag6</th>
      <th>meanLastPeriod_lag7</th>
      <th>meanLastPeriod_lag1_lag7_diff</th>
      <th>sdLastPeriod_lag1</th>
      <th>sdLastPeriod_lag2</th>
      <th>sdLastPeriod_lag3</th>
      <th>sdLastPeriod_lag4</th>
      <th>sdLastPeriod_lag5</th>
      <th>sdLastPeriod_lag6</th>
      <th>sdLastPeriod_lag7</th>
      <th>minLastPeriod_lag1</th>
      <th>minLastPeriod_lag7</th>
      <th>minLastPeriod_lag1_lag7_diff</th>
      <th>maxLastPeriod_lag1</th>
      <th>maxLastPeriod_lag7</th>
      <th>maxLastPeriod_lag1_lag7_diff</th>
      <th>diff1_lag1</th>
      <th>diff1_lag7</th>
      <th>diff1_lag1_lag7_diff</th>
      <th>diffLagPeriod_lag1</th>
      <th>diffLagPeriod_lag7</th>
      <th>diffLagPeriod_lag1_lag7_diff</th>
      <th>mean_diff1_lag1</th>
      <th>mean_diff1_lag7</th>
      <th>mean_diff1_lag1_lag7_diff</th>
      <th>sum_qty</th>
      <th>avg_discount_mean_value_lag1</th>
      <th>avg_discount_count_lag1</th>
      <th>avg_from_blik_lag1</th>
      <th>avg_from_paypass_lag1</th>
      <th>avg_from_payu_lag1</th>
      <th>avg_total_lag1</th>
      <th>avg_total_to_discount_lag1</th>
      <th>avg_total_base_lag1</th>
      <th>avg_sum_fv_lag1</th>
      <th>avg_transaction_discount_count_lag1</th>
      <th>roc1_lag1</th>
      <th>rocPeriod_lag1</th>
      <th>days_since_prev_delivery</th>
      <th>sales_since_prev_delivery</th>
      <th>available_products</th>
      <th>is_delivery_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.360496e+06</td>
      <td>5360496</td>
      <td>5.360496e+06</td>
      <td>5360496</td>
      <td>5360496</td>
      <td>5360496</td>
      <td>5360496</td>
      <td>5.360496e+06</td>
      <td>5360496</td>
      <td>2840752</td>
      <td>5360496</td>
      <td>3632502</td>
      <td>2336346.0</td>
      <td>2336346.0</td>
      <td>3930214</td>
      <td>5.360496e+06</td>
      <td>4.999443e+06</td>
      <td>4.999443e+06</td>
      <td>4.998511e+06</td>
      <td>4.998511e+06</td>
      <td>4.999443e+06</td>
      <td>4.999443e+06</td>
      <td>4.998511e+06</td>
      <td>4.070153e+06</td>
      <td>4.999443e+06</td>
      <td>4.999443e+06</td>
      <td>4.998511e+06</td>
      <td>4.999443e+06</td>
      <td>4.070153e+06</td>
      <td>4.999443e+06</td>
      <td>4.998511e+06</td>
      <td>4.998511e+06</td>
      <td>5360496</td>
      <td>5360496</td>
      <td>5.360496e+06</td>
      <td>5.360496e+06</td>
      <td>5.331152e+06</td>
      <td>5.301896e+06</td>
      <td>5.272670e+06</td>
      <td>5.243488e+06</td>
      <td>5.214306e+06</td>
      <td>5.185124e+06</td>
      <td>5.155944e+06</td>
      <td>5.126791e+06</td>
      <td>5.097737e+06</td>
      <td>5.068781e+06</td>
      <td>5.039988e+06</td>
      <td>5.011318e+06</td>
      <td>4.982648e+06</td>
      <td>4.953978e+06</td>
      <td>5.155944e+06</td>
      <td>5.126791e+06</td>
      <td>5.097737e+06</td>
      <td>5.068781e+06</td>
      <td>5.039988e+06</td>
      <td>5.011318e+06</td>
      <td>4.982648e+06</td>
      <td>4.982648e+06</td>
      <td>5.155944e+06</td>
      <td>5.126791e+06</td>
      <td>5.097737e+06</td>
      <td>5.068781e+06</td>
      <td>5.039988e+06</td>
      <td>5.011318e+06</td>
      <td>4.982648e+06</td>
      <td>5.155944e+06</td>
      <td>4.982648e+06</td>
      <td>4.982648e+06</td>
      <td>5.155944e+06</td>
      <td>4.982648e+06</td>
      <td>4.982648e+06</td>
      <td>5.301896e+06</td>
      <td>5.126791e+06</td>
      <td>5.126791e+06</td>
      <td>5.126791e+06</td>
      <td>4.953978e+06</td>
      <td>4.953978e+06</td>
      <td>5.126791e+06</td>
      <td>4.953978e+06</td>
      <td>4.953978e+06</td>
      <td>5.185124e+06</td>
      <td>5.331152e+06</td>
      <td>5.331152e+06</td>
      <td>5.331152e+06</td>
      <td>5.331152e+06</td>
      <td>5.331152e+06</td>
      <td>5.331152e+06</td>
      <td>5.331152e+06</td>
      <td>5.331152e+06</td>
      <td>5331152.0</td>
      <td>5.331152e+06</td>
      <td>5.331152e+06</td>
      <td>5.331152e+06</td>
      <td>4.779475e+06</td>
      <td>4.779475e+06</td>
      <td>4.805004e+06</td>
      <td>5.360496e+06</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>375</td>
      <td>NaN</td>
      <td>298</td>
      <td>14</td>
      <td>14</td>
      <td>137</td>
      <td>NaN</td>
      <td>33</td>
      <td>10</td>
      <td>22</td>
      <td>1</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7</td>
      <td>4</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>59fc5325c94b722506678bd1</td>
      <td>NaN</td>
      <td>5a587706cf5c8134b3a9891d</td>
      <td>5a6f110ca0899f5ca2f7d6e9</td>
      <td>Dania Lunch Duże</td>
      <td>Dyniowe curry z indykiem</td>
      <td>NaN</td>
      <td>Warszawa</td>
      <td>Dieta Samuraja</td>
      <td>350g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 °C</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>czwartek</td>
      <td>Q3</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>25498</td>
      <td>NaN</td>
      <td>101533</td>
      <td>1144201</td>
      <td>1144201</td>
      <td>112303</td>
      <td>NaN</td>
      <td>2353704</td>
      <td>994953</td>
      <td>1195365</td>
      <td>3632502</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3477491</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>776919</td>
      <td>2034382</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.195335e-01</td>
      <td>NaN</td>
      <td>1.104991e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>9.166092e-02</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>6.627107e+00</td>
      <td>7.470809e+00</td>
      <td>1.920212e+01</td>
      <td>8.337381e+00</td>
      <td>3.861783e+00</td>
      <td>2.961271e+02</td>
      <td>1.531352e+02</td>
      <td>2.124834e+00</td>
      <td>2.689635e+00</td>
      <td>8.034908e+00</td>
      <td>1.689004e+01</td>
      <td>5.981112e-01</td>
      <td>1.746482e+02</td>
      <td>7.774678e+00</td>
      <td>4.951239e+00</td>
      <td>8.053772e+00</td>
      <td>1.905875e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.216968e+00</td>
      <td>2.960717e+01</td>
      <td>9.486317e-02</td>
      <td>9.478666e-02</td>
      <td>9.474460e-02</td>
      <td>9.506039e-02</td>
      <td>9.545451e-02</td>
      <td>9.583937e-02</td>
      <td>9.582765e-02</td>
      <td>9.566920e-02</td>
      <td>9.549787e-02</td>
      <td>9.540085e-02</td>
      <td>9.530975e-02</td>
      <td>9.563293e-02</td>
      <td>9.593152e-02</td>
      <td>9.612921e-02</td>
      <td>9.324190e-02</td>
      <td>9.334906e-02</td>
      <td>9.344387e-02</td>
      <td>9.353105e-02</td>
      <td>9.356136e-02</td>
      <td>9.358200e-02</td>
      <td>9.358882e-02</td>
      <td>-2.614129e-03</td>
      <td>1.639429e-01</td>
      <td>1.640696e-01</td>
      <td>1.641682e-01</td>
      <td>1.642481e-01</td>
      <td>1.642443e-01</td>
      <td>1.642216e-01</td>
      <td>1.641720e-01</td>
      <td>6.128849e-05</td>
      <td>6.221591e-05</td>
      <td>-4.013930e-07</td>
      <td>4.068483e-01</td>
      <td>4.073537e-01</td>
      <td>-1.017913e-02</td>
      <td>-1.289350e-03</td>
      <td>-1.241322e-03</td>
      <td>1.025788e-03</td>
      <td>-3.887032e-03</td>
      <td>-3.434210e-03</td>
      <td>-1.647161e-04</td>
      <td>-5.552903e-04</td>
      <td>-4.906014e-04</td>
      <td>-2.353087e-05</td>
      <td>6.524010e-01</td>
      <td>1.083268e-03</td>
      <td>1.690120e-03</td>
      <td>4.194721e-02</td>
      <td>1.674438e-01</td>
      <td>2.564613e-02</td>
      <td>2.541403e+00</td>
      <td>7.954283e-01</td>
      <td>3.110736e+00</td>
      <td>0.0</td>
      <td>9.892658e-02</td>
      <td>-1.643735e-03</td>
      <td>-8.121109e-03</td>
      <td>1.686129e+01</td>
      <td>3.515495e-01</td>
      <td>1.218998e+00</td>
      <td>1.169069e-01</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.244153e-01</td>
      <td>NaN</td>
      <td>5.583013e+01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.885467e-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>4.418595e+00</td>
      <td>6.744022e+00</td>
      <td>1.500976e+01</td>
      <td>7.253456e+00</td>
      <td>3.267804e+00</td>
      <td>1.590917e+02</td>
      <td>1.187398e+02</td>
      <td>2.980594e+00</td>
      <td>1.875738e+00</td>
      <td>5.511651e+00</td>
      <td>1.304603e+01</td>
      <td>3.823469e-01</td>
      <td>2.287943e+02</td>
      <td>5.423359e+00</td>
      <td>5.725387e+00</td>
      <td>1.246568e+01</td>
      <td>1.499202e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.921179e+00</td>
      <td>1.269062e+01</td>
      <td>4.513092e-01</td>
      <td>4.513945e-01</td>
      <td>4.515465e-01</td>
      <td>4.524057e-01</td>
      <td>4.533894e-01</td>
      <td>4.543876e-01</td>
      <td>4.546307e-01</td>
      <td>4.545370e-01</td>
      <td>4.543946e-01</td>
      <td>4.542686e-01</td>
      <td>4.540837e-01</td>
      <td>4.549490e-01</td>
      <td>4.557977e-01</td>
      <td>4.564781e-01</td>
      <td>2.466131e-01</td>
      <td>2.469229e-01</td>
      <td>2.472161e-01</td>
      <td>2.474940e-01</td>
      <td>2.476744e-01</td>
      <td>2.478502e-01</td>
      <td>2.480214e-01</td>
      <td>1.995848e-01</td>
      <td>3.691396e-01</td>
      <td>3.694786e-01</td>
      <td>3.697978e-01</td>
      <td>3.700779e-01</td>
      <td>3.702582e-01</td>
      <td>3.704252e-01</td>
      <td>3.705731e-01</td>
      <td>7.902433e-03</td>
      <td>7.963431e-03</td>
      <td>1.113690e-02</td>
      <td>9.371576e-01</td>
      <td>9.408787e-01</td>
      <td>8.323831e-01</td>
      <td>5.500891e-01</td>
      <td>5.536175e-01</td>
      <td>7.811386e-01</td>
      <td>5.350323e-01</td>
      <td>5.370825e-01</td>
      <td>7.523585e-01</td>
      <td>7.643318e-02</td>
      <td>7.672608e-02</td>
      <td>1.074798e-01</td>
      <td>1.724346e+00</td>
      <td>2.766163e-02</td>
      <td>3.947974e-02</td>
      <td>1.704341e-01</td>
      <td>3.541303e-01</td>
      <td>1.368020e-01</td>
      <td>5.225088e+00</td>
      <td>3.098942e+00</td>
      <td>6.169283e+00</td>
      <td>0.0</td>
      <td>2.925222e-01</td>
      <td>2.389877e-01</td>
      <td>6.231182e-01</td>
      <td>3.068664e+01</td>
      <td>1.024493e+00</td>
      <td>2.677701e+01</td>
      <td>3.213094e-01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>1.004000e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>5.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.700000e+00</td>
      <td>2.000000e-01</td>
      <td>0.000000e+00</td>
      <td>2.500000e+01</td>
      <td>2.030000e+01</td>
      <td>0.000000e+00</td>
      <td>7.000000e-01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.700000e+00</td>
      <td>7.000000e-01</td>
      <td>0.000000e+00</td>
      <td>1.000000e-01</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-1.057143e+01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-2.900000e+01</td>
      <td>-3.000000e+01</td>
      <td>-3.000000e+01</td>
      <td>-3.100000e+01</td>
      <td>-3.000000e+01</td>
      <td>-3.000000e+01</td>
      <td>-3.200000e+01</td>
      <td>-4.285714e+00</td>
      <td>-4.285714e+00</td>
      <td>-4.571429e+00</td>
      <td>0.000000e+00</td>
      <td>-2.775558e-16</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-1.172396e-13</td>
      <td>-1.776357e-13</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>-3.044522e+00</td>
      <td>-3.465736e+00</td>
      <td>1.000000e+00</td>
      <td>0.000000e+00</td>
      <td>-2.400000e+01</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>1.057000e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>5.000000e+00</td>
      <td>3.200000e+00</td>
      <td>9.200000e+00</td>
      <td>3.000000e+00</td>
      <td>1.500000e+00</td>
      <td>1.620000e+02</td>
      <td>7.600000e+01</td>
      <td>7.000000e-01</td>
      <td>1.300000e+00</td>
      <td>4.400000e+00</td>
      <td>4.080000e+00</td>
      <td>2.000000e-01</td>
      <td>2.250000e+01</td>
      <td>4.000000e+00</td>
      <td>1.600000e+00</td>
      <td>9.000000e-01</td>
      <td>5.000000e-01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.000000e+00</td>
      <td>2.100000e+01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>1.117000e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>5.000000e+00</td>
      <td>6.800000e+00</td>
      <td>1.300000e+01</td>
      <td>5.900000e+00</td>
      <td>3.000000e+00</td>
      <td>2.668000e+02</td>
      <td>1.080000e+02</td>
      <td>1.100000e+00</td>
      <td>2.100000e+00</td>
      <td>6.700000e+00</td>
      <td>1.400000e+01</td>
      <td>7.000000e-01</td>
      <td>4.380000e+01</td>
      <td>6.900000e+00</td>
      <td>2.500000e+00</td>
      <td>2.000000e+00</td>
      <td>1.900000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8.000000e+00</td>
      <td>3.200000e+01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>5.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>1.152000e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>5.000000e+00</td>
      <td>8.900000e+00</td>
      <td>2.340000e+01</td>
      <td>1.260000e+01</td>
      <td>5.100000e+00</td>
      <td>4.235000e+02</td>
      <td>1.620000e+02</td>
      <td>2.200000e+00</td>
      <td>3.500000e+00</td>
      <td>1.050000e+01</td>
      <td>2.990000e+01</td>
      <td>8.000000e-01</td>
      <td>2.400000e+02</td>
      <td>9.700000e+00</td>
      <td>4.900000e+00</td>
      <td>7.300000e+00</td>
      <td>3.300000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000e+01</td>
      <td>4.000000e+01</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.0</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>9.000000e+00</td>
      <td>0.000000e+00</td>
      <td>2.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.000000e+00</td>
      <td>NaN</td>
      <td>1.193000e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>2.300000e+01</td>
      <td>3.800000e+01</td>
      <td>6.130000e+01</td>
      <td>2.950000e+01</td>
      <td>1.430000e+01</td>
      <td>6.370000e+02</td>
      <td>4.879000e+02</td>
      <td>3.340000e+01</td>
      <td>1.700000e+01</td>
      <td>3.540000e+01</td>
      <td>4.830000e+01</td>
      <td>2.120000e+00</td>
      <td>9.830000e+02</td>
      <td>2.680000e+01</td>
      <td>4.000000e+01</td>
      <td>4.210000e+01</td>
      <td>5.000000e+00</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.200000e+01</td>
      <td>5.200000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>1.214286e+01</td>
      <td>1.214286e+01</td>
      <td>1.214286e+01</td>
      <td>1.214286e+01</td>
      <td>1.214286e+01</td>
      <td>1.214286e+01</td>
      <td>1.214286e+01</td>
      <td>7.571429e+00</td>
      <td>1.122285e+01</td>
      <td>1.122285e+01</td>
      <td>1.122285e+01</td>
      <td>1.122285e+01</td>
      <td>1.122285e+01</td>
      <td>1.122285e+01</td>
      <td>1.122285e+01</td>
      <td>2.000000e+00</td>
      <td>2.000000e+00</td>
      <td>2.000000e+00</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>2.800000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>4.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.000000e+01</td>
      <td>3.100000e+01</td>
      <td>4.285714e+00</td>
      <td>4.285714e+00</td>
      <td>4.428571e+00</td>
      <td>8.500000e+01</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>1.000000e+00</td>
      <td>9.994000e+01</td>
      <td>7.194000e+01</td>
      <td>1.878400e+02</td>
      <td>0.0</td>
      <td>3.000000e+00</td>
      <td>3.044522e+00</td>
      <td>3.465736e+00</td>
      <td>2.580000e+02</td>
      <td>3.900000e+01</td>
      <td>2.202300e+04</td>
      <td>1.000000e+00</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sample(n=50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>will_it_sell</th>
      <th>pos_id</th>
      <th>product_id_unified</th>
      <th>company_id</th>
      <th>category_id</th>
      <th>category_name</th>
      <th>product_name</th>
      <th>partner_product</th>
      <th>address_city</th>
      <th>diet</th>
      <th>size</th>
      <th>cooking_time</th>
      <th>cooking_mv</th>
      <th>cooking_ov</th>
      <th>storage_temp</th>
      <th>vat</th>
      <th>bialko_100</th>
      <th>weglow_100</th>
      <th>cukry_calk</th>
      <th>tluszcz_nasyc_calk</th>
      <th>energia_calk</th>
      <th>energia_100</th>
      <th>tluszcz_nasyc_100</th>
      <th>blonnik_100</th>
      <th>tluszcz_calk</th>
      <th>bialko_calk</th>
      <th>sol_100</th>
      <th>weglow_calk</th>
      <th>blonnik_calk</th>
      <th>tluszcz_100</th>
      <th>cukry_100</th>
      <th>sol_calk</th>
      <th>weekday</th>
      <th>quarter</th>
      <th>month</th>
      <th>week</th>
      <th>qty_lag1</th>
      <th>qty_lag2</th>
      <th>qty_lag3</th>
      <th>qty_lag4</th>
      <th>qty_lag5</th>
      <th>qty_lag6</th>
      <th>qty_lag7</th>
      <th>qty_lag8</th>
      <th>qty_lag9</th>
      <th>qty_lag10</th>
      <th>qty_lag11</th>
      <th>qty_lag12</th>
      <th>qty_lag13</th>
      <th>qty_lag14</th>
      <th>meanLastPeriod_lag1</th>
      <th>meanLastPeriod_lag2</th>
      <th>meanLastPeriod_lag3</th>
      <th>meanLastPeriod_lag4</th>
      <th>meanLastPeriod_lag5</th>
      <th>meanLastPeriod_lag6</th>
      <th>meanLastPeriod_lag7</th>
      <th>meanLastPeriod_lag1_lag7_diff</th>
      <th>sdLastPeriod_lag1</th>
      <th>sdLastPeriod_lag2</th>
      <th>sdLastPeriod_lag3</th>
      <th>sdLastPeriod_lag4</th>
      <th>sdLastPeriod_lag5</th>
      <th>sdLastPeriod_lag6</th>
      <th>sdLastPeriod_lag7</th>
      <th>minLastPeriod_lag1</th>
      <th>minLastPeriod_lag7</th>
      <th>minLastPeriod_lag1_lag7_diff</th>
      <th>maxLastPeriod_lag1</th>
      <th>maxLastPeriod_lag7</th>
      <th>maxLastPeriod_lag1_lag7_diff</th>
      <th>diff1_lag1</th>
      <th>diff1_lag7</th>
      <th>diff1_lag1_lag7_diff</th>
      <th>diffLagPeriod_lag1</th>
      <th>diffLagPeriod_lag7</th>
      <th>diffLagPeriod_lag1_lag7_diff</th>
      <th>mean_diff1_lag1</th>
      <th>mean_diff1_lag7</th>
      <th>mean_diff1_lag1_lag7_diff</th>
      <th>sum_qty</th>
      <th>avg_discount_mean_value_lag1</th>
      <th>avg_discount_count_lag1</th>
      <th>avg_from_blik_lag1</th>
      <th>avg_from_paypass_lag1</th>
      <th>avg_from_payu_lag1</th>
      <th>avg_total_lag1</th>
      <th>avg_total_to_discount_lag1</th>
      <th>avg_total_base_lag1</th>
      <th>avg_sum_fv_lag1</th>
      <th>avg_transaction_discount_count_lag1</th>
      <th>roc1_lag1</th>
      <th>rocPeriod_lag1</th>
      <th>days_since_prev_delivery</th>
      <th>sales_since_prev_delivery</th>
      <th>available_products</th>
      <th>is_delivery_day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4885010</th>
      <td>0</td>
      <td>5b4893ddee2a423f39dacbf6</td>
      <td>1156</td>
      <td>5b39bb530663ab48e336173e</td>
      <td>591301c83dd75608a9c2ef1b</td>
      <td>Napoje</td>
      <td>Smoothie BeRAW Breakfast energy</td>
      <td>0</td>
      <td>Skawina</td>
      <td>NaN</td>
      <td>250ml</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sobota</td>
      <td>Q4</td>
      <td>10</td>
      <td>43</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2160445</th>
      <td>0</td>
      <td>5cab37f2a77669455cbbaa78</td>
      <td>1067</td>
      <td>5caaeaa8822b5e2d312ac807</td>
      <td>5cd1a4d32b10792bc08dab31</td>
      <td>Pan Pomidor - Pierogi</td>
      <td>Pierogi z dynią, jarmużem, quinoą i kolendrą</td>
      <td>0</td>
      <td>Wrocław</td>
      <td>NaN</td>
      <td>240g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1-6 °C</td>
      <td>5</td>
      <td>3.90</td>
      <td>23.0</td>
      <td>1.1</td>
      <td>0.40</td>
      <td>155.0</td>
      <td>155.0</td>
      <td>0.4</td>
      <td>2.3</td>
      <td>4.80</td>
      <td>3.9</td>
      <td>1.10</td>
      <td>23.0</td>
      <td>2.3</td>
      <td>4.8</td>
      <td>1.1</td>
      <td>1.100</td>
      <td>poniedziałek</td>
      <td>Q3</td>
      <td>9</td>
      <td>37</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>514033</th>
      <td>0</td>
      <td>5a37f93f753e9f3591458dd7</td>
      <td>1189</td>
      <td>5af3e4736e089c600a14dd86</td>
      <td>590053bdc5c79d3575eb44f6</td>
      <td>Zupy</td>
      <td>Barszcz z mlekiem kokosowym</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>Zupy</td>
      <td>300g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>1.20</td>
      <td>6.6</td>
      <td>12.6</td>
      <td>4.10</td>
      <td>130.3</td>
      <td>43.4</td>
      <td>1.4</td>
      <td>1.9</td>
      <td>4.70</td>
      <td>3.5</td>
      <td>0.40</td>
      <td>199.0</td>
      <td>5.6</td>
      <td>1.6</td>
      <td>4.2</td>
      <td>1.300</td>
      <td>wtorek</td>
      <td>Q3</td>
      <td>9</td>
      <td>39</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>508094</th>
      <td>0</td>
      <td>5b2b8fa60663ab48e334dd2b</td>
      <td>1063</td>
      <td>5b30ea120663ab48e3354bfc</td>
      <td>59005cd6c5c79d3575eb450d</td>
      <td>Przekąski</td>
      <td>BeRAW Baton protein 38% - surowe kakao w gorzk...</td>
      <td>0</td>
      <td>Kraków</td>
      <td>NaN</td>
      <td>60g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>38.00</td>
      <td>36.0</td>
      <td>13.2</td>
      <td>3.24</td>
      <td>229.8</td>
      <td>383.0</td>
      <td>5.4</td>
      <td>NaN</td>
      <td>5.16</td>
      <td>22.8</td>
      <td>0.06</td>
      <td>21.6</td>
      <td>NaN</td>
      <td>8.6</td>
      <td>22.0</td>
      <td>0.036</td>
      <td>środa</td>
      <td>Q2</td>
      <td>6</td>
      <td>26</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.285714</td>
      <td>0.755929</td>
      <td>0.755929</td>
      <td>0.755929</td>
      <td>0.755929</td>
      <td>0.755929</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>-8.881784e-16</td>
      <td>0.000000</td>
      <td>6.990000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.945910</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2429560</th>
      <td>1</td>
      <td>5c8bae66a7d3a504da7a4b6b</td>
      <td>1137</td>
      <td>5c7fb0d36b25e24bf5028f65</td>
      <td>5d1b55aa5379175d45e9360a</td>
      <td>Mr Thai</td>
      <td>Sesame Beef</td>
      <td>0</td>
      <td>Katowice</td>
      <td>NaN</td>
      <td>380g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>8.90</td>
      <td>26.1</td>
      <td>3.9</td>
      <td>0.70</td>
      <td>162.0</td>
      <td>162.0</td>
      <td>0.7</td>
      <td>0.7</td>
      <td>2.10</td>
      <td>8.9</td>
      <td>0.10</td>
      <td>26.1</td>
      <td>0.7</td>
      <td>2.1</td>
      <td>3.9</td>
      <td>0.100</td>
      <td>piątek</td>
      <td>Q3</td>
      <td>8</td>
      <td>32</td>
      <td>4.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.142857</td>
      <td>0.857143</td>
      <td>0.857143</td>
      <td>0.857143</td>
      <td>0.857143</td>
      <td>0.857143</td>
      <td>0.857143</td>
      <td>0.285714</td>
      <td>1.676163</td>
      <td>1.214986</td>
      <td>1.214986</td>
      <td>1.214986</td>
      <td>1.214986</td>
      <td>1.214986</td>
      <td>1.214986</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>-1.0</td>
      <td>0.285714</td>
      <td>0.428571</td>
      <td>-0.142857</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.375</td>
      <td>0.625000</td>
      <td>0.000000</td>
      <td>1.599000e+01</td>
      <td>0.000000</td>
      <td>15.990000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.287682</td>
      <td>0.980829</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5141712</th>
      <td>0</td>
      <td>5c79285d7b6c863a5d522c20</td>
      <td>1146</td>
      <td>5c7cd37e3ae6e53aff1548b4</td>
      <td>59005cd6c5c79d3575eb450d</td>
      <td>Przekąski</td>
      <td>Superfood SPORT - banan, białko</td>
      <td>1</td>
      <td>Warszawa</td>
      <td>NaN</td>
      <td>35g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8</td>
      <td>11.00</td>
      <td>61.3</td>
      <td>14.7</td>
      <td>0.50</td>
      <td>135.8</td>
      <td>388.0</td>
      <td>1.4</td>
      <td>7.8</td>
      <td>5.20</td>
      <td>3.9</td>
      <td>0.01</td>
      <td>215.0</td>
      <td>2.7</td>
      <td>14.9</td>
      <td>42.1</td>
      <td>0.000</td>
      <td>środa</td>
      <td>Q2</td>
      <td>4</td>
      <td>17</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2334542</th>
      <td>1</td>
      <td>5caf375e2c9b633ae260d078</td>
      <td>1181</td>
      <td>5c924f69b6cb840dcac430fe</td>
      <td>5cd1a4f40a544c2d0d156fea</td>
      <td>Pan Pomidor - Zupy</td>
      <td>Marokańska z quinoą, batatem i kolendrą</td>
      <td>0</td>
      <td>Katowice</td>
      <td>NaN</td>
      <td>400g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-6 °C</td>
      <td>8</td>
      <td>1.60</td>
      <td>5.1</td>
      <td>2.6</td>
      <td>0.10</td>
      <td>45.0</td>
      <td>45.0</td>
      <td>0.1</td>
      <td>NaN</td>
      <td>1.50</td>
      <td>1.6</td>
      <td>0.78</td>
      <td>5.1</td>
      <td>NaN</td>
      <td>1.5</td>
      <td>2.6</td>
      <td>0.780</td>
      <td>poniedziałek</td>
      <td>Q3</td>
      <td>7</td>
      <td>28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.428571</td>
      <td>0.428571</td>
      <td>0.428571</td>
      <td>0.428571</td>
      <td>0.428571</td>
      <td>0.428571</td>
      <td>0.000000</td>
      <td>0.428571</td>
      <td>1.133893</td>
      <td>1.133893</td>
      <td>1.133893</td>
      <td>1.133893</td>
      <td>1.133893</td>
      <td>1.133893</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.698333e+01</td>
      <td>3.666667</td>
      <td>20.643333</td>
      <td>0.0</td>
      <td>0.666667</td>
      <td>0.000000</td>
      <td>1.609438</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>867374</th>
      <td>0</td>
      <td>5bc9b10ba6c5c61a73d3fefa</td>
      <td>1078</td>
      <td>5a572369cf5c8134b3a98692</td>
      <td>5cb9b8eedf68013fb09db8f0</td>
      <td>Makarony</td>
      <td>Wegański z masłem orzechowym</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>NaN</td>
      <td>250g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>8.00</td>
      <td>18.6</td>
      <td>1.5</td>
      <td>1.50</td>
      <td>415.0</td>
      <td>166.0</td>
      <td>0.6</td>
      <td>6.7</td>
      <td>12.90</td>
      <td>20.0</td>
      <td>1.00</td>
      <td>465.0</td>
      <td>16.8</td>
      <td>5.1</td>
      <td>0.6</td>
      <td>2.500</td>
      <td>niedziela</td>
      <td>Q4</td>
      <td>10</td>
      <td>43</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>110.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4047454</th>
      <td>0</td>
      <td>5cd56095aa566714bfc60c5c</td>
      <td>1098</td>
      <td>5c98d473bedc6e4786a40398</td>
      <td>5a6f110ca0899f5ca2f7d6e9</td>
      <td>Dania Lunch Duże</td>
      <td>Pieczone placki ziemniaczane z gulaszem drobiowym</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>Kuchnia Słowiańska</td>
      <td>400g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>7.20</td>
      <td>12.1</td>
      <td>3.2</td>
      <td>2.20</td>
      <td>444.0</td>
      <td>111.0</td>
      <td>0.6</td>
      <td>3.3</td>
      <td>12.00</td>
      <td>28.9</td>
      <td>0.90</td>
      <td>48.4</td>
      <td>13.4</td>
      <td>3.0</td>
      <td>0.8</td>
      <td>3.600</td>
      <td>czwartek</td>
      <td>Q3</td>
      <td>9</td>
      <td>39</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.428571</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.786796</td>
      <td>0.786796</td>
      <td>0.786796</td>
      <td>0.755929</td>
      <td>0.755929</td>
      <td>0.755929</td>
      <td>0.755929</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>2.0</td>
      <td>-3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>-0.142857</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.599000e+01</td>
      <td>0.000000</td>
      <td>15.990000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.386294</td>
      <td>3.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1946567</th>
      <td>0</td>
      <td>5b1b963bcaef965005d0e6b0</td>
      <td>1061</td>
      <td>5b1f8a5ecaef965005d12d82</td>
      <td>5abe0aed049e180557e22330</td>
      <td>Sałatki</td>
      <td>FitSalad - Sałatka z fetą, oliwkami i pomidork...</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>--</td>
      <td>350g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>4.40</td>
      <td>15.8</td>
      <td>1.8</td>
      <td>7.00</td>
      <td>504.0</td>
      <td>144.0</td>
      <td>2.0</td>
      <td>2.1</td>
      <td>22.70</td>
      <td>15.4</td>
      <td>0.80</td>
      <td>55.3</td>
      <td>7.5</td>
      <td>6.5</td>
      <td>0.5</td>
      <td>2.700</td>
      <td>sobota</td>
      <td>Q2</td>
      <td>5</td>
      <td>21</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2998481</th>
      <td>0</td>
      <td>5cd563af73d91d192aba5de5</td>
      <td>1149</td>
      <td>5cc01fa1def57b4350f57e70</td>
      <td>59005cd6c5c79d3575eb450d</td>
      <td>Przekąski</td>
      <td>Superfood ENERGIA - kokos, lukuma</td>
      <td>1</td>
      <td>Warszawa</td>
      <td>NaN</td>
      <td>35g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8</td>
      <td>5.60</td>
      <td>55.0</td>
      <td>12.7</td>
      <td>3.40</td>
      <td>135.1</td>
      <td>386.0</td>
      <td>9.8</td>
      <td>7.2</td>
      <td>6.20</td>
      <td>2.0</td>
      <td>0.01</td>
      <td>19.3</td>
      <td>2.5</td>
      <td>17.7</td>
      <td>36.2</td>
      <td>0.000</td>
      <td>niedziela</td>
      <td>Q2</td>
      <td>6</td>
      <td>26</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>5.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4911007</th>
      <td>0</td>
      <td>5c641afa4dc1b258d9423850</td>
      <td>1062</td>
      <td>5c6687ab6961290fe741922e</td>
      <td>5a0033206cdc0d08a6591bfb</td>
      <td>Dania Lunch Małe</td>
      <td>Pęczotto ze szpinakiem</td>
      <td>0</td>
      <td>Kraków</td>
      <td>Kuchnia Słowiańska</td>
      <td>350g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>7.90</td>
      <td>11.4</td>
      <td>4.6</td>
      <td>2.50</td>
      <td>336.0</td>
      <td>96.0</td>
      <td>0.7</td>
      <td>3.2</td>
      <td>4.60</td>
      <td>27.7</td>
      <td>0.70</td>
      <td>39.9</td>
      <td>11.2</td>
      <td>1.3</td>
      <td>1.3</td>
      <td>2.500</td>
      <td>wtorek</td>
      <td>Q3</td>
      <td>9</td>
      <td>38</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>71.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1984312</th>
      <td>0</td>
      <td>5afc6a7035348125c0a0ce58</td>
      <td>1061</td>
      <td>5a572369cf5c8134b3a98692</td>
      <td>5abe0aed049e180557e22330</td>
      <td>Sałatki</td>
      <td>FitSalad - Sałatka z fetą, oliwkami i pomidork...</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>--</td>
      <td>350g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>4.40</td>
      <td>15.8</td>
      <td>1.8</td>
      <td>7.00</td>
      <td>504.0</td>
      <td>144.0</td>
      <td>2.0</td>
      <td>2.1</td>
      <td>22.70</td>
      <td>15.4</td>
      <td>0.80</td>
      <td>55.3</td>
      <td>7.5</td>
      <td>6.5</td>
      <td>0.5</td>
      <td>2.700</td>
      <td>piątek</td>
      <td>Q3</td>
      <td>7</td>
      <td>27</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>51.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4477031</th>
      <td>0</td>
      <td>59561e972e5db6320199ec77</td>
      <td>1026</td>
      <td>5bbdd41f15c00937a08115b9</td>
      <td>59005cd6c5c79d3575eb450d</td>
      <td>Przekąski</td>
      <td>BeRAW Baton healthy snack - masło orzechowe</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>NaN</td>
      <td>40g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8</td>
      <td>7.00</td>
      <td>38.0</td>
      <td>13.2</td>
      <td>1.20</td>
      <td>150.0</td>
      <td>375.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>6.04</td>
      <td>2.8</td>
      <td>0.14</td>
      <td>15.2</td>
      <td>NaN</td>
      <td>15.1</td>
      <td>33.0</td>
      <td>0.056</td>
      <td>niedziela</td>
      <td>Q1</td>
      <td>3</td>
      <td>13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4323764</th>
      <td>0</td>
      <td>5cd2c7b75314a576b410fb6f</td>
      <td>1168</td>
      <td>5cc01f99def57b4350f57e6f</td>
      <td>5a0033206cdc0d08a6591bfb</td>
      <td>Dania Lunch Małe</td>
      <td>Indyk pieczony z warzywami korzeniowymi</td>
      <td>0</td>
      <td>Wrocław</td>
      <td>Dieta Paleo</td>
      <td>350g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>6.90</td>
      <td>11.1</td>
      <td>7.4</td>
      <td>3.70</td>
      <td>300.7</td>
      <td>85.9</td>
      <td>1.1</td>
      <td>2.2</td>
      <td>6.20</td>
      <td>24.2</td>
      <td>0.60</td>
      <td>38.9</td>
      <td>7.6</td>
      <td>1.8</td>
      <td>2.1</td>
      <td>2.200</td>
      <td>sobota</td>
      <td>Q4</td>
      <td>11</td>
      <td>45</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>166.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1184386</th>
      <td>0</td>
      <td>5c62bc241a466d77f43a47ba</td>
      <td>1146</td>
      <td>5c13adaae343b123806613c2</td>
      <td>59005cd6c5c79d3575eb450d</td>
      <td>Przekąski</td>
      <td>Superfood SPORT - banan, białko</td>
      <td>1</td>
      <td>Bytom</td>
      <td>NaN</td>
      <td>35g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>8</td>
      <td>11.00</td>
      <td>61.3</td>
      <td>14.7</td>
      <td>0.50</td>
      <td>135.8</td>
      <td>388.0</td>
      <td>1.4</td>
      <td>7.8</td>
      <td>5.20</td>
      <td>3.9</td>
      <td>0.01</td>
      <td>215.0</td>
      <td>2.7</td>
      <td>14.9</td>
      <td>42.1</td>
      <td>0.000</td>
      <td>wtorek</td>
      <td>Q2</td>
      <td>5</td>
      <td>20</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.571429</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.571429</td>
      <td>1.511858</td>
      <td>1.511858</td>
      <td>1.511858</td>
      <td>1.511858</td>
      <td>1.511858</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.750000</td>
      <td>0.000000e+00</td>
      <td>6.485000</td>
      <td>6.485000</td>
      <td>0.0</td>
      <td>0.750000</td>
      <td>0.000000</td>
      <td>1.098612</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2010690</th>
      <td>0</td>
      <td>5afc6cc535348125c0a0ce7e</td>
      <td>1069</td>
      <td>5b503c4a9586df16bbb6e07a</td>
      <td>5cd1a4d32b10792bc08dab31</td>
      <td>Pan Pomidor - Pierogi</td>
      <td>Pierogi ruskie</td>
      <td>0</td>
      <td>Kraków</td>
      <td>NaN</td>
      <td>240g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1-6 °C</td>
      <td>5</td>
      <td>5.20</td>
      <td>25.0</td>
      <td>0.9</td>
      <td>0.30</td>
      <td>142.0</td>
      <td>142.0</td>
      <td>0.3</td>
      <td>1.1</td>
      <td>2.20</td>
      <td>5.2</td>
      <td>1.20</td>
      <td>25.0</td>
      <td>1.1</td>
      <td>2.2</td>
      <td>0.9</td>
      <td>1.200</td>
      <td>środa</td>
      <td>Q3</td>
      <td>8</td>
      <td>34</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>12.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2098619</th>
      <td>0</td>
      <td>5b97ac06eb573f399702e383</td>
      <td>1125</td>
      <td>5b96678e33246312580716d9</td>
      <td>5d7103b8c8c4a843bc5b5706</td>
      <td>Lucky Fish</td>
      <td>KIMBAP z łososiem</td>
      <td>0</td>
      <td>Kraków</td>
      <td>NaN</td>
      <td>120g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0-6 °C</td>
      <td>5</td>
      <td>7.30</td>
      <td>25.0</td>
      <td>0.5</td>
      <td>0.30</td>
      <td>209.0</td>
      <td>209.0</td>
      <td>0.3</td>
      <td>NaN</td>
      <td>8.40</td>
      <td>7.3</td>
      <td>0.56</td>
      <td>25.0</td>
      <td>NaN</td>
      <td>8.4</td>
      <td>0.5</td>
      <td>0.560</td>
      <td>piątek</td>
      <td>Q4</td>
      <td>10</td>
      <td>41</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.142857</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>-0.142857</td>
      <td>0.377964</td>
      <td>0.487950</td>
      <td>0.487950</td>
      <td>0.487950</td>
      <td>0.487950</td>
      <td>0.487950</td>
      <td>0.487950</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>-0.142857</td>
      <td>0.000000</td>
      <td>-0.142857</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>8.990000e+00</td>
      <td>0.000000</td>
      <td>8.990000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-0.693147</td>
      <td>-0.693147</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1344063</th>
      <td>0</td>
      <td>5bc8733fa6c5c61a73d3a855</td>
      <td>1063</td>
      <td>5bc724f8a6c5c61a73d3469b</td>
      <td>59005cd6c5c79d3575eb450d</td>
      <td>Przekąski</td>
      <td>BeRAW Baton protein 38% - surowe kakao w gorzk...</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>NaN</td>
      <td>60g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>23</td>
      <td>38.00</td>
      <td>36.0</td>
      <td>13.2</td>
      <td>3.24</td>
      <td>229.8</td>
      <td>383.0</td>
      <td>5.4</td>
      <td>NaN</td>
      <td>5.16</td>
      <td>22.8</td>
      <td>0.06</td>
      <td>21.6</td>
      <td>NaN</td>
      <td>8.6</td>
      <td>22.0</td>
      <td>0.036</td>
      <td>sobota</td>
      <td>Q1</td>
      <td>3</td>
      <td>13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1338599</th>
      <td>0</td>
      <td>5b680f31fcf5082d8504c993</td>
      <td>1188</td>
      <td>5b3f02a90663ab48e3369b5b</td>
      <td>590053bdc5c79d3575eb44f6</td>
      <td>Zupy</td>
      <td>Marchewka z imbirem i kaszą jaglaną</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>Zupy</td>
      <td>300g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>1.50</td>
      <td>11.1</td>
      <td>5.9</td>
      <td>2.40</td>
      <td>179.2</td>
      <td>59.7</td>
      <td>0.8</td>
      <td>1.3</td>
      <td>3.40</td>
      <td>4.5</td>
      <td>0.40</td>
      <td>33.2</td>
      <td>4.0</td>
      <td>1.1</td>
      <td>2.0</td>
      <td>1.200</td>
      <td>piątek</td>
      <td>Q3</td>
      <td>8</td>
      <td>32</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3198095</th>
      <td>0</td>
      <td>5b4cab19ee2a423f39dba28f</td>
      <td>1190</td>
      <td>5b3492210663ab48e335a085</td>
      <td>590053bdc5c79d3575eb44f6</td>
      <td>Zupy</td>
      <td>Pomidor z chili</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>Zupy</td>
      <td>300g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>0.50</td>
      <td>2.9</td>
      <td>5.2</td>
      <td>2.40</td>
      <td>60.9</td>
      <td>20.3</td>
      <td>0.8</td>
      <td>1.1</td>
      <td>2.00</td>
      <td>1.5</td>
      <td>0.50</td>
      <td>8.6</td>
      <td>3.3</td>
      <td>1.0</td>
      <td>1.7</td>
      <td>1.000</td>
      <td>niedziela</td>
      <td>Q2</td>
      <td>6</td>
      <td>22</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1958226</th>
      <td>0</td>
      <td>5c9e04be80237d156eb00b85</td>
      <td>1188</td>
      <td>5ca1ac861b2b36155c424c73</td>
      <td>590053bdc5c79d3575eb44f6</td>
      <td>Zupy</td>
      <td>Zupa Marchewkowa</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>Zupy</td>
      <td>300g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>1.50</td>
      <td>11.1</td>
      <td>5.9</td>
      <td>2.40</td>
      <td>179.2</td>
      <td>59.7</td>
      <td>0.8</td>
      <td>1.3</td>
      <td>3.40</td>
      <td>4.5</td>
      <td>0.40</td>
      <td>33.2</td>
      <td>4.0</td>
      <td>1.1</td>
      <td>2.0</td>
      <td>1.200</td>
      <td>poniedziałek</td>
      <td>Q2</td>
      <td>5</td>
      <td>19</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>10.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>167389</th>
      <td>0</td>
      <td>59aff3feffa5f308210e04d9</td>
      <td>1154</td>
      <td>596ca2e8cf81a34d0c22e638</td>
      <td>591301c83dd75608a9c2ef1b</td>
      <td>Napoje</td>
      <td>Smoothie BeRAW Detox #coolGREENS</td>
      <td>1</td>
      <td>Kraków</td>
      <td>NaN</td>
      <td>250ml</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>piątek</td>
      <td>Q1</td>
      <td>3</td>
      <td>12</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>15.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4106338</th>
      <td>0</td>
      <td>5c6eaec7531c6b7fe27e1658</td>
      <td>1113</td>
      <td>596ca436cf81a34d0c22e644</td>
      <td>5a6f110ca0899f5ca2f7d6e9</td>
      <td>Dania Lunch Duże</td>
      <td>Pieczeń rzymska w siemieniu lnianym</td>
      <td>0</td>
      <td>Kraków</td>
      <td>Kuchnia Słowiańska</td>
      <td>500g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>7.00</td>
      <td>4.5</td>
      <td>1.8</td>
      <td>11.90</td>
      <td>520.0</td>
      <td>104.0</td>
      <td>2.4</td>
      <td>5.4</td>
      <td>26.50</td>
      <td>35.0</td>
      <td>1.00</td>
      <td>22.5</td>
      <td>26.8</td>
      <td>5.3</td>
      <td>0.4</td>
      <td>5.000</td>
      <td>czwartek</td>
      <td>Q3</td>
      <td>7</td>
      <td>28</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>27.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>55908</th>
      <td>0</td>
      <td>5baa229ca27a6774b7fa92b9</td>
      <td>1049</td>
      <td>5baa1ba0a27a6774b7fa8fc0</td>
      <td>5a6f110ca0899f5ca2f7d6e9</td>
      <td>Dania Lunch Duże</td>
      <td>Ostra wołowina z kaszą gryczaną</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>Dieta Samuraja</td>
      <td>500g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>7.60</td>
      <td>12.8</td>
      <td>4.0</td>
      <td>3.50</td>
      <td>495.0</td>
      <td>99.0</td>
      <td>0.7</td>
      <td>1.4</td>
      <td>8.00</td>
      <td>38.0</td>
      <td>0.70</td>
      <td>64.0</td>
      <td>7.0</td>
      <td>1.6</td>
      <td>0.8</td>
      <td>3.500</td>
      <td>środa</td>
      <td>Q2</td>
      <td>5</td>
      <td>20</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.377964</td>
      <td>0.377964</td>
      <td>0.487950</td>
      <td>0.487950</td>
      <td>0.487950</td>
      <td>0.487950</td>
      <td>0.377964</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.099000e+01</td>
      <td>16.990000</td>
      <td>16.990000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1830463</th>
      <td>0</td>
      <td>5ce29992d88a9a3711818c7e</td>
      <td>1189</td>
      <td>5d15ee3e4152364df186f6e3</td>
      <td>590053bdc5c79d3575eb44f6</td>
      <td>Zupy</td>
      <td>Zupa Buraczkowa</td>
      <td>0</td>
      <td>Kraków</td>
      <td>Zupy</td>
      <td>300g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>1.20</td>
      <td>6.6</td>
      <td>12.6</td>
      <td>4.10</td>
      <td>130.3</td>
      <td>43.4</td>
      <td>1.4</td>
      <td>1.9</td>
      <td>4.70</td>
      <td>3.5</td>
      <td>0.40</td>
      <td>199.0</td>
      <td>5.6</td>
      <td>1.6</td>
      <td>4.2</td>
      <td>1.300</td>
      <td>piątek</td>
      <td>Q3</td>
      <td>8</td>
      <td>31</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>39.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5142272</th>
      <td>0</td>
      <td>595635fe2e5db6320199ecbc</td>
      <td>1004</td>
      <td>5bacd53826a9cb3d33cf4c42</td>
      <td>5a6f110ca0899f5ca2f7d6e9</td>
      <td>Dania Lunch Duże</td>
      <td>Indyk z pieczarkami i ryżem</td>
      <td>0</td>
      <td>Jagiellońska 74</td>
      <td>Dieta Samuraja</td>
      <td>500g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>6.90</td>
      <td>17.3</td>
      <td>11.3</td>
      <td>5.20</td>
      <td>543.6</td>
      <td>108.7</td>
      <td>1.0</td>
      <td>1.6</td>
      <td>7.80</td>
      <td>34.3</td>
      <td>0.50</td>
      <td>86.7</td>
      <td>8.0</td>
      <td>1.6</td>
      <td>2.3</td>
      <td>2.300</td>
      <td>wtorek</td>
      <td>Q1</td>
      <td>3</td>
      <td>13</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1225585</th>
      <td>0</td>
      <td>5b5b101cec42762a8558ba6d</td>
      <td>1181</td>
      <td>5b5ebc2b01925031d92916bb</td>
      <td>5cd1a4f40a544c2d0d156fea</td>
      <td>Pan Pomidor - Zupy</td>
      <td>Marokańska z quinoą, batatem i kolendrą</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>NaN</td>
      <td>400g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-6 °C</td>
      <td>8</td>
      <td>1.60</td>
      <td>5.1</td>
      <td>2.6</td>
      <td>0.10</td>
      <td>45.0</td>
      <td>45.0</td>
      <td>0.1</td>
      <td>NaN</td>
      <td>1.50</td>
      <td>1.6</td>
      <td>0.78</td>
      <td>5.1</td>
      <td>NaN</td>
      <td>1.5</td>
      <td>2.6</td>
      <td>0.780</td>
      <td>wtorek</td>
      <td>Q3</td>
      <td>9</td>
      <td>36</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>-0.142857</td>
      <td>0.142857</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.098612</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2605247</th>
      <td>0</td>
      <td>5c18e6dde343b1238066f470</td>
      <td>1124</td>
      <td>5b86a06b2dd588270f00a981</td>
      <td>591301913dd75608a9c2ef19</td>
      <td>Śniadania</td>
      <td>FitBreak - Ryż na mleku kokosowym z owocami</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>--</td>
      <td>350g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>2.00</td>
      <td>23.2</td>
      <td>26.5</td>
      <td>14.30</td>
      <td>505.8</td>
      <td>144.5</td>
      <td>4.1</td>
      <td>1.2</td>
      <td>17.10</td>
      <td>7.0</td>
      <td>0.10</td>
      <td>81.3</td>
      <td>4.2</td>
      <td>4.9</td>
      <td>7.6</td>
      <td>0.500</td>
      <td>wtorek</td>
      <td>Q2</td>
      <td>4</td>
      <td>16</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4130720</th>
      <td>0</td>
      <td>595638312e5db6320199ecc1</td>
      <td>1155</td>
      <td>5af529626e089c600a14f590</td>
      <td>591301c83dd75608a9c2ef1b</td>
      <td>Napoje</td>
      <td>Smoothie Be Raw Vegan protein mango</td>
      <td>1</td>
      <td>Kraków</td>
      <td>NaN</td>
      <td>250ml</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>środa</td>
      <td>Q3</td>
      <td>8</td>
      <td>34</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>-0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.377964</td>
      <td>0.377964</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2507865</th>
      <td>1</td>
      <td>5b1bb68ecaef965005d0e938</td>
      <td>1073</td>
      <td>5b1f8a17caef965005d12d7e</td>
      <td>5cb9b8eedf68013fb09db8f0</td>
      <td>Makarony</td>
      <td>Bolognese drobiowe</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>NaN</td>
      <td>230g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>8.90</td>
      <td>13.8</td>
      <td>3.7</td>
      <td>1.80</td>
      <td>266.8</td>
      <td>116.0</td>
      <td>0.8</td>
      <td>3.8</td>
      <td>4.40</td>
      <td>20.5</td>
      <td>0.80</td>
      <td>317.0</td>
      <td>8.7</td>
      <td>1.9</td>
      <td>1.6</td>
      <td>1.700</td>
      <td>środa</td>
      <td>Q3</td>
      <td>7</td>
      <td>31</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0.428571</td>
      <td>0.857143</td>
      <td>0.857143</td>
      <td>0.857143</td>
      <td>0.857143</td>
      <td>0.857143</td>
      <td>1.000000</td>
      <td>-0.571429</td>
      <td>0.786796</td>
      <td>1.573592</td>
      <td>1.573592</td>
      <td>1.573592</td>
      <td>1.573592</td>
      <td>1.573592</td>
      <td>1.527525</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>-2.0</td>
      <td>1.0</td>
      <td>-2.0</td>
      <td>3.0</td>
      <td>-3.0</td>
      <td>0.0</td>
      <td>-3.0</td>
      <td>-0.428571</td>
      <td>0.000000</td>
      <td>-0.428571</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.666667</td>
      <td>0.333333</td>
      <td>4.993333e+00</td>
      <td>2.330000</td>
      <td>7.990000</td>
      <td>0.0</td>
      <td>0.666667</td>
      <td>-0.693147</td>
      <td>-0.847298</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1694986</th>
      <td>0</td>
      <td>5baa229ca27a6774b7fa92b9</td>
      <td>1176</td>
      <td>5d15ed5b5376d84d27e4568d</td>
      <td>5cd1a4f40a544c2d0d156fea</td>
      <td>Pan Pomidor - Zupy</td>
      <td>Indyjska z ciecierzycą i curry</td>
      <td>0</td>
      <td>Kraków</td>
      <td>NaN</td>
      <td>400g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-6 °C</td>
      <td>8</td>
      <td>2.30</td>
      <td>3.8</td>
      <td>0.8</td>
      <td>0.10</td>
      <td>47.0</td>
      <td>47.0</td>
      <td>0.1</td>
      <td>NaN</td>
      <td>1.50</td>
      <td>2.3</td>
      <td>0.86</td>
      <td>3.8</td>
      <td>NaN</td>
      <td>1.5</td>
      <td>0.8</td>
      <td>0.860</td>
      <td>czwartek</td>
      <td>Q3</td>
      <td>8</td>
      <td>35</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.285714</td>
      <td>0.428571</td>
      <td>0.428571</td>
      <td>0.428571</td>
      <td>0.428571</td>
      <td>0.428571</td>
      <td>0.285714</td>
      <td>0.000000</td>
      <td>0.487950</td>
      <td>0.534522</td>
      <td>0.534522</td>
      <td>0.534522</td>
      <td>0.534522</td>
      <td>0.534522</td>
      <td>0.487950</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>-2.0</td>
      <td>-0.142857</td>
      <td>0.142857</td>
      <td>-0.285714</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.598500e+01</td>
      <td>0.000000</td>
      <td>15.985000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>-0.405465</td>
      <td>0.693147</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4886544</th>
      <td>1</td>
      <td>5b8d24dc90b4de59396b66c5</td>
      <td>1122</td>
      <td>59562d9de9b1ed3755974024</td>
      <td>5a6f110ca0899f5ca2f7d6e9</td>
      <td>Dania Lunch Duże</td>
      <td>Chili con carne z ryżem</td>
      <td>0</td>
      <td>Kraków</td>
      <td>Kuchnia Latynoska</td>
      <td>500g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>6.50</td>
      <td>14.2</td>
      <td>9.5</td>
      <td>3.30</td>
      <td>515.0</td>
      <td>103.0</td>
      <td>0.7</td>
      <td>4.0</td>
      <td>6.70</td>
      <td>32.6</td>
      <td>0.70</td>
      <td>710.0</td>
      <td>19.8</td>
      <td>1.3</td>
      <td>1.9</td>
      <td>3.500</td>
      <td>środa</td>
      <td>Q3</td>
      <td>9</td>
      <td>36</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>2.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3584301</th>
      <td>0</td>
      <td>5b30d00b0663ab48e33548c3</td>
      <td>1168</td>
      <td>5b3492210663ab48e335a085</td>
      <td>5a0033206cdc0d08a6591bfb</td>
      <td>Dania Lunch Małe</td>
      <td>Indyk pieczony z warzywami korzeniowymi</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>Dieta Paleo</td>
      <td>350g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>6.90</td>
      <td>11.1</td>
      <td>7.4</td>
      <td>3.70</td>
      <td>300.7</td>
      <td>85.9</td>
      <td>1.1</td>
      <td>2.2</td>
      <td>6.20</td>
      <td>24.2</td>
      <td>0.60</td>
      <td>38.9</td>
      <td>7.6</td>
      <td>1.8</td>
      <td>2.1</td>
      <td>2.200</td>
      <td>środa</td>
      <td>Q4</td>
      <td>10</td>
      <td>40</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>452133</th>
      <td>0</td>
      <td>5b471f19ee2a423f39da6fcd</td>
      <td>1124</td>
      <td>59562d52e9b1ed3755974022</td>
      <td>591301913dd75608a9c2ef19</td>
      <td>Śniadania</td>
      <td>FitBreak - Ryż na mleku kokosowym z owocami</td>
      <td>0</td>
      <td>Prądnicka 65</td>
      <td>--</td>
      <td>350g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>2.00</td>
      <td>23.2</td>
      <td>26.5</td>
      <td>14.30</td>
      <td>505.8</td>
      <td>144.5</td>
      <td>4.1</td>
      <td>1.2</td>
      <td>17.10</td>
      <td>7.0</td>
      <td>0.10</td>
      <td>81.3</td>
      <td>4.2</td>
      <td>4.9</td>
      <td>7.6</td>
      <td>0.500</td>
      <td>sobota</td>
      <td>Q2</td>
      <td>6</td>
      <td>25</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>-0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.377964</td>
      <td>0.377964</td>
      <td>0.377964</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.098612</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2834580</th>
      <td>0</td>
      <td>5a5c7e0acf5c8134b3a98c8f</td>
      <td>1101</td>
      <td>5bd9aff772847a312e4d8c8e</td>
      <td>5cd1a4f40a544c2d0d156fea</td>
      <td>Pan Pomidor - Zupy</td>
      <td>Pomidorowa z bazylią</td>
      <td>0</td>
      <td>Kraków</td>
      <td>NaN</td>
      <td>400g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-6 °C</td>
      <td>8</td>
      <td>1.20</td>
      <td>4.6</td>
      <td>4.6</td>
      <td>0.80</td>
      <td>44.0</td>
      <td>44.0</td>
      <td>0.8</td>
      <td>NaN</td>
      <td>1.70</td>
      <td>1.2</td>
      <td>0.64</td>
      <td>4.6</td>
      <td>NaN</td>
      <td>1.7</td>
      <td>4.6</td>
      <td>0.640</td>
      <td>środa</td>
      <td>Q4</td>
      <td>11</td>
      <td>45</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>49.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1323942</th>
      <td>0</td>
      <td>5956341c2e5db6320199ecb3</td>
      <td>1061</td>
      <td>593f7d34b0bea3175242d9bf</td>
      <td>5abe0aed049e180557e22330</td>
      <td>Sałatki</td>
      <td>FitSalad - Sałatka z fetą, oliwkami i pomidork...</td>
      <td>0</td>
      <td>Kraków</td>
      <td>--</td>
      <td>350g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>4.40</td>
      <td>15.8</td>
      <td>1.8</td>
      <td>7.00</td>
      <td>504.0</td>
      <td>144.0</td>
      <td>2.0</td>
      <td>2.1</td>
      <td>22.70</td>
      <td>15.4</td>
      <td>0.80</td>
      <td>55.3</td>
      <td>7.5</td>
      <td>6.5</td>
      <td>0.5</td>
      <td>2.700</td>
      <td>niedziela</td>
      <td>Q4</td>
      <td>10</td>
      <td>43</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>166.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1601686</th>
      <td>0</td>
      <td>5a6a0b87a0899f5ca2f7c6db</td>
      <td>1160</td>
      <td>5ce29739d88a9a3711818bbb</td>
      <td>5a6f110ca0899f5ca2f7d6e9</td>
      <td>Dania Lunch Duże</td>
      <td>Kurczak w sosie orzechowym z ryżem</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>Kuchnia Orientalna</td>
      <td>500g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>9.70</td>
      <td>9.1</td>
      <td>0.3</td>
      <td>6.00</td>
      <td>620.0</td>
      <td>124.0</td>
      <td>1.2</td>
      <td>4.0</td>
      <td>22.50</td>
      <td>48.3</td>
      <td>0.60</td>
      <td>455.0</td>
      <td>20.0</td>
      <td>4.5</td>
      <td>0.1</td>
      <td>3.200</td>
      <td>poniedziałek</td>
      <td>Q4</td>
      <td>10</td>
      <td>41</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.285714</td>
      <td>0.428571</td>
      <td>0.571429</td>
      <td>0.714286</td>
      <td>0.571429</td>
      <td>-0.285714</td>
      <td>0.487950</td>
      <td>0.487950</td>
      <td>0.487950</td>
      <td>0.534522</td>
      <td>0.534522</td>
      <td>0.487950</td>
      <td>0.534522</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>-0.142857</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.500</td>
      <td>0.500000</td>
      <td>0.000000</td>
      <td>1.119000e+01</td>
      <td>11.190000</td>
      <td>15.990000</td>
      <td>0.0</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>-0.405465</td>
      <td>4.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4453331</th>
      <td>0</td>
      <td>5b471c05ee2a423f39da6f06</td>
      <td>1110</td>
      <td>5b5ebb9901925031d929169b</td>
      <td>5a6f110ca0899f5ca2f7d6e9</td>
      <td>Dania Lunch Duże</td>
      <td>Wołowina z marchewkowym puree</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>Dieta Paleo</td>
      <td>500g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>6.80</td>
      <td>3.8</td>
      <td>3.0</td>
      <td>8.00</td>
      <td>340.0</td>
      <td>68.0</td>
      <td>1.6</td>
      <td>1.3</td>
      <td>12.50</td>
      <td>34.0</td>
      <td>0.80</td>
      <td>19.0</td>
      <td>6.5</td>
      <td>2.5</td>
      <td>0.6</td>
      <td>4.000</td>
      <td>czwartek</td>
      <td>Q4</td>
      <td>10</td>
      <td>42</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>562483</th>
      <td>0</td>
      <td>5c18e2882d1e1a498b462943</td>
      <td>1120</td>
      <td>5c18cf4b2d1e1a498b46228f</td>
      <td>5a0033206cdc0d08a6591bfb</td>
      <td>Dania Lunch Małe</td>
      <td>Risotto z indykiem i cukinią</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>Dieta Samuraja</td>
      <td>350g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>8.80</td>
      <td>12.5</td>
      <td>6.0</td>
      <td>5.10</td>
      <td>385.0</td>
      <td>110.0</td>
      <td>1.5</td>
      <td>1.1</td>
      <td>8.30</td>
      <td>30.7</td>
      <td>0.70</td>
      <td>43.8</td>
      <td>4.0</td>
      <td>2.4</td>
      <td>1.7</td>
      <td>2.300</td>
      <td>środa</td>
      <td>Q2</td>
      <td>6</td>
      <td>24</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>-0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.377964</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.791759</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1600845</th>
      <td>0</td>
      <td>5b8fa4158667b879c9f85c8e</td>
      <td>1101</td>
      <td>5be4424774546727b8c6058c</td>
      <td>5cd1a4f40a544c2d0d156fea</td>
      <td>Pan Pomidor - Zupy</td>
      <td>Pomidorowa z bazylią</td>
      <td>0</td>
      <td>Kraków</td>
      <td>NaN</td>
      <td>400g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-6 °C</td>
      <td>8</td>
      <td>1.20</td>
      <td>4.6</td>
      <td>4.6</td>
      <td>0.80</td>
      <td>44.0</td>
      <td>44.0</td>
      <td>0.8</td>
      <td>NaN</td>
      <td>1.70</td>
      <td>1.2</td>
      <td>0.64</td>
      <td>4.6</td>
      <td>NaN</td>
      <td>1.7</td>
      <td>4.6</td>
      <td>0.640</td>
      <td>czwartek</td>
      <td>Q2</td>
      <td>6</td>
      <td>24</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>504149</th>
      <td>0</td>
      <td>5b6174ec04852c4704a271d7</td>
      <td>1187</td>
      <td>5c90ec23a4b28f5f98967a31</td>
      <td>590053bdc5c79d3575eb44f6</td>
      <td>Zupy</td>
      <td>Zupa Pietruszkowa</td>
      <td>0</td>
      <td>Wrocław, Biskupice Podgórne</td>
      <td>Zupy</td>
      <td>300g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>1.00</td>
      <td>5.9</td>
      <td>8.6</td>
      <td>6.40</td>
      <td>140.0</td>
      <td>46.7</td>
      <td>2.1</td>
      <td>1.7</td>
      <td>7.60</td>
      <td>3.0</td>
      <td>0.40</td>
      <td>17.7</td>
      <td>5.1</td>
      <td>2.5</td>
      <td>2.9</td>
      <td>1.100</td>
      <td>czwartek</td>
      <td>Q4</td>
      <td>12</td>
      <td>52</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5013521</th>
      <td>1</td>
      <td>5cc040375e70463ad9d520dd</td>
      <td>1084</td>
      <td>5cc01cdd8ed57342ff70efd5</td>
      <td>5cb9b8eedf68013fb09db8f0</td>
      <td>Makarony</td>
      <td>Grzybowy z indykiem</td>
      <td>0</td>
      <td>Wrocław</td>
      <td>NaN</td>
      <td>240g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>9.20</td>
      <td>12.3</td>
      <td>0.4</td>
      <td>1.50</td>
      <td>262.5</td>
      <td>105.0</td>
      <td>0.6</td>
      <td>2.8</td>
      <td>3.50</td>
      <td>23.0</td>
      <td>0.70</td>
      <td>308.0</td>
      <td>7.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>1.800</td>
      <td>poniedziałek</td>
      <td>Q3</td>
      <td>7</td>
      <td>30</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.142857</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.142857</td>
      <td>0.377964</td>
      <td>0.377964</td>
      <td>0.377964</td>
      <td>0.377964</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>7.990000e+00</td>
      <td>0.000000</td>
      <td>7.990000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.945910</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4992585</th>
      <td>0</td>
      <td>5b8d24dc90b4de59396b66c5</td>
      <td>1117</td>
      <td>59562d9de9b1ed3755974024</td>
      <td>591301913dd75608a9c2ef19</td>
      <td>Śniadania</td>
      <td>FitRoślanka kakao</td>
      <td>0</td>
      <td>Kraków</td>
      <td>413,0</td>
      <td>350g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>3.16</td>
      <td>17.0</td>
      <td>17.4</td>
      <td>4.20</td>
      <td>413.0</td>
      <td>118.0</td>
      <td>1.2</td>
      <td>5.3</td>
      <td>11.70</td>
      <td>11.1</td>
      <td>0.20</td>
      <td>595.0</td>
      <td>12.1</td>
      <td>3.3</td>
      <td>5.0</td>
      <td>0.600</td>
      <td>poniedziałek</td>
      <td>Q3</td>
      <td>7</td>
      <td>29</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2903817</th>
      <td>0</td>
      <td>5c17ac232d1e1a498b45eef9</td>
      <td>1004</td>
      <td>5c178d48e343b1238066affd</td>
      <td>5a6f110ca0899f5ca2f7d6e9</td>
      <td>Dania Lunch Duże</td>
      <td>Indyk z pieczarkami i ryżem</td>
      <td>0</td>
      <td>Katowice</td>
      <td>Dieta Samuraja</td>
      <td>500g</td>
      <td>2-3 min.</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>6.90</td>
      <td>17.3</td>
      <td>11.3</td>
      <td>5.20</td>
      <td>543.6</td>
      <td>108.7</td>
      <td>1.0</td>
      <td>1.6</td>
      <td>7.80</td>
      <td>34.3</td>
      <td>0.50</td>
      <td>86.7</td>
      <td>8.0</td>
      <td>1.6</td>
      <td>2.3</td>
      <td>2.300</td>
      <td>czwartek</td>
      <td>Q2</td>
      <td>4</td>
      <td>15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3026555</th>
      <td>0</td>
      <td>59703d4771059a6bc70a854d</td>
      <td>1051</td>
      <td>596ca515cf81a34d0c22e645</td>
      <td>5abe0aed049e180557e22330</td>
      <td>Sałatki</td>
      <td>FitSalad - Sałatka z kurczakiem i grillowanymi...</td>
      <td>0</td>
      <td>Kraków</td>
      <td>--</td>
      <td>350g</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>8.50</td>
      <td>13.1</td>
      <td>7.0</td>
      <td>1.70</td>
      <td>413.0</td>
      <td>118.0</td>
      <td>0.5</td>
      <td>2.4</td>
      <td>9.90</td>
      <td>29.9</td>
      <td>0.60</td>
      <td>45.9</td>
      <td>8.5</td>
      <td>2.8</td>
      <td>2.0</td>
      <td>2.000</td>
      <td>czwartek</td>
      <td>Q1</td>
      <td>2</td>
      <td>6</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>31.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>998590</th>
      <td>1</td>
      <td>5c614d5fc88bae77186d4c34</td>
      <td>1122</td>
      <td>5c612de4cfa3a62fcbee67a6</td>
      <td>5a6f110ca0899f5ca2f7d6e9</td>
      <td>Dania Lunch Duże</td>
      <td>Chili con carne z ryżem</td>
      <td>0</td>
      <td>Wrocław</td>
      <td>Kuchnia Latynoska</td>
      <td>500g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>6.50</td>
      <td>14.2</td>
      <td>9.5</td>
      <td>3.30</td>
      <td>515.0</td>
      <td>103.0</td>
      <td>0.7</td>
      <td>4.0</td>
      <td>6.70</td>
      <td>32.6</td>
      <td>0.70</td>
      <td>710.0</td>
      <td>19.8</td>
      <td>1.3</td>
      <td>1.9</td>
      <td>3.500</td>
      <td>poniedziałek</td>
      <td>Q3</td>
      <td>9</td>
      <td>40</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>1.142857</td>
      <td>1.142857</td>
      <td>1.142857</td>
      <td>1.142857</td>
      <td>1.142857</td>
      <td>1.142857</td>
      <td>1.142857</td>
      <td>0.000000</td>
      <td>2.267787</td>
      <td>2.267787</td>
      <td>2.267787</td>
      <td>2.267787</td>
      <td>2.267787</td>
      <td>2.267787</td>
      <td>2.267787</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>-6.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>-4.0</td>
      <td>0.000000</td>
      <td>0.571429</td>
      <td>-0.571429</td>
      <td>2.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.125</td>
      <td>0.875000</td>
      <td>0.000000</td>
      <td>1.661375e+01</td>
      <td>2.622500</td>
      <td>17.738750</td>
      <td>0.0</td>
      <td>0.125000</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3236484</th>
      <td>0</td>
      <td>5b30c32d0663ab48e33546c6</td>
      <td>1168</td>
      <td>5b3484bb0663ab48e3359f48</td>
      <td>5a0033206cdc0d08a6591bfb</td>
      <td>Dania Lunch Małe</td>
      <td>Indyk pieczony z warzywami korzeniowymi</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>Dieta Paleo</td>
      <td>350g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>6.90</td>
      <td>11.1</td>
      <td>7.4</td>
      <td>3.70</td>
      <td>300.7</td>
      <td>85.9</td>
      <td>1.1</td>
      <td>2.2</td>
      <td>6.20</td>
      <td>24.2</td>
      <td>0.60</td>
      <td>38.9</td>
      <td>7.6</td>
      <td>1.8</td>
      <td>2.1</td>
      <td>2.200</td>
      <td>środa</td>
      <td>Q4</td>
      <td>12</td>
      <td>52</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-1.0</td>
      <td>1.0</td>
      <td>0.000000</td>
      <td>-0.142857</td>
      <td>0.142857</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>-1.386294</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4468201</th>
      <td>0</td>
      <td>5ba2370901f82e03b4137e22</td>
      <td>1188</td>
      <td>5ba22beb01f82e03b413792d</td>
      <td>590053bdc5c79d3575eb44f6</td>
      <td>Zupy</td>
      <td>Marchewka z imbirem i kaszą jaglaną</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>Zupy</td>
      <td>300g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>1.50</td>
      <td>11.1</td>
      <td>5.9</td>
      <td>2.40</td>
      <td>179.2</td>
      <td>59.7</td>
      <td>0.8</td>
      <td>1.3</td>
      <td>3.40</td>
      <td>4.5</td>
      <td>0.40</td>
      <td>33.2</td>
      <td>4.0</td>
      <td>1.1</td>
      <td>2.0</td>
      <td>1.200</td>
      <td>wtorek</td>
      <td>Q2</td>
      <td>4</td>
      <td>15</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3933522</th>
      <td>0</td>
      <td>5afc6a7035348125c0a0ce58</td>
      <td>1191</td>
      <td>5a572369cf5c8134b3a98692</td>
      <td>5a0033206cdc0d08a6591bfb</td>
      <td>Dania Lunch Małe</td>
      <td>Hiszpańska Tortilla</td>
      <td>0</td>
      <td>Warszawa</td>
      <td>Dieta Paleo</td>
      <td>180g</td>
      <td>2-3 min.</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2-5 °C</td>
      <td>5</td>
      <td>8.60</td>
      <td>17.0</td>
      <td>2.8</td>
      <td>12.10</td>
      <td>342.5</td>
      <td>201.5</td>
      <td>7.1</td>
      <td>2.1</td>
      <td>18.70</td>
      <td>14.6</td>
      <td>1.10</td>
      <td>28.9</td>
      <td>3.6</td>
      <td>11.0</td>
      <td>1.7</td>
      <td>1.800</td>
      <td>poniedziałek</td>
      <td>Q4</td>
      <td>12</td>
      <td>49</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>6.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



The fist glance at the table above leads to decuding that cooking time in oven or microwave oven are not varrying variables.


```python
all(df.cooking_mv.notna() == df.cooking_ov.notna())
```




    True




```python
inds = df[df.cooking_mv.notna()].index
all(df.cooking_mv[inds] == df.cooking_ov[inds])
```




    True



Atrributes cooking_mv and cooking_ov are not distinguishable


```python
def impute_alt_or_zero(x, y):
    if not np.isnan(x) and not np.isnan(y):
        return max(x, y)
    if not np.isnan(x):
        return x
    if not np.isnan(y):
        return y
    return .0

def join_cooking_mv_and_ov(df):
    df["cooking_mv_or_ov"] = np.vectorize(impute_alt_or_zero)(df.cooking_mv, df.cooking_ov)
```


```python
attrs_to_be_removed = {'cooking_mv', 'cooking_ov'}
numeric_attrs = set(df.select_dtypes(include=np.number).columns.to_list())
```


```python
attrs_with_ids = set(filter(lambda c: "id" in c, attrs))
attrs_with_ids
```




    {'category_id', 'company_id', 'pos_id', 'product_id_unified'}




```python
attrs_with_names = set(filter(lambda c: "name" in c, attrs))
attrs_with_names
```




    {'category_name', 'product_name'}




```python
numeric_attrs -= attrs_with_ids
discrete_attrs = attrs - numeric_attrs
discrete_attrs
```




    {'address_city',
     'category_id',
     'category_name',
     'company_id',
     'cooking_time',
     'diet',
     'pos_id',
     'product_id_unified',
     'product_name',
     'quarter',
     'size',
     'storage_temp',
     'weekday'}




```python
discrete_descr = df[discrete_attrs].describe(include="all")
discrete_descr
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>company_id</th>
      <th>product_id_unified</th>
      <th>diet</th>
      <th>category_name</th>
      <th>address_city</th>
      <th>product_name</th>
      <th>weekday</th>
      <th>size</th>
      <th>storage_temp</th>
      <th>cooking_time</th>
      <th>category_id</th>
      <th>pos_id</th>
      <th>quarter</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5360496</td>
      <td>5.360496e+06</td>
      <td>2840752</td>
      <td>5360496</td>
      <td>5360496</td>
      <td>5360496</td>
      <td>5360496</td>
      <td>5360496</td>
      <td>3930214</td>
      <td>3632502</td>
      <td>5360496</td>
      <td>5360496</td>
      <td>5360496</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>298</td>
      <td>NaN</td>
      <td>10</td>
      <td>14</td>
      <td>33</td>
      <td>137</td>
      <td>7</td>
      <td>22</td>
      <td>5</td>
      <td>1</td>
      <td>14</td>
      <td>375</td>
      <td>4</td>
    </tr>
    <tr>
      <th>top</th>
      <td>5a587706cf5c8134b3a9891d</td>
      <td>NaN</td>
      <td>Dieta Samuraja</td>
      <td>Dania Lunch Duże</td>
      <td>Warszawa</td>
      <td>Dyniowe curry z indykiem</td>
      <td>czwartek</td>
      <td>350g</td>
      <td>2-5 °C</td>
      <td>2-3 min.</td>
      <td>5a6f110ca0899f5ca2f7d6e9</td>
      <td>59fc5325c94b722506678bd1</td>
      <td>Q3</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>101533</td>
      <td>NaN</td>
      <td>994953</td>
      <td>1144201</td>
      <td>2353704</td>
      <td>112303</td>
      <td>776919</td>
      <td>1195365</td>
      <td>3477491</td>
      <td>3632502</td>
      <td>1144201</td>
      <td>25498</td>
      <td>2034382</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>NaN</td>
      <td>1.104991e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>NaN</td>
      <td>5.583013e+01</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>NaN</td>
      <td>1.004000e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>NaN</td>
      <td>1.057000e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>NaN</td>
      <td>1.117000e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>NaN</td>
      <td>1.152000e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>NaN</td>
      <td>1.193000e+03</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



I remove category and product ids in favour of the names.


```python
attrs_to_be_removed.update({'category_id', 'product_id_unified'})
discrete_attrs = attrs - numeric_attrs - attrs_to_be_removed
```

### Size attribute


```python
list(pd.unique(df["size"]))
```




    ['35g',
     '350g',
     '240g',
     '500g',
     '250ml',
     '180g',
     '50g',
     '300g',
     '230g',
     '400g',
     '40g',
     '60g',
     '250g',
     '370g',
     '380g',
     '300ml',
     '200ml',
     '120g',
     '85g',
     '220g',
     '2x68g',
     '100g']



Size attribute could be convert to numeric weight and volume attributes.


```python
import re

def multiply_all_numbers_from_str(str):
    return np.prod(list(map(float, re.findall(r"\d+", str))))

def get_weight(size):
    if not size.endswith("g"):
        return np.nan
    return multiply_all_numbers_from_str(size)

def get_volume(size):
    if not size.endswith("ml"):
        return np.nan
    return multiply_all_numbers_from_str(size)

def convert_size_to_weight_and_volume(df):
    df["weight"] = np.vectorize(get_weight)(df["size"])
    df["volume"] = np.vectorize(get_volume)(df["size"])
```


```python
attrs_to_be_removed.add("size")
discrete_attrs.remove("size")
numeric_attrs.update({"weight", "volume"})
```

### Storage temperature attribute


```python
pd.unique(df.storage_temp)
```




    array([nan, '2-5 °C', '1-6 °C', '2-6 °C', '0-6 °C', '0-7 °C'],
          dtype=object)



I will change storage temperature to stored in fridge attribute.


```python
def convert_storage_temp_to_in_fridge(df):
    df["stored_in_fridge"] = df["storage_temp"].notna() * 1.0

attrs_to_be_removed.add("storage_temp")
discrete_attrs.remove("storage_temp")
numeric_attrs.add("stored_in_fridge")
```

### Generalizing some discrete attributes

There are some discrete attributes with no `NA` in any record and some of them should be generalizable.


```python
counts = discrete_descr.loc["count",]
counts[counts == df.shape[0]].keys().intersection(discrete_attrs)
```




    Index(['company_id', 'category_name', 'address_city', 'product_name',
           'weekday', 'pos_id', 'quarter'],
          dtype='object')



'weekday' and 'quarter' attributes are not extensible so I will focus on 'pos_id', 'company_id', 'product_name', 'category_name', 'address_city'.

#### Sales locations


```python
pos_counts = df.pos_id.value_counts()
pos_counts
```




    59fc5325c94b722506678bd1    25498
    595e36122e84531cd204096b    24850
    5b61699404852c4704a26f28    24658
    5b3381160663ab48e335895f    24620
    5b1b963bcaef965005d0e6b0    24590
                                ...  
    5dc694443bf8f15dc97ec3c7       78
    5dc69e723bf8f15dc97ec4de       46
    5dcd4c06dacdf109c179d79b       31
    5dcd2fb58fc7d108e6b8bc09       29
    5dcbccb4aa5f032f78d77579       28
    Name: pos_id, Length: 375, dtype: int64




```python
pos_counts[pos_counts < 300]
```




    5dbff693bb783c5423163b8d    292
    5dbac6883bf8f15dc97d4251    210
    5dc28ac6b902d15de1f0152b    193
    5c7901d27b6c863a5d521f21     93
    5dc694443bf8f15dc97ec3c7     78
    5dc69e723bf8f15dc97ec4de     46
    5dcd4c06dacdf109c179d79b     31
    5dcd2fb58fc7d108e6b8bc09     29
    5dcbccb4aa5f032f78d77579     28
    Name: pos_id, dtype: int64



I will delete information for locations with less than 100 sales records for training purposes. I expect the samples with the location information removed to help me generalize the predictive model for new data. I want it to predict sales in completely new locations.


```python
pos_id_cats = pos_counts[pos_counts >= 100].index
pos_ids_to_be_forgotten = set(pos_counts.index.difference(pos_id_cats))
np.sum(pos_counts[pos_ids_to_be_forgotten])
```




    305



#### Companies


```python
company_counts = df.company_id.value_counts()
company_counts
```




    5a587706cf5c8134b3a9891d    101533
    5c48270855f3af637b69b13f     94045
    5b9fa5e201f82e03b412bcff     81488
    5b503c4a9586df16bbb6e07a     77930
    5c924f69b6cb840dcac430fe     75667
                                 ...  
    5dc534a2fdcd0622bd7b34e5       239
    5dbff3dd9f38f239c2ffeb0c       188
    5e4550dd177446520bcd2c91       101
    5d764e24bf7a586310594da4        72
    5d764e2fd0c59c62f781ff36        46
    Name: company_id, Length: 298, dtype: int64




```python
company_counts[company_counts<300]
```




    5dbff5229f38f239c2ffeb55    270
    5ce2969fc828ba60d8093297    245
    5dc534a2fdcd0622bd7b34e5    239
    5dbff3dd9f38f239c2ffeb0c    188
    5e4550dd177446520bcd2c91    101
    5d764e24bf7a586310594da4     72
    5d764e2fd0c59c62f781ff36     46
    Name: company_id, dtype: int64




```python
company_id_cats = company_counts[company_counts >= 200].index
companies_to_be_forgotten = company_counts.index.difference(company_id_cats)
np.sum(company_counts[companies_to_be_forgotten])
```




    407



#### Products names


```python
product_counts = df.product_name.value_counts()
product_counts
```




    Dyniowe curry z indykiem                 112303
    Ostra wołowina z kaszą gryczaną          109728
    Wołowina z marchewkowym puree            108298
    Risotto z indykiem i cukinią             107248
    Szaszłyk z ryżem w kurkumie              104697
                                              ...  
    Silny Łasuch - Łasuch i jego orzeszki       153
    FitElixir - Black                           144
    Silny Łasuch - 2w1                           95
    All'arrabiata                                71
    Pomidorowa z chilli                           9
    Name: product_name, Length: 137, dtype: int64




```python
product_counts[product_counts<400]
```




    Silny Łasuch - Marchew w tropikach       342
    Silny Łasuch - Cynamonowe jabłuszko      342
    Chili con carne z ryżen                  230
    ROŚLEKO - jaglano - orzechowe            218
    Allarrabbiata                            203
    Superfood - ZDROWIE - kakao - maca       193
    Kaszotto z soczewicą i grzybami          190
    ROŚLEKO - kakao                          184
    Silny Łasuch - Łasuch i jego orzeszki    153
    FitElixir - Black                        144
    Silny Łasuch - 2w1                        95
    All'arrabiata                             71
    Pomidorowa z chilli                        9
    Name: product_name, dtype: int64



I will delete names of products with less than 200 sales records for training purposes. I would like to predict sales for completely new products.


```python
product_name_cats = product_counts[product_counts >= 200].index
products_names_to_be_forgotten = product_counts.index.difference(product_name_cats)
np.sum(product_counts[products_names_to_be_forgotten])
```




    1039



#### Cattegories


```python
category_counts = df.category_name.value_counts()
category_counts
```




    Dania Lunch Duże         1144201
    Przekąski                 929010
    Dania Lunch Małe          797960
    Napoje                    498033
    Zupy                      486330
    Makarony                  412650
    Śniadania                 265078
    Pan Pomidor - Zupy        258243
    Sałatki                   208141
    Pan Pomidor - Pierogi     166430
    Mr Thai                   157172
    Lucky Fish                 28050
    Desery                      5557
    DayUp                       3641
    Name: category_name, dtype: int64




```python
len(category_counts)
```




    14



There only 14 product categories so I do nothing with them and I do not expect the presence of any new category.

#### Address cities


```python
address_counts = df.address_city.value_counts()
address_counts
```




    Warszawa                           2353704
    Kraków                             1577282
    Wrocław                             694811
    Katowice                            234123
    Skawina                              71726
    11                                   50702
    Gliwice                              41412
    Niepołomice                          39114
    Wieliczka                            38747
    Kobierzyce, Bielany Wrocławskie      28283
    Zabierzów                            25683
    Balice                               22416
    Wroclaw                              20324
    Jagiellońska 74                      16850
    30-001                               15393
    Bielany Wrocławskie                  14924
    Waszawa                              14923
    Ruda Śląska                          14873
    Prądnicka 65                         14257
    Podłęże, Kraków                      11648
    Wysoka, Wrocław                      11360
    Bytom                                 8362
    Wrocław, Biskupice Podgórne           7342
    Wrocławiu                             6949
    51.126901, 16.978188                  6130
    Nowa Wieś Wrocławska                  5358
    Krakow                                5084
    12                                    4554
    02-092                                2465
    Wysoka                                1072
    30-150                                 239
    21                                     194
    Zabłocie 20/22                         192
    Name: address_city, dtype: int64



I will delete address city information for the locations with less than 200 sales records. I would like to predict sales at unknown places.


```python
address_city_cats = address_counts[address_counts >= 200].index
addresses_to_be_forgotten = address_counts.index.difference(address_city_cats)
np.sum(address_counts[addresses_to_be_forgotten])
```




    386



#### Saving the indices of the samples with attributes values to be forgotten
I will use them for mantaining generality the model and validating the trained results.


```python
missing_cats_samples_idxs = {}
missing_cats_samples_idxs["pos_id"] = np.where(df.pos_id.isin(pos_ids_to_be_forgotten))[0]
missing_cats_samples_idxs["company_id"] = np.where(df.company_id.isin(companies_to_be_forgotten))[0]
missing_cats_samples_idxs["product_name"] = np.where(df.product_name.isin(products_names_to_be_forgotten))[0]
missing_cats_samples_idxs["address_city"] = np.where(df.address_city.isin(addresses_to_be_forgotten))[0]

important_missings_attrs = ["pos_id", "company_id", "product_name", "address_city"]
```

### Preprocessing the data


```python
cats_for_attr = {'pos_id': pos_id_cats,
                 'company_id': company_id_cats,
                 'product_name': product_name_cats,
                 'address_city': address_city_cats}

for attr in discrete_attrs - {'pos_id', 'company_id', 'product_name', 'address_city'}:
    _, uniques = pd.factorize(df[attr])
    cats_for_attr[attr] = uniques
```


```python
def remove_unnecessary_attrs(df):
    for attr in attrs_to_be_removed:
        del df[attr]

def convert_to_categoricals(df):
    for attr, attr_cats in cats_for_attr.items():
        df[attr] = df[attr].astype(pd.CategoricalDtype(attr_cats))

def with_categoricals_as_dummies(df):
    return pd.get_dummies(df,dummy_na=True, sparse=True)

def preprocessed_data_in_place(df):
    convert_size_to_weight_and_volume(df)
    join_cooking_mv_and_ov(df)
    convert_storage_temp_to_in_fridge(df)
    convert_to_categoricals(df)
    remove_unnecessary_attrs(df)
    return with_categoricals_as_dummies(df)
```

## Spliting data for training and validation

### Set weights of samples and extract important ones


```python
samples_weights = np.empty(df.shape[0])
samples_weights.fill(0.01)
important_indices = set()

for attr in important_missings_attrs:
    for idx in missing_cats_samples_idxs[attr]:
        samples_weights[idx] += 0.24
    important_indices.update(missing_cats_samples_idxs[attr])

important_indices = pd.Index(important_indices).unique()
    
important_samples = df.loc[important_indices, ]

avg_samples = df.drop(important_indices, axis=0)
```

### Create training and validation sets


```python
from sklearn.model_selection import train_test_split
avg_samples_train, avg_samples_val = train_test_split(avg_samples, random_state=0, test_size=0.05)
impt_samples_train, impt_samples_val = train_test_split(important_samples, random_state=0, test_size=0.3)
```


```python
samples_train = pd.concat([avg_samples_train, impt_samples_train]).sample(frac=1)
samples_val = pd.concat([avg_samples_val, impt_samples_val]).sample(frac=1)
```


```python
def x_y_split(df):
    y = df.will_it_sell
    X = df.drop(["will_it_sell"], axis=1)
    return X, y
    
def as_sparse(df):
    return df.astype(pd.SparseDtype("float", np.nan))
```


```python
X_train, y_train = x_y_split(samples_train)
X_val, y_val = x_y_split(samples_val)

X_train = as_sparse(preprocessed_data_in_place(X_train))
y_train = as_sparse(y_train)
X_val = as_sparse(preprocessed_data_in_place(X_val))
y_val = as_sparse(y_val)
```


```python
X_impt_val, y_impt_val = x_y_split(impt_samples_val)
X_impt_val = as_sparse(preprocessed_data_in_place(X_impt_val))
y_impt_val = as_sparse(y_impt_val)
```

## Chalange test set preparation and answer saver


```python
X_test = pd.read_csv("FitFood_competition_data_test.csv", sep=";")
X_test = preprocessed_data_in_place(X_test).drop(["will_it_sell"], 1)
X_test = X_test.astype(pd.SparseDtype("float", np.nan))
```


```python
import datetime

def save_ans(ans, ans_prefix_name="ans"):
    cur_dt = datetime.datetime.today()
    str_dt = cur_dt.strftime("%y-%m-%d_%H-%M-%S")
    file_name = f"{ans_prefix_name}_{str_dt}"
    with open(file_name, "w") as f:
        for p in ans:
            print(p, file=f)
    print(f"The answer saved as '{file_name}'")
            
            
def predict_ans(cls, X_test):
    return cls.predict_proba(X_test)[:,1]

def predict_and_save_ans(cls, ans_prefix_name="ans", X_test=X_test):
    ans = predict_ans(cls, X_test)
    save_ans(ans, ans_prefix_name)
```

## Features selection

### Analysis of correlations with target value


```python
df = preprocessed_data_in_place(df)
corrs = df.corrwith(df.will_it_sell)
corrs.sort_values(inplace=True, ascending=False)
```


```python
corrs[abs(corrs)>=0.04]
```




    will_it_sell                                       1.000000
    sum_qty                                            0.512098
    meanLastPeriod_lag1                                0.482397
    sdLastPeriod_lag1                                  0.479201
    meanLastPeriod_lag2                                0.471712
    sdLastPeriod_lag2                                  0.467968
    maxLastPeriod_lag1                                 0.467142
    meanLastPeriod_lag3                                0.461228
    sdLastPeriod_lag3                                  0.456877
    meanLastPeriod_lag4                                0.450535
    sdLastPeriod_lag4                                  0.445714
    meanLastPeriod_lag5                                0.439661
    sdLastPeriod_lag5                                  0.434390
    meanLastPeriod_lag6                                0.428485
    sdLastPeriod_lag6                                  0.422924
    meanLastPeriod_lag7                                0.417058
    sdLastPeriod_lag7                                  0.411217
    sales_since_prev_delivery                          0.407250
    maxLastPeriod_lag7                                 0.401263
    avg_total_base_lag1                                0.399453
    avg_total_lag1                                     0.363397
    avg_transaction_discount_count_lag1                0.308491
    avg_from_paypass_lag1                              0.305060
    qty_lag1                                           0.282265
    qty_lag2                                           0.273964
    qty_lag3                                           0.269665
    qty_lag4                                           0.265734
    qty_lag5                                           0.262395
    qty_lag6                                           0.258840
    qty_lag7                                           0.251873
    qty_lag8                                           0.241997
    qty_lag9                                           0.234147
    avg_total_to_discount_lag1                         0.232499
    qty_lag10                                          0.228964
    qty_lag11                                          0.224393
    qty_lag12                                          0.220038
    qty_lag13                                          0.215637
    qty_lag14                                          0.209650
    avg_from_blik_lag1                                 0.188022
    product_name_Dyniowe curry z indykiem              0.181632
    stored_in_fridge                                   0.178211
    cooking_time_2-3 min.                              0.171438
    weight                                             0.159482
    avg_from_payu_lag1                                 0.159038
    bialko_calk                                        0.154345
    category_name_Mr Thai                              0.144743
    sol_calk                                           0.142263
    category_name_Dania Lunch Duże                     0.133613
    sol_100                                            0.130555
    product_name_Pad Thai Chicken                      0.118034
    diet_Dieta Samuraja                                0.108491
    energia_calk                                       0.101061
    product_name_Sesame Beef                           0.099523
    quarter_Q1                                         0.094387
    product_name_Bolognese drobiowe                    0.094046
    rocPeriod_lag1                                     0.093254
    product_name_Paella z kurczakiem                   0.088178
    cooking_mv_or_ov                                   0.078659
    weglow_calk                                        0.077177
    product_name_Ostra wołowina z kaszą gryczaną       0.076841
    category_name_Makarony                             0.073703
    meanLastPeriod_lag1_lag7_diff                      0.070034
    product_name_Chili con carne z ryżem               0.067179
    avg_discount_count_lag1                            0.064511
    maxLastPeriod_lag1_lag7_diff                       0.063982
    is_delivery_day                                    0.060467
    product_name_Kurczak w sosie orzechowym z ryżem    0.058307
    avg_discount_mean_value_lag1                       0.051746
    pos_id_5c8ba011c667da0ef0790799                    0.051477
    company_id_5c8bade6a7d3a504da7a4b03                0.051332
    product_name_Wołowina z marchewkowym puree         0.051072
    company_id_593f7c8fb0bea3175242d9bb                0.047482
    product_name_Szpinak z fetą                        0.046300
    company_id_5be15161479c2d2a0197a927                0.045847
    company_id_5b1f8a5ecaef965005d12d82                0.043261
    pos_id_5b1b963bcaef965005d0e6b0                    0.043261
    diet_Kuchnia Orientalna                            0.042598
    diet_Kuchnia Śródziemnomorska                      0.042514
    category_name_Śniadania                           -0.046836
    diet_Zupy                                         -0.060679
    category_name_Zupy                                -0.062455
    quarter_Q3                                        -0.064771
    month                                             -0.072959
    week                                              -0.074631
    diet_nan                                          -0.081961
    partner_product                                   -0.090230
    blonnik_100                                       -0.093450
    category_name_Napoje                              -0.100115
    weglow_100                                        -0.102275
    vat                                               -0.104384
    tluszcz_nasyc_100                                 -0.107214
    energia_100                                       -0.110302
    days_since_prev_delivery                          -0.127343
    cukry_calk                                        -0.130633
    category_name_Przekąski                           -0.131299
    tluszcz_100                                       -0.133208
    cukry_100                                         -0.143850
    cooking_time_nan                                  -0.171438
    dtype: float64




```python
len(corrs[abs(corrs)>=0.04])
```




    98




```python
len(corrs[abs(corrs)>=0.01])
```




    438




```python
len(corrs[abs(corrs)>=0.025])
```




    171



### Features importance with random forrest


```python
X_small = X_val
y_small = y_val
```


```python
from sklearn.ensemble import ExtraTreesClassifier

forest = ExtraTreesClassifier(n_estimators=250, random_state=0)

forest.fit(X_small, y_small)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
```


```python
print("Feature ranking:")

ftrs = X_small.keys()
for i in range(len(ftrs)):
    print(f"{i + 1}. feature {indices[i]} ({ftrs[indices[i]]}) - importance {importances[indices[i]]}")
```

    Feature ranking:
    1. feature 64 (sum_qty) - importance 0.02857332555538195
    2. feature 19 (week) - importance 0.026891786531448355
    3. feature 68 (avg_from_paypass_lag1) - importance 0.021251139815816302
    4. feature 18 (month) - importance 0.020849031102915774
    5. feature 79 (available_products) - importance 0.01962649657615115
    6. feature 77 (days_since_prev_delivery) - importance 0.019343703637926778
    7. feature 34 (meanLastPeriod_lag1) - importance 0.01388388024562896
    8. feature 42 (sdLastPeriod_lag1) - importance 0.012889859686469315
    9. feature 36 (meanLastPeriod_lag3) - importance 0.012379493106474331
    10. feature 35 (meanLastPeriod_lag2) - importance 0.012144795825265719
    11. feature 74 (avg_transaction_discount_count_lag1) - importance 0.011964420664176374
    12. feature 45 (sdLastPeriod_lag4) - importance 0.011362275913776547
    13. feature 76 (rocPeriod_lag1) - importance 0.011260772429316009
    14. feature 72 (avg_total_base_lag1) - importance 0.011078064406763889
    15. feature 78 (sales_since_prev_delivery) - importance 0.01102854619204619
    16. feature 52 (maxLastPeriod_lag1) - importance 0.01086734642417393
    17. feature 39 (meanLastPeriod_lag6) - importance 0.01011703114308635
    18. feature 37 (meanLastPeriod_lag4) - importance 0.010082425197716745
    19. feature 48 (sdLastPeriod_lag7) - importance 0.010027416564651255
    20. feature 70 (avg_total_lag1) - importance 0.009980535322524811
    21. feature 67 (avg_from_blik_lag1) - importance 0.009749628890732442
    22. feature 40 (meanLastPeriod_lag7) - importance 0.009384454858402983
    23. feature 46 (sdLastPeriod_lag5) - importance 0.00935640202476865
    24. feature 43 (sdLastPeriod_lag2) - importance 0.008960081966136678
    25. feature 38 (meanLastPeriod_lag5) - importance 0.008934079871946119
    26. feature 44 (sdLastPeriod_lag3) - importance 0.00851327915739423
    27. feature 47 (sdLastPeriod_lag6) - importance 0.008282654119090349
    28. feature 765 (product_name_Dyniowe curry z indykiem) - importance 0.008219921750058878
    29. feature 53 (maxLastPeriod_lag7) - importance 0.007793375162582361
    30. feature 20 (qty_lag1) - importance 0.0076141888552904305
    31. feature 41 (meanLastPeriod_lag1_lag7_diff) - importance 0.006844419562389682
    32. feature 22 (qty_lag3) - importance 0.006621646838938722
    33. feature 21 (qty_lag2) - importance 0.006496959325073311
    34. feature 950 (quarter_Q3) - importance 0.006395884885886913
    35. feature 55 (diff1_lag1) - importance 0.006207801882188465
    36. feature 23 (qty_lag4) - importance 0.006196227730796649
    37. feature 941 (weekday_środa) - importance 0.005935382133512221
    38. feature 943 (weekday_piątek) - importance 0.005832676552956807
    39. feature 17 (sol_calk) - importance 0.005796830609460134
    40. feature 80 (is_delivery_day) - importance 0.005766324418789913
    41. feature 69 (avg_from_payu_lag1) - importance 0.005759344514857875
    42. feature 940 (weekday_czwartek) - importance 0.0057393264774328176
    43. feature 71 (avg_total_to_discount_lag1) - importance 0.005731499503345064
    44. feature 945 (weekday_niedziela) - importance 0.0057209880903950775
    45. feature 944 (weekday_sobota) - importance 0.005698645102650441
    46. feature 942 (weekday_wtorek) - importance 0.0056902400176049575
    47. feature 11 (bialko_calk) - importance 0.005631198875806737
    48. feature 946 (weekday_poniedziałek) - importance 0.0055644744304945365
    49. feature 895 (address_city_Warszawa) - importance 0.0053723489341210755
    50. feature 951 (quarter_Q4) - importance 0.005276736268016767
    51. feature 54 (maxLastPeriod_lag1_lag7_diff) - importance 0.005272981523912304
    52. feature 2 (bialko_100) - importance 0.005201905989302602
    53. feature 16 (cukry_100) - importance 0.005182289306039822
    54. feature 4 (cukry_calk) - importance 0.005098269344589055
    55. feature 24 (qty_lag5) - importance 0.005071681579507639
    56. feature 7 (energia_100) - importance 0.005012065702198107
    57. feature 57 (diff1_lag1_lag7_diff) - importance 0.004994605963512911
    58. feature 896 (address_city_Kraków) - importance 0.004983829969121294
    59. feature 60 (diffLagPeriod_lag1_lag7_diff) - importance 0.004977410640067965
    60. feature 948 (quarter_Q2) - importance 0.00493596161795843
    61. feature 75 (roc1_lag1) - importance 0.004879402260553727
    62. feature 949 (quarter_Q1) - importance 0.004738797760248712
    63. feature 761 (category_name_Mr Thai) - importance 0.004719674328723953
    64. feature 6 (energia_calk) - importance 0.004695606925258942
    65. feature 3 (weglow_100) - importance 0.004576027264122323
    66. feature 25 (qty_lag6) - importance 0.004573375230569302
    67. feature 63 (mean_diff1_lag1_lag7_diff) - importance 0.004551958065899159
    68. feature 12 (sol_100) - importance 0.00452266501278377
    69. feature 13 (weglow_calk) - importance 0.004453839136649479
    70. feature 81 (weight) - importance 0.004411429805393891
    71. feature 15 (tluszcz_100) - importance 0.004325528898120895
    72. feature 30 (qty_lag11) - importance 0.00426025078544975
    73. feature 29 (qty_lag10) - importance 0.004256945265776369
    74. feature 32 (qty_lag13) - importance 0.0042170500009926955
    75. feature 31 (qty_lag12) - importance 0.004086156013287876
    76. feature 26 (qty_lag7) - importance 0.003993905049443266
    77. feature 10 (tluszcz_calk) - importance 0.003982393160114528
    78. feature 8 (tluszcz_nasyc_100) - importance 0.003963481813215926
    79. feature 84 (stored_in_fridge) - importance 0.003959418115676285
    80. feature 5 (tluszcz_nasyc_calk) - importance 0.0038903316124225817
    81. feature 14 (blonnik_calk) - importance 0.003889994400089331
    82. feature 9 (blonnik_100) - importance 0.0038766889453927104
    83. feature 28 (qty_lag9) - importance 0.0038750679606908327
    84. feature 59 (diffLagPeriod_lag7) - importance 0.0038633921222672174
    85. feature 33 (qty_lag14) - importance 0.003795622387996871
    86. feature 58 (diffLagPeriod_lag1) - importance 0.003640725025777019
    87. feature 62 (mean_diff1_lag7) - importance 0.003635398282045855
    88. feature 56 (diff1_lag7) - importance 0.0036329743827119344
    89. feature 897 (address_city_Wrocław) - importance 0.0036158811890568154
    90. feature 938 (cooking_time_2-3 min.) - importance 0.0035139716186991564
    91. feature 61 (mean_diff1_lag1) - importance 0.003452858767240745
    92. feature 27 (qty_lag8) - importance 0.0033048249470104866
    93. feature 939 (cooking_time_nan) - importance 0.00304677714734007
    94. feature 754 (category_name_Dania Lunch Duże) - importance 0.0030201751602722694
    95. feature 898 (address_city_Katowice) - importance 0.002270964020773423
    96. feature 819 (product_name_Pad Thai Chicken) - importance 0.0021734419620462932
    97. feature 460 (company_id_5b6a82e76afd6509aeb6d995) - importance 0.0020807532957874976
    98. feature 83 (cooking_mv_or_ov) - importance 0.002053866538788215
    99. feature 927 (diet_Dieta Samuraja) - importance 0.0020376911800026516
    100. feature 797 (product_name_Bolognese drobiowe) - importance 0.0019923439079146815
    101. feature 818 (product_name_Sesame Beef) - importance 0.0019547206358394185
    102. feature 455 (company_id_5a587706cf5c8134b3a9891d) - importance 0.0019349242395016084
    103. feature 758 (category_name_Makarony) - importance 0.0017493533680826038
    104. feature 937 (diet_nan) - importance 0.001697395529879202
    105. feature 479 (company_id_593f7c8fb0bea3175242d9bb) - importance 0.0016514412871105608
    106. feature 461 (company_id_5bd9aff772847a312e4d8c8e) - importance 0.0016310637346062774
    107. feature 456 (company_id_5c48270855f3af637b69b13f) - importance 0.0015844873238219492
    108. feature 757 (category_name_Zupy) - importance 0.0015533134438530482
    109. feature 817 (product_name_Paella z kurczakiem) - importance 0.0014724197356310413
    110. feature 751 (category_name_Dania Lunch Małe) - importance 0.001424118734424982
    111. feature 750 (category_name_Przekąski) - importance 0.0013884061160360035
    112. feature 1 (vat) - importance 0.0013384325726450296
    113. feature 469 (company_id_59131ec0aab94617882cb2fc) - importance 0.0012674507909285983
    114. feature 459 (company_id_5c924f69b6cb840dcac430fe) - importance 0.0012508395277765038
    115. feature 462 (company_id_5b8d32b990b4de59396b6ba0) - importance 0.0012372495073579954
    116. feature 458 (company_id_5b503c4a9586df16bbb6e07a) - importance 0.00116011264922529
    117. feature 464 (company_id_5b1f8a17caef965005d12d7e) - importance 0.001151717422973313
    118. feature 484 (company_id_5c1a0baa2d1e1a498b466170) - importance 0.0011515579324752022
    119. feature 931 (diet_Zupy) - importance 0.0011382399674390936
    120. feature 468 (company_id_5c48274055f3af637b69b147) - importance 0.0011007271579323943
    121. feature 515 (company_id_5b1f8a5ecaef965005d12d82) - importance 0.0010881528732775735
    122. feature 486 (company_id_5ca360c14bb14a379ddb2357) - importance 0.0010815421296084386
    123. feature 457 (company_id_5b9fa5e201f82e03b412bcff) - importance 0.0010739342047605246
    124. feature 123 (pos_id_5b6800b8185d852e40c61af3) - importance 0.0010638563722070555
    125. feature 234 (pos_id_5c73d392531c6b7fe27f272e) - importance 0.0010563885616841652
    126. feature 463 (company_id_59562d9de9b1ed3755974024) - importance 0.0010436575591498908
    127. feature 478 (company_id_5ba22bda01f82e03b413792c) - importance 0.0010220827578908585
    128. feature 89 (pos_id_5b1b963bcaef965005d0e6b0) - importance 0.0010179658618508492
    129. feature 163 (pos_id_5b3cdde50663ab48e33669e3) - importance 0.001012876435890153
    130. feature 587 (company_id_5c8bade6a7d3a504da7a4b03) - importance 0.0010079938793451807
    131. feature 509 (company_id_5ca35eb77458b6267b435db5) - importance 0.0009987214170537682
    132. feature 171 (pos_id_5ba10d0e01f82e03b41326f4) - importance 0.0009982539024024314
    133. feature 929 (diet_Dieta Paleo) - importance 0.0009921451951223391
    134. feature 473 (company_id_596ca32bcf81a34d0c22e63c) - importance 0.0009793858304897043
    135. feature 238 (pos_id_5c8ba011c667da0ef0790799) - importance 0.0009751058942320995
    136. feature 87 (pos_id_5b61699404852c4704a26f28) - importance 0.0009662565248547363
    137. feature 465 (company_id_5a572369cf5c8134b3a98692) - importance 0.0009574200300150725
    138. feature 82 (volume) - importance 0.0009448015416229066
    139. feature 490 (company_id_5c4ac5acfd89245e80d4aa35) - importance 0.0009333132245570077
    140. feature 492 (company_id_5c73d670dd7f597c83100c64) - importance 0.0009292264315947465
    141. feature 766 (product_name_Ostra wołowina z kaszą gryczaną) - importance 0.0009187161893844596
    142. feature 280 (pos_id_5b6804da185d852e40c61be5) - importance 0.0009078976990310149
    143. feature 636 (company_id_5be15161479c2d2a0197a927) - importance 0.0009076781397029991
    144. feature 853 (product_name_Chili con carne z ryżem) - importance 0.0009076697247495647
    145. feature 899 (address_city_Skawina) - importance 0.0008914907262641715
    146. feature 756 (category_name_Śniadania) - importance 0.0008793652040449376
    147. feature 476 (company_id_596ca2e8cf81a34d0c22e638) - importance 0.0008740793244587025
    148. feature 186 (pos_id_5bcd854ff4a41472d46924ef) - importance 0.0008677278771155882
    149. feature 933 (diet_Kuchnia Latynoska) - importance 0.000862622662089938
    150. feature 146 (pos_id_5ba0ea5701f82e03b4131ace) - importance 0.0008608793038935196
    151. feature 474 (company_id_5ca33520b532a9058ac830ad) - importance 0.0008606801862429298
    152. feature 560 (company_id_5c35f7fe9557cb13926c53c7) - importance 0.0008560861147849856
    153. feature 903 (address_city_Wieliczka) - importance 0.0008234078153941773
    154. feature 503 (company_id_5bacd53826a9cb3d33cf4c42) - importance 0.0008137636289639593
    155. feature 547 (company_id_5a1ec2719eeb473d7d6cb03e) - importance 0.0008071230561665807
    156. feature 481 (company_id_5c41cd3cb578ec3384e5208d) - importance 0.0008069733854613894
    157. feature 276 (pos_id_5cab3c401b1e6d46ad6bf4e0) - importance 0.0008046845888379772
    158. feature 109 (pos_id_5b1bb68ecaef965005d0e938) - importance 0.0008026510629045201
    159. feature 387 (pos_id_5ba0f2fb01f82e03b4131e2c) - importance 0.0008011597443041256
    160. feature 491 (company_id_5b2b3a800663ab48e334d3e5) - importance 0.0007927511643276602
    161. feature 860 (product_name_Pieczone placki ziemniaczane z gulaszem drobiowym) - importance 0.0007851001578607187
    162. feature 128 (pos_id_5a563327cf5c8134b3a9860d) - importance 0.0007827840240608972
    163. feature 232 (pos_id_5c616cf7c88bae77186d5942) - importance 0.0007794036526244529
    164. feature 200 (pos_id_5b7e8afd003ec47c2d8b6d15) - importance 0.0007790276788789431
    165. feature 566 (company_id_5ce296b9d88a9a3711818bab) - importance 0.0007787886274302742
    166. feature 472 (company_id_5b5eba7501925031d9291668) - importance 0.0007769717854612156
    167. feature 229 (pos_id_5c4874b855f3af637b69cacb) - importance 0.000775533247565177
    168. feature 584 (company_id_5c612e45cfa3a62fcbee67ae) - importance 0.000774467571669934
    169. feature 101 (pos_id_5b62f89c185d852e40c52831) - importance 0.00077403488437539
    170. feature 767 (product_name_Wołowina z marchewkowym puree) - importance 0.0007696639976752941
    171. feature 475 (company_id_5c98d473bedc6e4786a40398) - importance 0.0007679422939776648
    172. feature 489 (company_id_5b3484bb0663ab48e3359f48) - importance 0.0007662469588422316
    173. feature 755 (category_name_Napoje) - importance 0.0007623183847005773
    174. feature 85 (pos_id_59fc5325c94b722506678bd1) - importance 0.0007607720392120551
    175. feature 236 (pos_id_5b616ea804852c4704a2705e) - importance 0.0007601347430994657
    176. feature 142 (pos_id_5b86679da9fd2f7143f74b1f) - importance 0.0007578407288247861
    177. feature 181 (pos_id_5ba0fc4a01f82e03b4132145) - importance 0.0007544938688842502
    178. feature 553 (company_id_5ca335b63976942657e402f7) - importance 0.0007528165286652535
    179. feature 66 (avg_discount_count_lag1) - importance 0.0007490824213048722
    180. feature 520 (company_id_5be150f71f7e9e23de34020a) - importance 0.0007458965629334472
    181. feature 471 (company_id_5b1f894dcaef965005d12d63) - importance 0.0007406110753860156
    182. feature 140 (pos_id_5b8688de2dd588270f00a1b2) - importance 0.0007187127137855957
    183. feature 928 (diet_--) - importance 0.0007135876975935525
    184. feature 528 (company_id_5c77d00e7b6c863a5d51dd37) - importance 0.0007124933399362388
    185. feature 102 (pos_id_5971be7678bf81352f7d85e6) - importance 0.0007105017508613338
    186. feature 209 (pos_id_5c1a5563e343b12380673fbf) - importance 0.0007089093270311629
    187. feature 470 (company_id_5ad8a0091da94a4f28d11ec0) - importance 0.0007010849201265268
    188. feature 94 (pos_id_595635fe2e5db6320199ecbc) - importance 0.0006992640974015993
    189. feature 98 (pos_id_5971d44978bf81352f7d8689) - importance 0.0006960045089100828
    190. feature 496 (company_id_59ae4ed699054725e9f8eaf2) - importance 0.0006925448760019366
    191. feature 467 (company_id_5af3e4736e089c600a14dd86) - importance 0.0006777724273245129
    192. feature 96 (pos_id_5b4713ffee2a423f39da6d0b) - importance 0.0006761767055455437
    193. feature 173 (pos_id_5b8cf928a3dd523eba9239b6) - importance 0.000675256809028357
    194. feature 577 (company_id_5b5ebb9901925031d929169b) - importance 0.0006744099150728368
    195. feature 524 (company_id_5a6b0634a0899f5ca2f7d293) - importance 0.000673798268596223
    196. feature 191 (pos_id_5bc848ffa6c5c61a73d3979f) - importance 0.0006716820485467871
    197. feature 507 (company_id_5c46edbc0a0b153373b7e49c) - importance 0.0006716038491761006
    198. feature 112 (pos_id_5a9eb80a9d8b982a7ebc8ca0) - importance 0.0006705200651988549
    199. feature 255 (pos_id_5c98fb91dc248c707e85b441) - importance 0.0006682191141834409
    200. feature 505 (company_id_5beada3478729f2c8afeeb90) - importance 0.0006630303846395306
    201. feature 119 (pos_id_5b4c9ca3ee2a423f39db9e40) - importance 0.0006585980462066033
    202. feature 277 (pos_id_5c1a3b6a2d1e1a498b466f8f) - importance 0.0006535914012298022
    203. feature 283 (pos_id_5ca35d9a7458b6267b435d7b) - importance 0.000649602404926245
    204. feature 267 (pos_id_5ca35bbbe1cd1a1ebce89bbd) - importance 0.0006482185777221537
    205. feature 170 (pos_id_5b472211ee2a423f39da70a6) - importance 0.0006465035930703719
    206. feature 92 (pos_id_5a5c7e0acf5c8134b3a98c8f) - importance 0.0006342660248882233
    207. feature 483 (company_id_5b07a4d1a73a156ec5356ad5) - importance 0.0006333656070613629
    208. feature 477 (company_id_5b86a06b2dd588270f00a981) - importance 0.0006297241108022952
    209. feature 932 (diet_Kuchnia Słowiańska) - importance 0.0006236152581097548
    210. feature 480 (company_id_5b63f0a9185d852e40c5588f) - importance 0.0006178310359730762
    211. feature 187 (pos_id_5bcd9ec3f4a41472d4692d8b) - importance 0.0006174895827264754
    212. feature 222 (pos_id_5c4af35cfd89245e80d4b7f9) - importance 0.0006144081069201499
    213. feature 930 (diet_Kuchnia Śródziemnomorska) - importance 0.000613887609585561
    214. feature 131 (pos_id_5b471f19ee2a423f39da6fcd) - importance 0.0006109676651512917
    215. feature 482 (company_id_5c41cd89b578ec3384e5209e) - importance 0.0006062429272878598
    216. feature 124 (pos_id_5b47069dee2a423f39da69f6) - importance 0.0006052778646906184
    217. feature 534 (company_id_5be56bafc2c8427b0f4f63e8) - importance 0.0006038060564439985
    218. feature 135 (pos_id_5a564812cf5c8134b3a98644) - importance 0.0005979779566657413
    219. feature 175 (pos_id_5b62f298185d852e40c526b1) - importance 0.000595487860831186
    220. feature 516 (company_id_5bc71fe2a6c5c61a73d3448d) - importance 0.0005924888589597557
    221. feature 158 (pos_id_5b339c340663ab48e3358c9b) - importance 0.0005899638105283481
    222. feature 494 (company_id_5bd9ae7f28a8882be911293e) - importance 0.0005891865875554803
    223. feature 902 (address_city_Niepołomice) - importance 0.000588654423243923
    224. feature 116 (pos_id_5b5b101cec42762a8558ba6d) - importance 0.0005860386606304074
    225. feature 95 (pos_id_5b1a4205caef965005d0cd81) - importance 0.000584572133781768
    226. feature 529 (company_id_5be1511e1f7e9e23de34020c) - importance 0.0005836345815060092
    227. feature 0 (partner_product) - importance 0.0005833985568850675
    228. feature 514 (company_id_5c7fb0d36b25e24bf5028f65) - importance 0.0005814223605938723
    229. feature 122 (pos_id_5afc651135348125c0a0cde1) - importance 0.0005799677898485562
    230. feature 526 (company_id_5b5ebc2b01925031d92916bb) - importance 0.0005798400295137574
    231. feature 485 (company_id_596ca35ccf81a34d0c22e63d) - importance 0.0005798374929632822
    232. feature 753 (category_name_Pan Pomidor - Pierogi) - importance 0.00057963258418515
    233. feature 499 (company_id_5c1a09a5e343b12380672a58) - importance 0.0005794822384162152
    234. feature 149 (pos_id_5b8d0218a3dd523eba923c25) - importance 0.000573764869403265
    235. feature 113 (pos_id_5a563f0dcf5c8134b3a98629) - importance 0.0005736413478579501
    236. feature 394 (pos_id_5bd1ca9ada98d463e0fec34c) - importance 0.0005721833344505438
    237. feature 104 (pos_id_5a1ebfd9753e9f3591456bc3) - importance 0.0005707098623639791
    238. feature 901 (address_city_Gliwice) - importance 0.0005667806378765381
    239. feature 820 (product_name_Kurczak w sosie orzechowym z ryżem) - importance 0.0005630567744633372
    240. feature 264 (pos_id_5c94bc783a11ee42d6540c3d) - importance 0.0005625686316775378
    241. feature 162 (pos_id_5b7d3a469d42836761a0b1b4) - importance 0.0005593725841134432
    242. feature 226 (pos_id_5c4483480a0b153373b77017) - importance 0.0005586527502409212
    243. feature 904 (address_city_Kobierzyce, Bielany Wrocławskie) - importance 0.0005541861868080536
    244. feature 90 (pos_id_596e2a5371059a6bc70a818b) - importance 0.0005533533516356779
    245. feature 498 (company_id_5c8f8138a7d3a504da7b20fa) - importance 0.0005532642552262175
    246. feature 326 (pos_id_5ce29992d88a9a3711818c7e) - importance 0.0005489236342891769
    247. feature 934 (diet_Kuchnia Orientalna) - importance 0.0005480152662743727
    248. feature 300 (pos_id_5c8f9f24c667da0ef079ec1f) - importance 0.0005461306464154919
    249. feature 207 (pos_id_5b4893ddee2a423f39dacbf6) - importance 0.0005459067285842981
    250. feature 221 (pos_id_5c4af17d55f3af637b6a5758) - importance 0.0005432234010099937
    251. feature 576 (company_id_5c41cd650a0b153373b6fc6f) - importance 0.000541989993973158
    252. feature 603 (company_id_5bd01aa651478a069f6c3552) - importance 0.00054189175722433
    253. feature 257 (pos_id_5c8123261a02182c4627fcc3) - importance 0.0005416062994200163
    254. feature 218 (pos_id_5c40888d0a0b153373b6bca4) - importance 0.0005400233182335627
    255. feature 568 (company_id_5c18cf642d1e1a498b4622a0) - importance 0.0005374489960802185
    256. feature 211 (pos_id_5c18eb2be343b1238066f58d) - importance 0.0005359295844925473
    257. feature 562 (company_id_5bd9ae0a28a8882be9112932) - importance 0.0005351679109290037
    258. feature 108 (pos_id_5afc757b35348125c0a0cf3d) - importance 0.00053476523036831
    259. feature 205 (pos_id_5bd1bcffda98d463e0febe54) - importance 0.0005331799327616665
    260. feature 487 (company_id_5b3492210663ab48e335a085) - importance 0.00053104592782462
    261. feature 219 (pos_id_5c44a4080a0b153373b77572) - importance 0.0005301981512466126
    262. feature 768 (product_name_Risotto z indykiem i cukinią) - importance 0.0005300818809433734
    263. feature 786 (product_name_Pikantny ryż jaśminowy smażony z jajkiem) - importance 0.000529836559058675
    264. feature 91 (pos_id_5b1bad72caef965005d0e891) - importance 0.0005261376443991504
    265. feature 237 (pos_id_5c46ccfab578ec3384e601b3) - importance 0.0005259356333274972
    266. feature 316 (pos_id_5c62bc241a466d77f43a47ba) - importance 0.0005254759262604042
    267. feature 233 (pos_id_5c4809d80a0b153373b82090) - importance 0.0005249065817149605
    268. feature 799 (product_name_Szpinak z fetą) - importance 0.0005246769591866148
    269. feature 851 (product_name_Curry z kurczakiem) - importance 0.0005240025891462053
    270. feature 466 (company_id_5956225fecb63b62774330dc) - importance 0.0005232053293205615
    271. feature 523 (company_id_5b3494c40663ab48e335a0bd) - importance 0.0005206565241343847
    272. feature 538 (company_id_5c7fa70c6b25e24bf5028742) - importance 0.0005192678689512774
    273. feature 111 (pos_id_5b755eed9c7c767ae0573f0d) - importance 0.0005155443832774078
    274. feature 501 (company_id_59afaaf999054725e9f8ec45) - importance 0.0005153483499310456
    275. feature 574 (company_id_5c63cb461a466d77f43a8198) - importance 0.0005141281869167404
    276. feature 115 (pos_id_5a6a0b87a0899f5ca2f7c6db) - importance 0.0005117454806016507
    277. feature 235 (pos_id_5c628e9d1a466d77f43a36a9) - importance 0.0005110121795084987
    278. feature 803 (product_name_Teriyaki z kurczakiem) - importance 0.0005091079371176055
    279. feature 294 (pos_id_5cb99db2df68013fb09db250) - importance 0.0005070803696852557
    280. feature 88 (pos_id_5b3381160663ab48e335895f) - importance 0.0005068535199661077
    281. feature 770 (product_name_Pulpety drobiowe w sosie pomidorowym) - importance 0.0005065990755327681
    282. feature 488 (company_id_5ad89fd91da94a4f28d11ebe) - importance 0.0005060592810829979
    283. feature 398 (pos_id_5d286af097a5c32879fa7d3c) - importance 0.0005055029976641275
    284. feature 497 (company_id_5b39bb530663ab48e336173e) - importance 0.000503693024366621
    285. feature 103 (pos_id_5a98feef9d8b982a7ebc8455) - importance 0.0005020757813734192
    286. feature 650 (company_id_599be8172e13ef5e3d56d050) - importance 0.0005013220380464791
    287. feature 759 (category_name_Pan Pomidor - Zupy) - importance 0.0005012836298341557
    288. feature 597 (company_id_5d15ece24152364df186f629) - importance 0.0004999020855426141
    289. feature 97 (pos_id_5aa69a76cb73bd163e8b2439) - importance 0.0004981525347283191
    290. feature 274 (pos_id_5b8511abcfd86b6b29a1348f) - importance 0.0004980704844108828
    291. feature 208 (pos_id_5c122e601462a123065f3a7d) - importance 0.0004970124447240397
    292. feature 508 (company_id_5926d84126456576dfcc3e68) - importance 0.0004964244582280138
    293. feature 337 (pos_id_5cdbda11ccee817a5ba6f84c) - importance 0.0004936980620176533
    294. feature 297 (pos_id_5ba2370901f82e03b4137e22) - importance 0.0004928401869916162
    295. feature 517 (company_id_59562c16e9b1ed375597401e) - importance 0.0004914343201285732
    296. feature 552 (company_id_5b5ead5701925031d9291430) - importance 0.0004906811087614942
    297. feature 169 (pos_id_5b5b1b910b5f85308e75c75e) - importance 0.00048606489219121064
    298. feature 129 (pos_id_5b07aa5ca73a156ec5356b53) - importance 0.0004845521228438972
    299. feature 194 (pos_id_5bc9b10ba6c5c61a73d3fefa) - importance 0.000484039722360706
    300. feature 275 (pos_id_5c94b4c01032d642b6347b7a) - importance 0.00048399068014727737
    301. feature 99 (pos_id_5b4cab19ee2a423f39dba28f) - importance 0.0004826115293006235
    302. feature 605 (company_id_5ba22beb01f82e03b413792d) - importance 0.00048230285355838005
    303. feature 262 (pos_id_5c98f89ea0d5866f86eb95c6) - importance 0.0004818196018838365
    304. feature 197 (pos_id_5b8fa0a88667b879c9f85ba4) - importance 0.000479119467043986
    305. feature 110 (pos_id_593e9e6ab0bea3175242d99c) - importance 0.0004783770471447278
    306. feature 250 (pos_id_5c7e7309d46a9a483518df08) - importance 0.00047784769719425057
    307. feature 161 (pos_id_5940f440b0bea3175242db32) - importance 0.00047746880284936185
    308. feature 308 (pos_id_5c123a1b1462a123065f3d20) - importance 0.0004754902684328744
    309. feature 612 (company_id_5c13b386e343b12380661541) - importance 0.0004751366665050666
    310. feature 662 (company_id_5c482765fd89245e80d411c6) - importance 0.0004741738629590762
    311. feature 506 (company_id_5c7e89bdd46a9a483518e5cb) - importance 0.00047397674458198144
    312. feature 537 (company_id_5ba35eabe39f154c743ad734) - importance 0.0004738641270127031
    313. feature 914 (address_city_Podłęże, Kraków) - importance 0.0004730946138312786
    314. feature 282 (pos_id_596e421371059a6bc70a81c5) - importance 0.0004723983530450329
    315. feature 144 (pos_id_5b71452cd8f80d6ebf31d602) - importance 0.0004713060287712475
    316. feature 493 (company_id_5ce2967ac828ba60d809328e) - importance 0.0004703382234712881
    317. feature 369 (pos_id_5926e40426456576dfcc3ea5) - importance 0.00046971148046747813
    318. feature 160 (pos_id_5b7c20d0986a9a41c7dc37bc) - importance 0.0004632449026262826
    319. feature 224 (pos_id_5c446051b578ec3384e58f47) - importance 0.0004617351946665268
    320. feature 629 (company_id_59562c8de9b1ed375597401f) - importance 0.0004616547204142871
    321. feature 286 (pos_id_5c9e263a80237d156eb01343) - importance 0.0004609978572753168
    322. feature 141 (pos_id_59a97444ffa5f308210dfea2) - importance 0.000459016238238709
    323. feature 195 (pos_id_5bc8733fa6c5c61a73d3a855) - importance 0.00045744522546663837
    324. feature 139 (pos_id_59352acd8ae0a56c49a319a4) - importance 0.00045725719981747895
    325. feature 796 (product_name_Leczo węgierskie z babką jaglaną) - importance 0.0004522043557693068
    326. feature 159 (pos_id_5b911f5fcd07f81d7e1f8d3c) - importance 0.00045186388261204253
    327. feature 65 (avg_discount_mean_value_lag1) - importance 0.0004495505140802291
    328. feature 253 (pos_id_5c73dd37531c6b7fe27f2b03) - importance 0.00044944030984374653
    329. feature 245 (pos_id_5c641ee5e367f5183f058936) - importance 0.00044832472310363806
    330. feature 269 (pos_id_5c94cd731032d642b6348430) - importance 0.000448013726096731
    331. feature 198 (pos_id_59aff3feffa5f308210e04d9) - importance 0.00044681033046296636
    332. feature 382 (pos_id_5c614d5fc88bae77186d4c34) - importance 0.000443529095501863
    333. feature 580 (company_id_5c4ac57b55f3af637b6a4a61) - importance 0.00044211639459249853
    334. feature 202 (pos_id_5b8d08a290b4de59396b5c5e) - importance 0.00044075303798267994
    335. feature 145 (pos_id_5b471872ee2a423f39da6e1b) - importance 0.0004406436358738455
    336. feature 106 (pos_id_59703d4771059a6bc70a854d) - importance 0.00044063301383961284
    337. feature 134 (pos_id_5b1677753315661b2d9da43b) - importance 0.00044028941449483067
    338. feature 320 (pos_id_5cc04a932c15f345ede8d44e) - importance 0.0004400520932944424
    339. feature 519 (company_id_5b5eb00901925031d9291495) - importance 0.0004396220613123409
    340. feature 291 (pos_id_5caf375e2c9b633ae260d078) - importance 0.0004372853113875753
    341. feature 227 (pos_id_5c4488940a0b153373b770df) - importance 0.00043685434727251346
    342. feature 594 (company_id_5b9fa5a001f82e03b412bcea) - importance 0.0004366516106459633
    343. feature 201 (pos_id_5b8d139390b4de59396b60ac) - importance 0.00043506051603500994
    344. feature 590 (company_id_5c6687c46961290fe7419231) - importance 0.00043470107009945544
    345. feature 118 (pos_id_5a9a56b19d8b982a7ebc8692) - importance 0.0004346478709082316
    346. feature 266 (pos_id_5b8d2f3190b4de59396b6a83) - importance 0.00043432158575461456
    347. feature 565 (company_id_5bead07c78729f2c8afee97a) - importance 0.000434021072468688
    348. feature 564 (company_id_5b63f0de185d852e40c55899) - importance 0.00043278867211778464
    349. feature 260 (pos_id_5c7e6e9a54a4233b10f34b8f) - importance 0.0004311516080510864
    350. feature 582 (company_id_5c41d9e00a0b153373b6ff77) - importance 0.00043001478793546605
    351. feature 911 (address_city_Waszawa) - importance 0.00042573887662662
    352. feature 203 (pos_id_5bc862d5a6c5c61a73d3a164) - importance 0.0004249740402570408
    353. feature 512 (company_id_5bd9b07c28a8882be91129b1) - importance 0.0004238918966254033
    354. feature 230 (pos_id_5c49c3fc55f3af637b6a1716) - importance 0.00042307405456483237
    355. feature 166 (pos_id_5b97ac06eb573f399702e383) - importance 0.0004229752886982247
    356. feature 559 (company_id_5bc886f6a6c5c61a73d3af80) - importance 0.00042259105962759506
    357. feature 752 (category_name_Sałatki) - importance 0.0004222406085731606
    358. feature 248 (pos_id_5c641afa4dc1b258d9423850) - importance 0.00041984298591577634
    359. feature 206 (pos_id_5bc85ef9a6c5c61a73d39f82) - importance 0.00041951007775497363
    360. feature 367 (pos_id_5b7d36e69d42836761a0b0a4) - importance 0.00041875865745738165
    361. feature 575 (company_id_5c612de4cfa3a62fcbee67a6) - importance 0.0004183031464870716
    362. feature 510 (company_id_5ca1b024e2258413d5bcd30e) - importance 0.0004175083450694718
    363. feature 539 (company_id_5cc01f99def57b4350f57e6f) - importance 0.0004166401457890567
    364. feature 541 (company_id_5b96678e33246312580716d9) - importance 0.00041584625199843437
    365. feature 167 (pos_id_5afc6a7035348125c0a0ce58) - importance 0.00041483926982454836
    366. feature 272 (pos_id_5ca219adb6db8a1a6acbb87c) - importance 0.000414159156634103
    367. feature 596 (company_id_5c7e86a954a4233b10f354b1) - importance 0.00041381451471734446
    368. feature 592 (company_id_5c6687ab6961290fe741922e) - importance 0.000410468747751548
    369. feature 900 (address_city_11) - importance 0.0004100587523591135
    370. feature 778 (product_name_Zielone curry wegańskie) - importance 0.00040852649307886944
    371. feature 217 (pos_id_5c2c7eed9c2c3d11ee5d5ac0) - importance 0.0004074748379950779
    372. feature 314 (pos_id_597044a771059a6bc70a8580) - importance 0.00040500124746132497
    373. feature 284 (pos_id_5ca1f6f71b2b36155c425c10) - importance 0.0004043848091099336
    374. feature 100 (pos_id_5a56415fcf5c8134b3a9862e) - importance 0.0004034598478364388
    375. feature 223 (pos_id_5c4497980a0b153373b77392) - importance 0.0004033595730754719
    376. feature 215 (pos_id_5c17aea22d1e1a498b45ef81) - importance 0.0004019324433007641
    377. feature 675 (company_id_596ca3cfcf81a34d0c22e642) - importance 0.000401276219208964
    378. feature 598 (company_id_5ca335403976942657e40240) - importance 0.00039923621181724915
    379. feature 644 (company_id_5ba4ef5cd0444377df490c4f) - importance 0.00039854529912742106
    380. feature 256 (pos_id_5c8bae66a7d3a504da7a4b6b) - importance 0.0003977232054316396
    381. feature 188 (pos_id_5bd1b751da98d463e0febc0d) - importance 0.00039769763096054534
    382. feature 254 (pos_id_5c8fa44aa7d3a504da7b2d7f) - importance 0.0003971195696684071
    383. feature 378 (pos_id_5b339f100663ab48e3358cf4) - importance 0.00039691184075526153
    384. feature 246 (pos_id_5c7e6bca54a4233b10f34998) - importance 0.00039616536441992444
    385. feature 550 (company_id_59ce1dcbc6f19b5d177c20b5) - importance 0.00039541690693454683
    386. feature 613 (company_id_5a79b5753672b770f8f34cb1) - importance 0.00039512879251658585
    387. feature 680 (company_id_5d15ee3e4152364df186f6e3) - importance 0.0003934243884571486
    388. feature 285 (pos_id_5ca21800b6db8a1a6acbb83e) - importance 0.0003931728506586982
    389. feature 771 (product_name_Białe kaszotto z suszonymi pomidorami) - importance 0.00039198525513961356
    390. feature 527 (company_id_5bbdd41f15c00937a08115b9) - importance 0.00039125909401413884
    391. feature 906 (address_city_Balice) - importance 0.00039055993305744664
    392. feature 502 (company_id_5aaa3eadcb73bd163e8b3ffc) - importance 0.00038994763136554413
    393. feature 371 (pos_id_5ba89667d0444377df4a192b) - importance 0.00038853708284444934
    394. feature 132 (pos_id_5a4cc784d8510f474645af2c) - importance 0.00038569716165353935
    395. feature 189 (pos_id_5bc88e3ea6c5c61a73d3b1c3) - importance 0.00038500427313316766
    396. feature 335 (pos_id_5c1229cde343b1238065c80d) - importance 0.00038339335060406563
    397. feature 531 (company_id_5a97d9be9d8b982a7ebc81c7) - importance 0.0003828672144091702
    398. feature 769 (product_name_Szaszłyk z ryżem w kurkumie) - importance 0.0003817907786856541
    399. feature 120 (pos_id_59561e972e5db6320199ec77) - importance 0.0003814529390814259
    400. feature 304 (pos_id_5baa1882a27a6774b7fa8ea2) - importance 0.0003807293344501817
    401. feature 182 (pos_id_5b8fa4158667b879c9f85c8e) - importance 0.00038033491987131315
    402. feature 548 (company_id_5be4424774546727b8c6058c) - importance 0.000377099280370929
    403. feature 183 (pos_id_5b8fc02125c26a15d6677eb6) - importance 0.000376330167737554
    404. feature 610 (company_id_5baa1b2fa27a6774b7fa8fa2) - importance 0.0003745022816074335
    405. feature 525 (company_id_593f7cddb0bea3175242d9bd) - importance 0.0003743932994728487
    406. feature 148 (pos_id_5af9739d6e089c600a154256) - importance 0.00037414215292456716
    407. feature 620 (company_id_5c13adc8e343b123806613c5) - importance 0.0003733820776201455
    408. feature 270 (pos_id_5c791f287b6c863a5d5228d1) - importance 0.0003723307849490861
    409. feature 172 (pos_id_5a20320e753e9f3591456ed2) - importance 0.0003707018620799621
    410. feature 772 (product_name_Indyk pieczony z warzywami korzeniowymi) - importance 0.00037047744430110163
    411. feature 114 (pos_id_5b1e851ecaef965005d11ba7) - importance 0.00037037692370389436
    412. feature 330 (pos_id_5b471c05ee2a423f39da6f06) - importance 0.0003687035092687187
    413. feature 137 (pos_id_5b59bcdf76628d7b96cdb492) - importance 0.0003686523773640505
    414. feature 545 (company_id_5b5ea8ab01925031d9291367) - importance 0.00036724028014063363
    415. feature 661 (company_id_5d15ed715376d84d27e45696) - importance 0.0003661569707496198
    416. feature 495 (company_id_593f7d34b0bea3175242d9bf) - importance 0.0003660966815204348
    417. feature 125 (pos_id_5b2b8fa60663ab48e334dd2b) - importance 0.0003660449799967492
    418. feature 121 (pos_id_595616d92e5db6320199ec48) - importance 0.00036507514497666205
    419. feature 154 (pos_id_596e44ac71059a6bc70a81d4) - importance 0.0003643296669971522
    420. feature 535 (company_id_5b92639e6c8c383e7c8045e9) - importance 0.0003637150984364379
    421. feature 558 (company_id_5c6e8878531c6b7fe27e08c8) - importance 0.00036352215967030776
    422. feature 563 (company_id_59562ce3e9b1ed3755974020) - importance 0.0003623128939852152
    423. feature 153 (pos_id_5b471650ee2a423f39da6d8c) - importance 0.0003614513332762425
    424. feature 847 (product_name_Sweet&Sour Chicken) - importance 0.0003614444201547139
    425. feature 549 (company_id_5c13adaae343b123806613c2) - importance 0.0003613217768680793
    426. feature 190 (pos_id_5bc9ab49a6c5c61a73d3fc2b) - importance 0.00036079599020024755
    427. feature 176 (pos_id_5b4ba51fee2a423f39db6741) - importance 0.00035847527088164966
    428. feature 557 (company_id_5be44236aac41f2436a1b868) - importance 0.0003584225665626413
    429. feature 544 (company_id_5b39bb970663ab48e3361743) - importance 0.00035829112305181483
    430. feature 379 (pos_id_5cffa2f26426b30f294421c6) - importance 0.0003582351757438994
    431. feature 332 (pos_id_5bd1c816da98d463e0fec262) - importance 0.0003575096499084608
    432. feature 780 (product_name_Dahl z garam masala) - importance 0.0003550730079271907
    433. feature 595 (company_id_5c7cd09d3ae6e53aff15483d) - importance 0.0003549126453100917
    434. feature 136 (pos_id_5b291edb0663ab48e334ac12) - importance 0.00035358933335293176
    435. feature 329 (pos_id_5b62f57f185d852e40c5276d) - importance 0.0003530503917817329
    436. feature 349 (pos_id_5cdbd6fe9083f77a44770070) - importance 0.0003530127899795438
    437. feature 156 (pos_id_5b3733940663ab48e335e294) - importance 0.00035290280816997033
    438. feature 543 (company_id_5b2b36340663ab48e334d39d) - importance 0.00035011317978581003
    439. feature 518 (company_id_596ca515cf81a34d0c22e645) - importance 0.0003496796931235381
    440. feature 268 (pos_id_5c8bb34dc667da0ef0790d7e) - importance 0.0003496504177792293
    441. feature 168 (pos_id_5a37f93f753e9f3591458dd7) - importance 0.0003480368532916957
    442. feature 247 (pos_id_5c7e696354a4233b10f347d1) - importance 0.00034781695193924063
    443. feature 907 (address_city_Wroclaw) - importance 0.0003473585219664588
    444. feature 383 (pos_id_5d0763a7f88ba26d014f773b) - importance 0.000345859467095691
    445. feature 615 (company_id_5b9262956c8c383e7c8045bf) - importance 0.0003449667969982439
    446. feature 569 (company_id_5bbdc6a015c00937a0810fc7) - importance 0.00034474639112748267
    447. feature 164 (pos_id_5a44d792f9cdb15f7f7d9d49) - importance 0.0003444985871238595
    448. feature 540 (company_id_5b35c7a90663ab48e335be72) - importance 0.0003443379869931808
    449. feature 157 (pos_id_5b713f4fd8f80d6ebf31d49f) - importance 0.00034343482345767335
    450. feature 177 (pos_id_5ba0d8dd01f82e03b41313a0) - importance 0.0003426998907386655
    451. feature 551 (company_id_5bd018b451478a069f6c34fa) - importance 0.0003411424004017975
    452. feature 837 (product_name_Gnocchi primavera) - importance 0.0003409462181010295
    453. feature 289 (pos_id_5c94d70e1032d642b63487c2) - importance 0.00034045601329096197
    454. feature 601 (company_id_5c8f6c59c667da0ef079d9fc) - importance 0.00033860126933185863
    455. feature 356 (pos_id_5cc040375e70463ad9d520dd) - importance 0.0003384959767662926
    456. feature 573 (company_id_5ce29739d88a9a3711818bbb) - importance 0.0003380684056685266
    457. feature 632 (company_id_5b3f02a90663ab48e3369b5b) - importance 0.0003363647371337268
    458. feature 530 (company_id_5b1f892ecaef965005d12d60) - importance 0.00033516969176518367
    459. feature 204 (pos_id_5bd1c54dda98d463e0fec169) - importance 0.0003328744818888643
    460. feature 258 (pos_id_5c78f63e3ae6e53aff147ec5) - importance 0.0003328444582403113
    461. feature 348 (pos_id_5cd5659973d91d192aba5ec6) - importance 0.00033121128363279546
    462. feature 126 (pos_id_5b16964d3315661b2d9da7b6) - importance 0.0003293138154857209
    463. feature 324 (pos_id_5cdbc463ccee817a5ba6f35d) - importance 0.00032908706020373977
    464. feature 416 (pos_id_5c18d8212d1e1a498b462592) - importance 0.00032794060246927443
    465. feature 401 (pos_id_5d35a48592520d08aefbc8e7) - importance 0.00032729473170630725
    466. feature 916 (address_city_Bytom) - importance 0.0003265600429882867
    467. feature 251 (pos_id_5c8bab7ec667da0ef0790b0d) - importance 0.0003256602076194876
    468. feature 646 (company_id_5cc01cdd8ed57342ff70efd5) - importance 0.0003249190800576862
    469. feature 806 (product_name_Tajska z curry i kolendrą) - importance 0.0003239256562536887
    470. feature 265 (pos_id_5c8b9a6ec667da0ef07905e4) - importance 0.0003236639674547623
    471. feature 599 (company_id_5c79053c7b6c863a5d521fdc) - importance 0.000323364234601245
    472. feature 199 (pos_id_5bcd9700f4a41472d4692a8b) - importance 0.0003225239075958334
    473. feature 395 (pos_id_5b488274ee2a423f39dac575) - importance 0.0003222094990181769
    474. feature 811 (product_name_Pierogi ruskie) - importance 0.0003216530218494853
    475. feature 147 (pos_id_5a4cf492753e9f359145a283) - importance 0.0003213084999171804
    476. feature 511 (company_id_596ca436cf81a34d0c22e644) - importance 0.00032071729003434017
    477. feature 408 (pos_id_5b4c9680ee2a423f39db9c42) - importance 0.0003200853157616096
    478. feature 500 (company_id_5bbdc99c15c00937a08110ca) - importance 0.0003195223540032639
    479. feature 151 (pos_id_5b3391aa0663ab48e3358b7d) - importance 0.0003194975610633615
    480. feature 556 (company_id_5bd01c3ba5ec2005adf8e024) - importance 0.0003192599247930239
    481. feature 521 (company_id_5c4826defd89245e80d411bf) - importance 0.00031849539083169136
    482. feature 664 (company_id_5c35f7ed9557cb13926c53bc) - importance 0.00031819184878736277
    483. feature 309 (pos_id_5cab442e6cb891454c9b8220) - importance 0.00031815936354699084
    484. feature 591 (company_id_5b7c09fb986a9a41c7dc31b0) - importance 0.0003175104221286331
    485. feature 192 (pos_id_5b9616e918e5ca05666c4df3) - importance 0.00031665292758147664
    486. feature 342 (pos_id_5c1232411462a123065f3b58) - importance 0.0003160630054160183
    487. feature 391 (pos_id_5b96228018e5ca05666c514b) - importance 0.000315655061536288
    488. feature 299 (pos_id_5c79198e7b6c863a5d5226fd) - importance 0.00031484957088944045
    489. feature 567 (company_id_5c18cf78e343b1238066ec89) - importance 0.0003143211576673361
    490. feature 781 (product_name_Cocido wegańskie) - importance 0.00031225763361831664
    491. feature 319 (pos_id_5cb9a647df68013fb09db489) - importance 0.0003122191799279326
    492. feature 210 (pos_id_5c18e9d72d1e1a498b462b75) - importance 0.0003102491249132309
    493. feature 532 (company_id_5aa1320aa769d810b9d0229b) - importance 0.00030964814977898683
    494. feature 359 (pos_id_5b680f31fcf5082d8504c993) - importance 0.000307586348495333
    495. feature 373 (pos_id_5b30cd970663ab48e3354851) - importance 0.00030721371524878194
    496. feature 504 (company_id_5a586f5acf5c8134b3a988ed) - importance 0.0003067417040665209
    497. feature 813 (product_name_Szynka duszona z warzywami) - importance 0.00030640500307075127
    498. feature 347 (pos_id_5cebc4c7aa843771f0c00522) - importance 0.0003060932044434312
    499. feature 288 (pos_id_5ca1f61880237d156eb0abef) - importance 0.00030559717608613445
    500. feature 581 (company_id_5a004ad16cdc0d08a6591c3a) - importance 0.0003050836343100504
    501. feature 608 (company_id_5bd0147651478a069f6c3473) - importance 0.0003049536000225217
    502. feature 808 (product_name_Pomidorowa z bazylią) - importance 0.0003039736495147705
    503. feature 105 (pos_id_5b1ba1edcaef965005d0e798) - importance 0.00030379015979240757
    504. feature 220 (pos_id_5c46dab20a0b153373b7dfb7) - importance 0.00030367661887026884
    505. feature 252 (pos_id_5c73e994dd7f597c831012fe) - importance 0.0003036600542053503
    506. feature 214 (pos_id_5c10ca6ad6a8ba18b74e3f7c) - importance 0.0003035031843704828
    507. feature 259 (pos_id_5c8fa223c667da0ef079ece2) - importance 0.0003024901746829727
    508. feature 370 (pos_id_5bd1ba89da98d463e0febd6d) - importance 0.00030219936421710657
    509. feature 127 (pos_id_5a7ab49d3672b770f8f34efd) - importance 0.0003011156871912744
    510. feature 404 (pos_id_5b8672aaa9fd2f7143f74f43) - importance 0.0003006925401433901
    511. feature 784 (product_name_Casserole z kurczakiem i kaparami) - importance 0.0003000947825652425
    512. feature 585 (company_id_5be1513b479c2d2a0197a925) - importance 0.0002995437175514428
    513. feature 542 (company_id_5cb41f8a767bfa626038add6) - importance 0.0002993517436116864
    514. feature 312 (pos_id_5cc047196fdf6843a300244f) - importance 0.00029771742300512227
    515. feature 296 (pos_id_5b3377bf0663ab48e3358824) - importance 0.00029755529950624367
    516. feature 415 (pos_id_5c48277a55f3af637b69b14a) - importance 0.0002974834031602923
    517. feature 303 (pos_id_5bc9984ba6c5c61a73d3f5c4) - importance 0.00029702106145088156
    518. feature 936 (diet_Kuchnia Środziemnomorska) - importance 0.00029697219314438834
    519. feature 554 (company_id_5ba0e78f01f82e03b41319f9) - importance 0.0002965924710434203
    520. feature 344 (pos_id_5cebc9b8aa843771f0c006b2) - importance 0.0002948581756052573
    521. feature 397 (pos_id_5bcd92b1f4a41472d469292d) - importance 0.0002940721890415435
    522. feature 174 (pos_id_5956341c2e5db6320199ecb3) - importance 0.000293187638342658
    523. feature 392 (pos_id_5d1ee8f74152364df18939d5) - importance 0.00029315064828727493
    524. feature 178 (pos_id_5b337dc20663ab48e33588d9) - importance 0.00029313949778552844
    525. feature 616 (company_id_5cc01f608ed57342ff70f093) - importance 0.00029236446039736897
    526. feature 579 (company_id_5c46ec38b578ec3384e608f9) - importance 0.00029205768951328276
    527. feature 912 (address_city_Ruda Śląska) - importance 0.00029009247630071365
    528. feature 196 (pos_id_5b9110e6cd07f81d7e1f8843) - importance 0.0002896220858703442
    529. feature 749 (company_id_nan) - importance 0.00028799932641870065
    530. feature 843 (product_name_Ostry ryż po tajsku) - importance 0.0002878464315898477
    531. feature 341 (pos_id_5ce55a61bfbf435a7c92fa1b) - importance 0.0002877783921046284
    532. feature 239 (pos_id_5b8d24dc90b4de59396b66c5) - importance 0.00028631437480567003
    533. feature 787 (product_name_Pomidor z chili) - importance 0.0002856527929264083
    534. feature 631 (company_id_5bbdd04e15c00937a0811376) - importance 0.0002855913435147912
    535. feature 216 (pos_id_5c18e6dde343b1238066f470) - importance 0.0002848513222244917
    536. feature 536 (company_id_5b1f89d6caef965005d12d76) - importance 0.0002842528777463853
    537. feature 588 (company_id_5c13ad83e343b123806613bb) - importance 0.00028412845925071545
    538. feature 805 (product_name_Indyk smażony z jabłkami) - importance 0.0002840768127217432
    539. feature 667 (company_id_5c8f86a9a7d3a504da7b2348) - importance 0.0002836184359553191
    540. feature 143 (pos_id_5b29179f0663ab48e334ab43) - importance 0.00028232483059680317
    541. feature 633 (company_id_5c46ede0b578ec3384e60991) - importance 0.00028152247919374646
    542. feature 290 (pos_id_5c98fe4fa0d5866f86eb96ff) - importance 0.0002811671338119732
    543. feature 777 (product_name_FitSalad - Sałatka z kurczakiem i grillowanymi warzywami) - importance 0.0002807450858626945
    544. feature 307 (pos_id_5c9e29d880237d156eb013d2) - importance 0.0002779023489282241
    545. feature 185 (pos_id_5baa229ca27a6774b7fa92b9) - importance 0.00027744417938606976
    546. feature 228 (pos_id_5c484d5afd89245e80d41cd5) - importance 0.00027737460002575455
    547. feature 249 (pos_id_5c6eaa346eca0d1f38c368d6) - importance 0.0002748634291002412
    548. feature 273 (pos_id_5c94c8cc1032d642b6348267) - importance 0.0002740951519536987
    549. feature 225 (pos_id_5c44918e0a0b153373b77272) - importance 0.0002733412494551862
    550. feature 688 (company_id_5b9262316c8c383e7c8045a8) - importance 0.0002718613621259941
    551. feature 133 (pos_id_5b5081e09586df16bbb6f217) - importance 0.00027119017006788033
    552. feature 107 (pos_id_595e32822e84531cd204094a) - importance 0.00027044648417010253
    553. feature 339 (pos_id_5ce64b2c9a72d65baf9c94cb) - importance 0.0002703702462662812
    554. feature 130 (pos_id_5b6174ec04852c4704a271d7) - importance 0.0002698796204095643
    555. feature 279 (pos_id_5caf41122c9b633ae260d294) - importance 0.0002698307586068236
    556. feature 315 (pos_id_5c17a88c2d1e1a498b45ee32) - importance 0.00026928877264642244
    557. feature 212 (pos_id_5c17ac232d1e1a498b45eef9) - importance 0.00026885664504939296
    558. feature 908 (address_city_Jagiellońska 74) - importance 0.00026848951316500285
    559. feature 150 (pos_id_5a4ce7b1753e9f359145a229) - importance 0.00026837709113702684
    560. feature 763 (category_name_Lucky Fish) - importance 0.000268044055155855
    561. feature 302 (pos_id_5c79182f3ae6e53aff1488ee) - importance 0.0002669071063953005
    562. feature 665 (company_id_5bd9ade772847a312e4d8c1b) - importance 0.0002666772214904958
    563. feature 848 (product_name_Indyk z pieczarkami i ryżem) - importance 0.0002666350499489415
    564. feature 334 (pos_id_5cd56fa27b91405c9a455962) - importance 0.0002665567328272454
    565. feature 533 (company_id_5b5ebc0601925031d92916b0) - importance 0.0002664858724546157
    566. feature 606 (company_id_5c46ec99b578ec3384e6090c) - importance 0.0002663146206357343
    567. feature 822 (product_name_Schab duszony z batatem) - importance 0.00026617699427609084
    568. feature 849 (product_name_Garlic Chicken) - importance 0.00026536614393424837
    569. feature 634 (company_id_5cc01fa78ed57342ff70f09d) - importance 0.00026497662938492937
    570. feature 643 (company_id_5ce296eed88a9a3711818bb0) - importance 0.00026421632533356
    571. feature 180 (pos_id_5afc6cc535348125c0a0ce7e) - importance 0.00026402005962283527
    572. feature 333 (pos_id_5cd56095aa566714bfc60c5c) - importance 0.00026347389978856176
    573. feature 586 (company_id_5c4826a1fd89245e80d41166) - importance 0.0002631642476337323
    574. feature 571 (company_id_5bc7275aa6c5c61a73d34799) - importance 0.00026249641004135376
    575. feature 386 (pos_id_5d0894a46426b30f2945dd2f) - importance 0.00026084690522977044
    576. feature 821 (product_name_Grzybowy z indykiem) - importance 0.000260476119794704
    577. feature 331 (pos_id_5cd2c7b75314a576b410fb6f) - importance 0.0002583819574435328
    578. feature 93 (pos_id_596e47f971059a6bc70a81ef) - importance 0.0002580932072769627
    579. feature 357 (pos_id_5cf7b59b17002f4e85898ddf) - importance 0.0002574462138102622
    580. feature 522 (company_id_59562dd1e9b1ed3755974025) - importance 0.0002570348324998353
    581. feature 338 (pos_id_5a4cd5b6753e9f359145a1cb) - importance 0.0002568623499790828
    582. feature 824 (product_name_Pieczone nuggetsy z frytkami z batata) - importance 0.00025654394349739986
    583. feature 607 (company_id_5caaeaa8822b5e2d312ac807) - importance 0.00025629129423694846
    584. feature 776 (product_name_FitSalad - Sałatka z fetą, oliwkami i pomidorkami suszonymi) - importance 0.00025426007431852054
    585. feature 421 (pos_id_5bd1b3dada98d463e0febadf) - importance 0.00025375214081812224
    586. feature 385 (pos_id_5b30c32d0663ab48e33546c6) - importance 0.00025272497962414495
    587. feature 295 (pos_id_5cab37f2a77669455cbbaa78) - importance 0.0002498954797183512
    588. feature 179 (pos_id_595638312e5db6320199ecc1) - importance 0.00024662646485333
    589. feature 422 (pos_id_5c866b4f57f8ad7603993be9) - importance 0.00024658173245957227
    590. feature 432 (pos_id_596e464f71059a6bc70a81e4) - importance 0.0002448869777012792
    591. feature 546 (company_id_5af529626e089c600a14f590) - importance 0.00024434592980283946
    592. feature 627 (company_id_5c178e482d1e1a498b45e6b1) - importance 0.00024395498567823158
    593. feature 281 (pos_id_5c79285d7b6c863a5d522c20) - importance 0.00024341825821673239
    594. feature 637 (company_id_5ba0e7be01f82e03b41319ff) - importance 0.0002431024130929726
    595. feature 859 (product_name_Klopsiki w sosie pomidorowym) - importance 0.00024302443153725992
    596. feature 393 (pos_id_59fc472ac94b722506678bad) - importance 0.0002428593096381324
    597. feature 242 (pos_id_5c6eaec7531c6b7fe27e1658) - importance 0.00024180081975974316
    598. feature 623 (company_id_5bc724f8a6c5c61a73d3469b) - importance 0.00023935988272890432
    599. feature 604 (company_id_5b7c105b986a9a41c7dc336e) - importance 0.0002382168688191917
    600. feature 782 (product_name_Smoothie BeRAW Breakfast energy) - importance 0.00023712808073142417
    601. feature 402 (pos_id_5d36db72037a6d08badca719) - importance 0.00023489720083185494
    602. feature 905 (address_city_Zabierzów) - importance 0.00023264053900777463
    603. feature 651 (company_id_5a0eb9b93abf453d0d5b2ad4) - importance 0.00022942226435623733
    604. feature 810 (product_name_Pierogi z soczewicą i suszonymi pomidorami) - importance 0.00022935188066637525
    605. feature 362 (pos_id_5cf7ad6eb03e874ed5eb2206) - importance 0.00022926509641582553
    606. feature 321 (pos_id_5cb9928e571460403ca06adf) - importance 0.00022906106816558436
    607. feature 138 (pos_id_5b47022eee2a423f39da6908) - importance 0.00022852083095595431
    608. feature 244 (pos_id_5c629b461a466d77f43a3a0f) - importance 0.00022845815748063277
    609. feature 346 (pos_id_5cd166ee0a544c2d0d156279) - importance 0.00022753357283724292
    610. feature 213 (pos_id_5c10d1eed6a8ba18b74e40c7) - importance 0.00022735528578781786
    611. feature 611 (company_id_5cae003777d34435ac133cf8) - importance 0.00022727955685508197
    612. feature 328 (pos_id_5cd56f547b91405c9a455905) - importance 0.00022710859318498074
    613. feature 343 (pos_id_5cdbd1ba9083f77a4476ff58) - importance 0.00022704843879783773
    614. feature 241 (pos_id_5c46d436b578ec3384e6032f) - importance 0.00022666796716022024
    615. feature 389 (pos_id_5b372d990663ab48e335e203) - importance 0.00022660444695890984
    616. feature 617 (company_id_5c6d6303531c6b7fe27dcc11) - importance 0.00022618962545644755
    617. feature 593 (company_id_59562d52e9b1ed3755974022) - importance 0.00022578836363790905
    618. feature 428 (pos_id_5c86640d57f8ad760399398f) - importance 0.00022553337211666654
    619. feature 429 (pos_id_5d68fff0c8c4a843bc5a06de) - importance 0.0002255325914341402
    620. feature 414 (pos_id_5c1236661462a123065f3c50) - importance 0.00022484554404434624
    621. feature 420 (pos_id_5d2826e3769816286982b757) - importance 0.00022404049084028777
    622. feature 412 (pos_id_5d36d0493e54a108a8c9d160) - importance 0.00022349271370949723
    623. feature 310 (pos_id_5caf32c325750f384eb3c779) - importance 0.00022346553770171332
    624. feature 709 (company_id_5beadd48bf6ea32c7a3b3516) - importance 0.0002231350183319918
    625. feature 913 (address_city_Prądnicka 65) - importance 0.00022309654501886104
    626. feature 804 (product_name_Łazanki z kapustą) - importance 0.00022219019181507504
    627. feature 647 (company_id_5cf7896a4de1ce44a055fdcc) - importance 0.00022166330181563903
    628. feature 231 (pos_id_5c17a4c02d1e1a498b45ed24) - importance 0.0002208328809314178
    629. feature 193 (pos_id_5b866c9ea9fd2f7143f74ce4) - importance 0.00021962098711174436
    630. feature 368 (pos_id_5c480f01fd89245e80d40ba6) - importance 0.00021900599016026658
    631. feature 919 (address_city_51.126901, 16.978188) - importance 0.00021679328955838458
    632. feature 271 (pos_id_5c6156dbc88bae77186d4fe7) - importance 0.00021585452659335713
    633. feature 340 (pos_id_5ce29094d88a9a3711818a2f) - importance 0.00021497769065622593
    634. feature 155 (pos_id_5b33a2b20663ab48e3358d5b) - importance 0.00021407745309148086
    635. feature 630 (company_id_5c18cf4b2d1e1a498b46228f) - importance 0.00021385624348099245
    636. feature 788 (product_name_Brokuł z czosnkiem i ryżem) - importance 0.000212715232428243
    637. feature 352 (pos_id_5cb996ca571460403ca06c08) - importance 0.00021230932350746693
    638. feature 293 (pos_id_5caf310b2e26ff2d68de0bf8) - importance 0.00021204060204291973
    639. feature 377 (pos_id_5cffb8c36426b30f294426c4) - importance 0.0002116636104218055
    640. feature 327 (pos_id_5c18e2882d1e1a498b462943) - importance 0.00021144201457275503
    641. feature 635 (company_id_5c87bb8634b9b6757151c555) - importance 0.0002112306029689987
    642. feature 410 (pos_id_5d36c1f83e54a108a8c9cebd) - importance 0.00021081094652882369
    643. feature 376 (pos_id_5ab2a8916578ab4a531c63d9) - importance 0.00021079983947303038
    644. feature 609 (company_id_5cadfff5f187f735ed704e45) - importance 0.00021038894249931543
    645. feature 351 (pos_id_5cebe1f17145ad7268198664) - importance 0.00020871272343336088
    646. feature 809 (product_name_Marokańska z quinoą, batatem i kolendrą) - importance 0.0002085831139971145
    647. feature 710 (company_id_5cc01faddef57b4350f57e71) - importance 0.00020780824247455584
    648. feature 642 (company_id_5ce29693d88a9a3711818ba6) - importance 0.00020745406229157547
    649. feature 723 (company_id_5d008f1cf88ba26d014e2353) - importance 0.00020489783785650222
    650. feature 425 (pos_id_5c6172731a466d77f439fb8a) - importance 0.00020421355489859947
    651. feature 322 (pos_id_5cb999ac0608cd75a131308d) - importance 0.00020416632242275192
    652. feature 619 (company_id_5cb970cddf68013fb09da910) - importance 0.0002041639831200231
    653. feature 407 (pos_id_5b714b89d8f80d6ebf31d7ab) - importance 0.00020411091015124286
    654. feature 86 (pos_id_595e36122e84531cd204096b) - importance 0.0002037935461942044
    655. feature 301 (pos_id_5c8f9c3aa7d3a504da7b2b0e) - importance 0.00020285744936081204
    656. feature 263 (pos_id_5c73da87dd7f597c83100dfa) - importance 0.0002023796729909553
    657. feature 513 (company_id_59562d1ae9b1ed3755974021) - importance 0.00020194007294650686
    658. feature 790 (product_name_Pieczeń rzymska w siemieniu lnianym) - importance 0.00020163554902086642
    659. feature 317 (pos_id_5c86699334b9b675715174ae) - importance 0.00020081349091692182
    660. feature 677 (company_id_5d008f0cf88ba26d014e2352) - importance 0.00020033570449688143
    661. feature 152 (pos_id_5926df3d26456576dfcc3e90) - importance 0.0002001634204525363
    662. feature 583 (company_id_5b30ea120663ab48e3354bfc) - importance 0.00019877955207860118
    663. feature 261 (pos_id_5c73d67edd7f597c83100c6a) - importance 0.0001987017224033864
    664. feature 660 (company_id_5d008ef16426b30f29444db9) - importance 0.00019860689701382117
    665. feature 618 (company_id_5c35f81cdc581313af320c6d) - importance 0.00019778814555138056
    666. feature 815 (product_name_Pęczotto ze szpinakiem) - importance 0.00019743943100205394
    667. feature 440 (pos_id_5cffb48d6426b30f294425c0) - importance 0.00019669634941112585
    668. feature 570 (company_id_5baa1ba0a27a6774b7fa8fc0) - importance 0.0001963118418822562
    669. feature 854 (product_name_Putanesca z oliwkami) - importance 0.00019581996291077983
    670. feature 909 (address_city_30-001) - importance 0.000194554882338039
    671. feature 117 (pos_id_5b1ba6d0caef965005d0e802) - importance 0.00019420968666889648
    672. feature 639 (company_id_5c86757934b9b67571517813) - importance 0.0001941116774919232
    673. feature 845 (product_name_Schab w orientalnym stylu) - importance 0.00019320885734231178
    674. feature 727 (company_id_5d008ebef88ba26d014e234d) - importance 0.00019282800995148541
    675. feature 783 (product_name_Smoothie BeRAW Detox #coolGREENS) - importance 0.00019281331231411726
    676. feature 807 (product_name_Indyjska z ciecierzycą i curry) - importance 0.00019278717629738588
    677. feature 602 (company_id_5c612e1dcfa3a62fcbee67a9) - importance 0.0001913468077674102
    678. feature 361 (pos_id_5cebd2c07145ad726819829a) - importance 0.00019069781794079098
    679. feature 165 (pos_id_5afc6f8e35348125c0a0cec8) - importance 0.00019052896321796711
    680. feature 713 (company_id_5baa1bbda27a6774b7fa8fc8) - importance 0.00018857901661745134
    681. feature 358 (pos_id_5b68191dfcf5082d8504cbda) - importance 0.00018766117349239094
    682. feature 668 (company_id_5d008f30c247d66d1d43f521) - importance 0.00018743990772196491
    683. feature 600 (company_id_5c73d9e2531c6b7fe27f29d9) - importance 0.00018714491049088344
    684. feature 240 (pos_id_5c6aa423ae90030afa270bb0) - importance 0.00018665860183015527
    685. feature 852 (product_name_Teriyaki Chicken) - importance 0.00018607782322915177
    686. feature 555 (company_id_5bacd4cf26a9cb3d33cf4bb9) - importance 0.0001856415989468928
    687. feature 380 (pos_id_5d1365db0416e22369ab0e06) - importance 0.000183739869075299
    688. feature 656 (company_id_5c178cfbe343b1238066aff3) - importance 0.00018369992864021402
    689. feature 298 (pos_id_5c8235dd34b9b67571508681) - importance 0.00018324660301787774
    690. feature 728 (company_id_5c7fab686b25e24bf5028b41) - importance 0.00018311249086736062
    691. feature 692 (company_id_5d15ee224152364df186f6d7) - importance 0.00018305337393300226
    692. feature 812 (product_name_Pierogi z dynią, jarmużem, quinoą i kolendrą) - importance 0.00018225995309614854
    693. feature 626 (company_id_5c178d48e343b1238066affd) - importance 0.0001822574015861233
    694. feature 452 (pos_id_5dbac6883bf8f15dc97d4251) - importance 0.00018118531822122172
    695. feature 381 (pos_id_5d136c407575ed1e91b51af5) - importance 0.0001790113853627337
    696. feature 718 (company_id_5c87ae9f34b9b6757151c208) - importance 0.00017741920894869177
    697. feature 670 (company_id_5d008f2a6426b30f29444dbd) - importance 0.00017724994293707676
    698. feature 802 (product_name_FitBreak - Chlebek gryczany z kurczakiem i hummusem) - importance 0.00017722021659572433
    699. feature 864 (product_name_Kopytka z dynią i jarmużem) - importance 0.00017637059143087543
    700. feature 418 (pos_id_5d36ccaf92520d08aefbf42c) - importance 0.00017569875070782952
    701. feature 353 (pos_id_5b30d00b0663ab48e33548c3) - importance 0.00017438975699854084
    702. feature 350 (pos_id_5cd563af73d91d192aba5de5) - importance 0.00017407292621197465
    703. feature 654 (company_id_5ca1b58ce2258413d5bcd415) - importance 0.00017314717526080217
    704. feature 652 (company_id_5c6d63c30eca040242a5a756) - importance 0.0001712815802929127
    705. feature 572 (company_id_5beaaf32bf6ea32c7a3b29b4) - importance 0.0001705373156712157
    706. feature 658 (company_id_5c92302ab3b33056dbe442eb) - importance 0.00017030993686631784
    707. feature 685 (company_id_5ce29721c828ba60d80932ab) - importance 0.00017030835156464449
    708. feature 243 (pos_id_5c615eafc88bae77186d5377) - importance 0.00016988689503264398
    709. feature 292 (pos_id_5c9e247680237d156eb012d1) - importance 0.0001691597746228464
    710. feature 622 (company_id_5ba0e74501f82e03b41319e2) - importance 0.00016915288157084442
    711. feature 365 (pos_id_5ce54ef19269c85a9a4f112f) - importance 0.00016645135936946807
    712. feature 920 (address_city_Nowa Wieś Wrocławska) - importance 0.00016644275190845633
    713. feature 655 (company_id_5ce2972a574f8c0602bd58c0) - importance 0.00016556471594057214
    714. feature 578 (company_id_5bacd51926a9cb3d33cf4c1f) - importance 0.00016527756307382018
    715. feature 640 (company_id_5cc01fa1def57b4350f57e70) - importance 0.00016516306919718535
    716. feature 699 (company_id_5d008f03f88ba26d014e2351) - importance 0.00016486844240153878
    717. feature 614 (company_id_5ca1b2a7e2258413d5bcd369) - importance 0.00016433392632898153
    718. feature 287 (pos_id_5a58c2d8cf5c8134b3a98b03) - importance 0.0001642567555305166
    719. feature 800 (product_name_Smoothie Be Raw Vegan protein acai) - importance 0.00016395752063201222
    720. feature 355 (pos_id_5ce29f63c828ba60d8093539) - importance 0.00016341620328927923
    721. feature 785 (product_name_Smoothie Be Raw Vegan protein mango) - importance 0.00016336217071717207
    722. feature 773 (product_name_BeRAW Baton healthy snack - masło orzechowe) - importance 0.0001625988045349467
    723. feature 878 (product_name_All'arrabbiata) - importance 0.00016189013285963902
    724. feature 774 (product_name_BeRAW Baton energy - surowe kakao, kokos) - importance 0.00016082329937185856
    725. feature 311 (pos_id_5c9e220280237d156eb0123d) - importance 0.00016040294801207943
    726. feature 798 (product_name_Barszcz z mlekiem kokosowym) - importance 0.00015981211573787692
    727. feature 686 (company_id_5c8f6b92a7d3a504da7b19f0) - importance 0.00015958577241458518
    728. feature 323 (pos_id_5caf3c8b7807893ad0176299) - importance 0.00015948139703091983
    729. feature 354 (pos_id_5ce29384c828ba60d80931d3) - importance 0.00015941995257834655
    730. feature 406 (pos_id_5cab4c086cb891454c9b8413) - importance 0.0001592740413529915
    731. feature 775 (product_name_FitBreak - Ryż na mleku kokosowym z owocami) - importance 0.0001592635704006255
    732. feature 801 (product_name_Marchewka z imbirem i kaszą jaglaną) - importance 0.0001570135691272928
    733. feature 638 (company_id_5c4ac5b955f3af637b6a4a65) - importance 0.000156958919705374
    734. feature 313 (pos_id_5cc03f045e70463ad9d52019) - importance 0.0001563129184554354
    735. feature 403 (pos_id_5b681f88fcf5082d8504cde6) - importance 0.00015540499305692214
    736. feature 910 (address_city_Bielany Wrocławskie) - importance 0.00015339283276129305
    737. feature 921 (address_city_Krakow) - importance 0.00015184372965539107
    738. feature 687 (company_id_5c90def3a7d3a504da7b7995) - importance 0.0001510336118295681
    739. feature 454 (pos_id_nan) - importance 0.0001506479551353461
    740. feature 437 (pos_id_5d6e4af83e54a108a8d21cb7) - importance 0.00015013639306245654
    741. feature 278 (pos_id_5c78fd177b6c863a5d521e08) - importance 0.00014920344760759375
    742. feature 794 (product_name_Superfood ZDROWIE - kakao, maca) - importance 0.0001490931865020031
    743. feature 184 (pos_id_5a564bf7cf5c8134b3a98648) - importance 0.00014878371869816198
    744. feature 711 (company_id_5cab058e822b5e2d312acd00) - importance 0.00014870720922059176
    745. feature 624 (company_id_5aaa3ee5cb73bd163e8b3ffe) - importance 0.00014850753819684032
    746. feature 409 (pos_id_5d36e78e92520d08aefbfafd) - importance 0.00014798711984542067
    747. feature 674 (company_id_5cae024f77d34435ac133d76) - importance 0.000147327780011604
    748. feature 698 (company_id_5b6a83206afd6509aeb6d99f) - importance 0.0001472284314379365
    749. feature 405 (pos_id_5d31765cc30c8408b4b02c2d) - importance 0.00014647616522119267
    750. feature 833 (product_name_Kremowy z kurczakiem) - importance 0.0001463386881237621
    751. feature 830 (product_name_Zupa Pomidorowa) - importance 0.0001457089126695334
    752. feature 364 (pos_id_5cebdde87145ad7268198589) - importance 0.0001453479564268268
    753. feature 363 (pos_id_5ce3d2e6d88a9a371181c913) - importance 0.00014425922822845002
    754. feature 325 (pos_id_5cc047e75e70463ad9d5238d) - importance 0.00014407985800237274
    755. feature 628 (company_id_5cc01f7c8ed57342ff70f09a) - importance 0.0001437297524334971
    756. feature 653 (company_id_5ce296a6c828ba60d8093298) - importance 0.00014300245428535503
    757. feature 306 (pos_id_5c7e78a354a4233b10f350be) - importance 0.00014279175280379358
    758. feature 360 (pos_id_5cd5716273d91d192aba6328) - importance 0.00014270475362989694
    759. feature 645 (company_id_5ce29741574f8c0602bd58c8) - importance 0.0001418817333920749
    760. feature 793 (product_name_Superfood ENERGIA - kokos, lukuma) - importance 0.00014134147867645333
    761. feature 400 (pos_id_5d28362797a5c32879fa6f75) - importance 0.0001411771706682322
    762. feature 396 (pos_id_5c6ea50d531c6b7fe27e139b) - importance 0.00014071048754965427
    763. feature 682 (company_id_5caaf9da7211b22c2502bfcc) - importance 0.00014070546790343152
    764. feature 738 (company_id_5a576b80cf5c8134b3a98808) - importance 0.00013962239132250503
    765. feature 305 (pos_id_5c9e0994e2258413d5bc438c) - importance 0.00013834558334297698
    766. feature 705 (company_id_5beac47fbf6ea32c7a3b2f77) - importance 0.00013816756595116708
    767. feature 720 (company_id_5d008f37c247d66d1d43f522) - importance 0.00013784790769671252
    768. feature 649 (company_id_5c7cd37e3ae6e53aff1548b4) - importance 0.00013691196291795398
    769. feature 561 (company_id_5a5876bfcf5c8134b3a98917) - importance 0.00013682196675738985
    770. feature 411 (pos_id_5b07a8d3a73a156ec5356b29) - importance 0.00013666255477917788
    771. feature 663 (company_id_5c73e01b531c6b7fe27f2be3) - importance 0.0001365686805207629
    772. feature 716 (company_id_5d15ed4b4152364df186f69b) - importance 0.00013649655366664156
    773. feature 697 (company_id_5d15ecef4152364df186f647) - importance 0.00013648168857475915
    774. feature 648 (company_id_5bacd4f426a9cb3d33cf4bd4) - importance 0.0001362695067086975
    775. feature 669 (company_id_5ce29749574f8c0602bd58cb) - importance 0.00013554585899053123
    776. feature 867 (product_name_Kotleciki ze szpinakiem z puree z zielonego groszku) - importance 0.00013541025833128717
    777. feature 419 (pos_id_5b291c490663ab48e334abce) - importance 0.0001350203208809398
    778. feature 863 (product_name_KIMBAP z łososiem) - importance 0.00013379496890926788
    779. feature 657 (company_id_5caf2cf3b3929d2dd0c7cec9) - importance 0.0001327903075031237
    780. feature 795 (product_name_Superfood UMYSŁ - goji acai) - importance 0.00013182226273616678
    781. feature 879 (product_name_KIMBAP z tuńczykiem) - importance 0.00013074017139290438
    782. feature 691 (company_id_5d15ee355376d84d27e456cd) - importance 0.0001302211352019791
    783. feature 791 (product_name_Superfood SPORT - banan, białko) - importance 0.0001292712937446228
    784. feature 441 (pos_id_5a0eb2cbffa5f308210e7486) - importance 0.00012872665058105662
    785. feature 678 (company_id_5b6a83556afd6509aeb6d9a8) - importance 0.00012847465785608073
    786. feature 693 (company_id_5d008f23f88ba26d014e2354) - importance 0.00012762986015349997
    787. feature 702 (company_id_5b7c1500986a9a41c7dc34af) - importance 0.0001271091579929397
    788. feature 318 (pos_id_5c9e04be80237d156eb00b85) - importance 0.00012557486411666005
    789. feature 434 (pos_id_5b4ca2eeee2a423f39db9fe9) - importance 0.00012465556207830772
    790. feature 433 (pos_id_5d690708ca0ee72fd208a996) - importance 0.0001237939824719618
    791. feature 717 (company_id_5baa1b4fa27a6774b7fa8fae) - importance 0.00012336694673005053
    792. feature 390 (pos_id_5c811efb4599f22d051878ce) - importance 0.00012308747110933432
    793. feature 424 (pos_id_5baa158da27a6774b7fa8ca3) - importance 0.00012255867410220352
    794. feature 926 (address_city_nan) - importance 0.0001218483094593684
    795. feature 374 (pos_id_5ce3cb0c0aebca2c62f3f196) - importance 0.00012145572020755606
    796. feature 725 (company_id_5d764df1bf7a586310594d8b) - importance 0.00012124573276195635
    797. feature 839 (product_name_Ogórkowa z koperkiem) - importance 0.00012069105876483737
    798. feature 413 (pos_id_5d281f86a9af282863de43c2) - importance 0.00012065412876183362
    799. feature 384 (pos_id_5cffb58bc247d66d1d43ccfa) - importance 0.0001202657628309986
    800. feature 703 (company_id_5d764dfe341dd662dea1c710) - importance 0.00012021150436374088
    801. feature 695 (company_id_5d008ee16426b30f29444db7) - importance 0.00011976745539010667
    802. feature 621 (company_id_5c34a54bdc581313af31c67a) - importance 0.00011969675135352394
    803. feature 439 (pos_id_5c49aa8855f3af637b6a0dc7) - importance 0.00011902492171985432
    804. feature 625 (company_id_5ca1ac861b2b36155c424c73) - importance 0.00011884224833258725
    805. feature 676 (company_id_5d008f4bc247d66d1d43f523) - importance 0.0001168753715153037
    806. feature 873 (product_name_Ensopado) - importance 0.00011624124705889326
    807. feature 823 (product_name_Zupa Marchewkowa) - importance 0.00011478758691319033
    808. feature 684 (company_id_5d14ae03ad08444d3ee1eecd) - importance 0.0001144480383984398
    809. feature 935 (diet_413,0) - importance 0.00011402967185753612
    810. feature 366 (pos_id_5ce2883dd88a9a371181876b) - importance 0.0001138447041179825
    811. feature 672 (company_id_5c6d520f93608759cf919a2d) - importance 0.00011364544968011889
    812. feature 855 (product_name_Zupa Dyniowa) - importance 0.00011320817710498831
    813. feature 443 (pos_id_5d5e2f74c9659d0da3099169) - importance 0.000110972438966924
    814. feature 827 (product_name_Zupa Brokułowa) - importance 0.00011074039685429537
    815. feature 673 (company_id_5c178f7d2d1e1a498b45e6f1) - importance 0.00011036730003482406
    816. feature 423 (pos_id_5b754e189c7c767ae05739b6) - importance 0.00010944157867555327
    817. feature 829 (product_name_Hiszpańska Tortilla) - importance 0.00010868534401137508
    818. feature 589 (company_id_5b1f898acaef965005d12d6a) - importance 0.00010572429622207627
    819. feature 706 (company_id_5c63cb3c1a466d77f43a8196) - importance 0.00010284458548231914
    820. feature 345 (pos_id_5ce556769269c85a9a4f12d8) - importance 0.00010160858349137857
    821. feature 779 (product_name_BeRAW Baton protein 38% - surowe kakao w gorzkiej czekoladzie) - importance 0.00010152585962033012
    822. feature 736 (company_id_5d15ee715376d84d27e456e2) - importance 0.00010004728939714717
    823. feature 336 (pos_id_5cd2c3b9f06e7d6a61999df6) - importance 9.924726004331955e-05
    824. feature 831 (product_name_FitRoślanka owoce czerwone) - importance 9.846498942210223e-05
    825. feature 666 (company_id_5ce296c2c828ba60d809329d) - importance 9.841622650825291e-05
    826. feature 375 (pos_id_5cf7b34617002f4e85898d54) - importance 9.756709214410869e-05
    827. feature 915 (address_city_Wysoka, Wrocław) - importance 9.68123604159418e-05
    828. feature 816 (product_name_Zupa Pietruszkowa) - importance 9.587770489411293e-05
    829. feature 417 (pos_id_5b86768ca9fd2f7143f750b2) - importance 9.552896442859883e-05
    830. feature 444 (pos_id_5c86710957f8ad7603993d55) - importance 9.507454801758341e-05
    831. feature 435 (pos_id_5d70d6da3e54a108a8d2819b) - importance 9.499384988321237e-05
    832. feature 700 (company_id_5cab0ac6822b5e2d312ace0e) - importance 9.494378829286042e-05
    833. feature 834 (product_name_Wegański z masłem orzechowym) - importance 9.394055032464566e-05
    834. feature 917 (address_city_Wrocław, Biskupice Podgórne) - importance 9.389934526295586e-05
    835. feature 679 (company_id_5c90ec23a4b28f5f98967a31) - importance 9.255448480474201e-05
    836. feature 671 (company_id_5ce29683574f8c0602bd58aa) - importance 9.248928745277598e-05
    837. feature 865 (product_name_Sushi SAJDO) - importance 9.184381123494332e-05
    838. feature 721 (company_id_5be445ecaac41f2436a1b8fa) - importance 9.163810088146873e-05
    839. feature 438 (pos_id_5b2910f20663ab48e334aaaa) - importance 9.075635674144102e-05
    840. feature 707 (company_id_5d15ee475376d84d27e456d7) - importance 9.0353811045849e-05
    841. feature 690 (company_id_5d008efb6426b30f29444dba) - importance 8.93170935070686e-05
    842. feature 701 (company_id_5d15ec614152364df186f5cf) - importance 8.909878310229021e-05
    843. feature 789 (product_name_Be Raw Baton Vegan Protein) - importance 8.869613388100007e-05
    844. feature 726 (company_id_5b2b37e20663ab48e334d3b9) - importance 8.750119338072299e-05
    845. feature 836 (product_name_FitRoślanka kakao) - importance 8.633576399574779e-05
    846. feature 372 (pos_id_5ce558079269c85a9a4f1368) - importance 8.594898580576967e-05
    847. feature 722 (company_id_5d5a90acc9659d0da308f5ca) - importance 8.562527559445194e-05
    848. feature 426 (pos_id_5b8518844f54cd7a668fcd60) - importance 8.541408741100383e-05
    849. feature 436 (pos_id_5b8faf4e8667b879c9f86114) - importance 8.424742322334416e-05
    850. feature 641 (company_id_5c4ace22fd89245e80d4ac18) - importance 8.34966215932459e-05
    851. feature 856 (product_name_FitSalad - Sałatka z tofu) - importance 8.305644537940999e-05
    852. feature 388 (pos_id_5cffb9ccc247d66d1d43ce07) - importance 8.189946775594876e-05
    853. feature 714 (company_id_5d15ed5b5376d84d27e4568d) - importance 8.00557068081707e-05
    854. feature 681 (company_id_5d008eb5c247d66d1d43f51e) - importance 7.892001672306119e-05
    855. feature 866 (product_name_FitBreak - Omlet z sosem salsa) - importance 7.823612303974757e-05
    856. feature 840 (product_name_BeRAW Baton Raspberry Choco Power) - importance 7.643222368516909e-05
    857. feature 694 (company_id_5c88ab8c34b9b6757151f8ca) - importance 7.596168759227163e-05
    858. feature 922 (address_city_12) - importance 7.58455028046451e-05
    859. feature 874 (product_name_Sushi UNAGI) - importance 7.547086857911073e-05
    860. feature 724 (company_id_5bd9ae5328a8882be911293a) - importance 7.473169972286788e-05
    861. feature 442 (pos_id_5d7f8a76341dd662dea3394d) - importance 7.415040551248458e-05
    862. feature 735 (company_id_5d008ed96426b30f29444db6) - importance 7.379561508219204e-05
    863. feature 427 (pos_id_5bd1c2dada98d463e0fec063) - importance 7.326017231381979e-05
    864. feature 704 (company_id_5c87b35d57f8ad7603998999) - importance 7.132516781208154e-05
    865. feature 731 (company_id_5b869df32dd588270f00a8d7) - importance 7.106613861212341e-05
    866. feature 894 (product_name_nan) - importance 7.076731535702716e-05
    867. feature 719 (company_id_5ce2968bc828ba60d8093294) - importance 7.036441911206971e-05
    868. feature 844 (product_name_BeRAW Baton Proteinowy - Żurawina) - importance 6.721752983970591e-05
    869. feature 715 (company_id_5d008ec76426b30f29444db4) - importance 6.720440043307701e-05
    870. feature 659 (company_id_5c6687d36961290fe7419234) - importance 6.58703101429188e-05
    871. feature 737 (company_id_5d764dd4bf7a586310594d84) - importance 6.400800753794087e-05
    872. feature 708 (company_id_5d008f436426b30f29444dbe) - importance 6.289506108838689e-05
    873. feature 838 (product_name_Zupa Buraczkowa) - importance 5.90164594335658e-05
    874. feature 828 (product_name_FitSmoothie - jabłko, gruszka, mięta) - importance 5.6214745224681756e-05
    875. feature 841 (product_name_BeRAW Baton Proteinowy - Wanilia) - importance 5.568769594003706e-05
    876. feature 734 (company_id_5d15ed654152364df186f6a4) - importance 5.4695617595560526e-05
    877. feature 826 (product_name_FitSmoothie - truskawka, malina, żurawina) - importance 5.4353670270134005e-05
    878. feature 792 (product_name_BeRAW Kuleczki whey protein - surowe kakao, sól himalajska) - importance 5.290201832791681e-05
    879. feature 712 (company_id_5d15ee4f4152364df186f6e7) - importance 5.271637483582188e-05
    880. feature 430 (pos_id_5d5fc860ca0ee72fd2072961) - importance 5.20166024804602e-05
    881. feature 449 (pos_id_5db1a544331ac7386908e021) - importance 5.1333092272529696e-05
    882. feature 857 (product_name_Pieczeń rzymska z siemieniem lnianym) - importance 5.125869570185579e-05
    883. feature 399 (pos_id_5d2465ef4152364df18a4842) - importance 5.088952424110073e-05
    884. feature 846 (product_name_FitBreak - Chlebek gryczany z fasolką w pomidorach) - importance 4.9938953619791364e-05
    885. feature 744 (company_id_5d764e93d0c59c62f781ff40) - importance 4.9629502906736066e-05
    886. feature 696 (company_id_5caafe917211b22c2502c098) - importance 4.9021760962865464e-05
    887. feature 835 (product_name_FitElixir - Yellow Balance) - importance 4.881372991651835e-05
    888. feature 431 (pos_id_5d5a357b3e54a108a8cf0a62) - importance 4.8500815736018786e-05
    889. feature 447 (pos_id_5db180f0d3163e50024c25d5) - importance 4.5566316633913984e-05
    890. feature 683 (company_id_5cc01f886fdf6843a30017e8) - importance 4.422825427285306e-05
    891. feature 923 (address_city_02-092) - importance 4.3949261590401476e-05
    892. feature 825 (product_name_SMOOTHIE CHIA - Malina + Ananas) - importance 4.290876398823162e-05
    893. feature 918 (address_city_Wrocławiu) - importance 4.111019600458206e-05
    894. feature 743 (company_id_5d008ed06426b30f29444db5) - importance 4.0683090969737016e-05
    895. feature 814 (product_name_Be RAW Kuleczki Protein Truffles) - importance 3.765070222943313e-05
    896. feature 925 (address_city_30-150) - importance 3.56188949854928e-05
    897. feature 746 (company_id_5dbff5229f38f239c2ffeb55) - importance 3.409454591190092e-05
    898. feature 689 (company_id_5c4ace1655f3af637b6a4c57) - importance 3.393209799707503e-05
    899. feature 832 (product_name_FitElixir - Black Detox) - importance 3.337290211211422e-05
    900. feature 445 (pos_id_5da42387c0681416afe34de0) - importance 3.219020748520583e-05
    901. feature 842 (product_name_BeRAW Baton Proteinowy - Orzechy Arachidowe) - importance 3.14778349839263e-05
    902. feature 729 (company_id_5cab016c7211b22c2502c109) - importance 3.131379253454185e-05
    903. feature 762 (category_name_DayUp) - importance 3.09720165168559e-05
    904. feature 861 (product_name_Pietruszka z gruszką) - importance 2.891726737701724e-05
    905. feature 733 (company_id_5ce296af574f8c0602bd58b1) - importance 2.83555104815667e-05
    906. feature 451 (pos_id_5dbff693bb783c5423163b8d) - importance 2.7474178899815207e-05
    907. feature 739 (company_id_5d15edf24152364df186f6c4) - importance 2.7264355382909785e-05
    908. feature 448 (pos_id_5db818de0e500c595cafaca9) - importance 2.6389813047505546e-05
    909. feature 450 (pos_id_5dbac8d9bb783c5423159cf7) - importance 2.5818810413145533e-05
    910. feature 730 (company_id_5c8f6bfaa7d3a504da7b19f9) - importance 2.5185960092623906e-05
    911. feature 889 (product_name_Silny Łasuch - Marchew w tropikach) - importance 2.463444681812419e-05
    912. feature 877 (product_name_Dyniowa z cynamonem) - importance 2.4495711355727556e-05
    913. feature 876 (product_name_KIMBAP z surimi) - importance 2.3834869261932406e-05
    914. feature 884 (product_name_DayUp Pearls mango) - importance 2.083027535484697e-05
    915. feature 890 (product_name_Silny Łasuch - Cynamonowe jabłuszko) - importance 2.07052995450626e-05
    916. feature 924 (address_city_Wysoka) - importance 1.982510251612681e-05
    917. feature 453 (pos_id_5dc28ac6b902d15de1f0152b) - importance 1.9473433786138738e-05
    918. feature 875 (product_name_FitElixir - yellow balance) - importance 1.863302512467559e-05
    919. feature 745 (company_id_5d764e55bf7a586310594dac) - importance 1.711857681401606e-05
    920. feature 883 (product_name_Superfood - SPORT - banan - białko) - importance 1.6267657637293884e-05
    921. feature 850 (product_name_BeRAW Kuleczki vegan protein - mięta, surowe kakao) - importance 1.5821043957966668e-05
    922. feature 893 (product_name_Allarrabbiata) - importance 1.3663986875465114e-05
    923. feature 51 (minLastPeriod_lag1_lag7_diff) - importance 1.340100148839851e-05
    924. feature 880 (product_name_Superfood - ENERGIA - kokos - lukuma) - importance 1.2920040229820534e-05
    925. feature 740 (company_id_5d764e19bf7a586310594d9f) - importance 1.1886628664810892e-05
    926. feature 862 (product_name_BeRAW Baton Vegan Protein) - importance 1.1803314513847028e-05
    927. feature 446 (pos_id_5d93355803fffb4963c8b9d1) - importance 1.1772625462630915e-05
    928. feature 871 (product_name_Kasza jaglana z bakaliami) - importance 1.0266656804294823e-05
    929. feature 888 (product_name_DayUp Wake&Joy banan-malina) - importance 9.91438266023154e-06
    930. feature 885 (product_name_DayUp Pearls malina) - importance 9.763973528219932e-06
    931. feature 49 (minLastPeriod_lag1) - importance 9.730144953987446e-06
    932. feature 891 (product_name_Chili con carne z ryżen) - importance 8.931964703722647e-06
    933. feature 892 (product_name_ROŚLEKO - jaglano - orzechowe) - importance 8.262449693356147e-06
    934. feature 760 (category_name_Desery) - importance 7.3456110791413025e-06
    935. feature 881 (product_name_DayUp Purple) - importance 6.694027082545099e-06
    936. feature 732 (company_id_5d764de3f2e03462c6f98396) - importance 6.12821400132623e-06
    937. feature 882 (product_name_DayUp Black) - importance 5.504716476633869e-06
    938. feature 50 (minLastPeriod_lag7) - importance 5.394896944574322e-06
    939. feature 868 (product_name_BeRAW Kuleczki Protein Truffles) - importance 4.562252565371665e-06
    940. feature 886 (product_name_DayUp Wake&Joy jabłko-cynamon) - importance 4.461026534724963e-06
    941. feature 887 (product_name_Superfood - UMYSŁ - goji acai) - importance 3.958690706131797e-06
    942. feature 872 (product_name_BeRAW Baton superfood - burak, surowe kakao) - importance 1.565470538368117e-06
    943. feature 748 (company_id_5dc534a2fdcd0622bd7b34e5) - importance 1.491964850888717e-06
    944. feature 858 (product_name_SUPERFOOD BAR - Chia, Jagody Goji, Żurawina) - importance 1.2720272886741435e-06
    945. feature 747 (company_id_5ce2969fc828ba60d8093297) - importance 1.1063661287793536e-06
    946. feature 869 (product_name_SUPERFOOD BAR - Surowe Kakao, orzechy nerkowca) - importance 9.580697902136989e-07
    947. feature 870 (product_name_SUPERFOOD BAR - Morela, Chlorella) - importance 7.958314652095627e-07
    948. feature 741 (company_id_5dc534aa3bf8f15dc97e96d2) - importance 1.677962598573699e-07
    949. feature 742 (company_id_5df098f070ca6078b319a4ab) - importance 6.586324423256537e-08
    950. feature 73 (avg_sum_fv_lag1) - importance 0.0
    951. feature 764 (category_name_nan) - importance 0.0
    952. feature 947 (weekday_nan) - importance 0.0
    953. feature 952 (quarter_nan) - importance 0.0



```python
def auc_score(cls, X_val=X_val, y_val=y_val):
    y_val_pred = cls.predict_proba(X_val)[:,1]
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_val, y_val_pred)
```

## Training on all the features

### More samples in a leaf


```python
trees = ExtraTreesClassifier(n_estimators=100, random_state=0, bootstrap=True,
                             min_samples_leaf=100, max_samples=0.2, n_jobs=-1)
trees.fit(X_train, y_train)
```




    ExtraTreesClassifier(bootstrap=True, max_samples=0.2, min_samples_leaf=100,
                         n_jobs=-1, random_state=0)




```python
auc_score(trees)
```




    0.9053929549124441




```python
auc_score(trees, X_impt_val, y_impt_val)
```




    0.8302667523702393




```python
predict_and_save_ans(trees, "ans_trees")
```

### Less samples in a leaf


```python
trees1 = ExtraTreesClassifier(n_estimators=100, random_state=1, bootstrap=True,
                             min_samples_leaf=30, max_samples=0.2, n_jobs=-1)
trees1.fit(X_train, y_train)
```




    ExtraTreesClassifier(bootstrap=True, max_samples=0.2, min_samples_leaf=30,
                         n_jobs=-1, random_state=1)




```python
auc_score(trees1)
```




    0.9175393820524615




```python
auc_score(trees1, X_impt_val, y_impt_val)
```




    0.8448296641491242




```python
predict_and_save_ans(trees1, "ans_trees1")
```

### Training each estimator on more samples


```python
trees2 = ExtraTreesClassifier(n_estimators=100, random_state=1, bootstrap=True,
                             min_samples_leaf=30, max_samples=0.7, n_jobs=-1)
trees2.fit(X_train, y_train)
```




    ExtraTreesClassifier(bootstrap=True, max_samples=0.7, min_samples_leaf=30,
                         n_jobs=-1, random_state=1)




```python
auc_score(trees2)
```




    0.9300189710101461




```python
auc_score(trees2, X_impt_val, y_impt_val)
```




    0.8582878033103005




```python
predict_and_save_ans(trees2, "ans_trees2")
```

## Fast comparison of feature selections with the same classifiers

I add samples weights at fitting stage in order to focus on important samples (more general ones with unknown locations, products names, categories names or companies ids).

### The most important features with no information about locations
*with the exception of Warsaw and Krakow*


```python
impt_ftrs85 = X_small.keys()[indices[:85]]
not_impt_ftrs85 = X_small.keys()[indices[85:]]
```


```python
impt_ftrs85
```




    Index(['sum_qty', 'week', 'avg_from_paypass_lag1', 'month',
           'available_products', 'days_since_prev_delivery', 'meanLastPeriod_lag1',
           'sdLastPeriod_lag1', 'meanLastPeriod_lag3', 'meanLastPeriod_lag2',
           'avg_transaction_discount_count_lag1', 'sdLastPeriod_lag4',
           'rocPeriod_lag1', 'avg_total_base_lag1', 'sales_since_prev_delivery',
           'maxLastPeriod_lag1', 'meanLastPeriod_lag6', 'meanLastPeriod_lag4',
           'sdLastPeriod_lag7', 'avg_total_lag1', 'avg_from_blik_lag1',
           'meanLastPeriod_lag7', 'sdLastPeriod_lag5', 'sdLastPeriod_lag2',
           'meanLastPeriod_lag5', 'sdLastPeriod_lag3', 'sdLastPeriod_lag6',
           'product_name_Dyniowe curry z indykiem', 'maxLastPeriod_lag7',
           'qty_lag1', 'meanLastPeriod_lag1_lag7_diff', 'qty_lag3', 'qty_lag2',
           'quarter_Q3', 'diff1_lag1', 'qty_lag4', 'weekday_środa',
           'weekday_piątek', 'sol_calk', 'is_delivery_day', 'avg_from_payu_lag1',
           'weekday_czwartek', 'avg_total_to_discount_lag1', 'weekday_niedziela',
           'weekday_sobota', 'weekday_wtorek', 'bialko_calk',
           'weekday_poniedziałek', 'address_city_Warszawa', 'quarter_Q4',
           'maxLastPeriod_lag1_lag7_diff', 'bialko_100', 'cukry_100', 'cukry_calk',
           'qty_lag5', 'energia_100', 'diff1_lag1_lag7_diff',
           'address_city_Kraków', 'diffLagPeriod_lag1_lag7_diff', 'quarter_Q2',
           'roc1_lag1', 'quarter_Q1', 'category_name_Mr Thai', 'energia_calk',
           'weglow_100', 'qty_lag6', 'mean_diff1_lag1_lag7_diff', 'sol_100',
           'weglow_calk', 'weight', 'tluszcz_100', 'qty_lag11', 'qty_lag10',
           'qty_lag13', 'qty_lag12', 'qty_lag7', 'tluszcz_calk',
           'tluszcz_nasyc_100', 'stored_in_fridge', 'tluszcz_nasyc_calk',
           'blonnik_calk', 'blonnik_100', 'qty_lag9', 'diffLagPeriod_lag7',
           'qty_lag14'],
          dtype='object')




```python
indices[:85]
```




    array([ 64,  19,  68,  18,  79,  77,  34,  42,  36,  35,  74,  45,  76,
            72,  78,  52,  39,  37,  48,  70,  67,  40,  46,  43,  38,  44,
            47, 765,  53,  20,  41,  22,  21, 950,  55,  23, 941, 943,  17,
            80,  69, 940,  71, 945, 944, 942,  11, 946, 895, 951,  54,   2,
            16,   4,  24,   7,  57, 896,  60, 948,  75, 949, 761,   6,   3,
            25,  63,  12,  13,  81,  15,  30,  29,  32,  31,  26,  10,   8,
            84,   5,  14,   9,  28,  59,  33])




```python
X_impt_ftrs85_train = X_train.drop(not_impt_ftrs85, axis=1)
X_impt_ftrs85_val = X_val.drop(not_impt_ftrs85, axis=1)
X_impt_ftrs85_impt_val = X_impt_val.drop(not_impt_ftrs85, axis=1)
```


```python
import multiprocessing

multiprocessing.cpu_count()
```




    40




```python
trees_impt85_fast = ExtraTreesClassifier(n_estimators=20, random_state=1, bootstrap=True,
                                         max_features=30, min_samples_leaf=1e-4, max_samples=0.2, n_jobs=30,
                                         verbose=1)
trees_impt85_fast.fit(X_impt_ftrs85_train, y_train, samples_weights[X_impt_ftrs85_train.index,])
```

    [Parallel(n_jobs=30)]: Using backend ThreadingBackend with 30 concurrent workers.
    [Parallel(n_jobs=30)]: Done   3 out of  20 | elapsed: 24.0min remaining: 136.0min
    [Parallel(n_jobs=30)]: Done  20 out of  20 | elapsed: 25.3min finished





    ExtraTreesClassifier(bootstrap=True, max_features=30, max_samples=0.2,
                         min_samples_leaf=0.0001, n_estimators=20, n_jobs=30,
                         random_state=1, verbose=True)




```python
auc_score(trees_impt85_fast, X_impt_ftrs85_val, y_val)
```

    [Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
    [Parallel(n_jobs=20)]: Done   2 out of  20 | elapsed:    0.8s remaining:    7.0s
    [Parallel(n_jobs=20)]: Done  20 out of  20 | elapsed:    0.9s finished





    0.9200937299058559




```python
auc_score(trees_impt85_fast, X_impt_ftrs85_impt_val, y_impt_val)
```

    [Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
    [Parallel(n_jobs=20)]: Done   2 out of  20 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=20)]: Done  20 out of  20 | elapsed:    0.0s finished





    0.8938916117628153



### The most correlated with target value
#### Top 97 features


```python
ftrs_corr97 = corrs[abs(corrs)>=0.04].keys().drop("will_it_sell")
not_ftrs_corr97 = corrs[abs(corrs)<0.04].keys()
```


```python
len(ftrs_corr97.intersection(impt_ftrs85))
```




    63




```python
len(ftrs_corr97.union(impt_ftrs85))
```




    119



The most correlated ones and the most important ones are mostly the same features


```python
X_ftrs_corr97_train = X_train.drop(not_ftrs_corr97, axis=1)
X_ftrs_corr97_val = X_val.drop(not_ftrs_corr97, axis=1)
X_ftrs_corr97_impt_val = X_impt_val.drop(not_ftrs_corr97, axis=1)
```


```python
trees_corr97 = ExtraTreesClassifier(n_estimators=20, random_state=1, bootstrap=True,
                                    max_features=30, min_samples_leaf=1e-4, max_samples=0.2, n_jobs=30,
                                    verbose=1)
trees_corr97.fit(X_ftrs_corr97_train, y_train, samples_weights[X_ftrs_corr97_train.index,])
```

    [Parallel(n_jobs=30)]: Using backend ThreadingBackend with 30 concurrent workers.
    [Parallel(n_jobs=30)]: Done   3 out of  20 | elapsed: 17.3min remaining: 97.9min
    [Parallel(n_jobs=30)]: Done  20 out of  20 | elapsed: 19.0min finished





    ExtraTreesClassifier(bootstrap=True, max_features=30, max_samples=0.2,
                         min_samples_leaf=0.0001, n_estimators=20, n_jobs=30,
                         random_state=1, verbose=True)




```python
auc_score(trees_corr97, X_ftrs_corr97_val, y_val)
```

    [Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
    [Parallel(n_jobs=20)]: Done   2 out of  20 | elapsed:    0.7s remaining:    6.2s
    [Parallel(n_jobs=20)]: Done  20 out of  20 | elapsed:    0.8s finished





    0.916871361045221




```python
auc_score(trees_corr97, X_ftrs_corr97_impt_val, y_impt_val)
```

    [Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
    [Parallel(n_jobs=20)]: Done   2 out of  20 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=20)]: Done  20 out of  20 | elapsed:    0.0s finished





    0.8846717820986661



#### Top 170 features


```python
ftrs_corr170 = corrs[abs(corrs)>=0.025].keys().drop("will_it_sell")
not_ftrs_corr170 = corrs[abs(corrs)<0.025].keys()
```


```python
len(ftrs_corr170)
```




    170




```python
X_ftrs_corr170_train = X_train.drop(not_ftrs_corr170, axis=1)
X_ftrs_corr170_val = X_val.drop(not_ftrs_corr170, axis=1)
X_ftrs_corr170_impt_val = X_impt_val.drop(not_ftrs_corr170, axis=1)
```


```python
trees_corr170 = ExtraTreesClassifier(n_estimators=20, random_state=1, bootstrap=True,
                                    max_features=30, min_samples_leaf=1e-4, max_samples=0.2, n_jobs=30,
                                    verbose=1)
trees_corr170.fit(X_ftrs_corr170_train, y_train, samples_weights[X_ftrs_corr170_train.index,])
```

    [Parallel(n_jobs=30)]: Using backend ThreadingBackend with 30 concurrent workers.
    [Parallel(n_jobs=30)]: Done   3 out of  20 | elapsed: 11.3min remaining: 63.8min
    [Parallel(n_jobs=30)]: Done  20 out of  20 | elapsed: 12.0min finished





    ExtraTreesClassifier(bootstrap=True, max_features=30, max_samples=0.2,
                         min_samples_leaf=0.0001, n_estimators=20, n_jobs=30,
                         random_state=1, verbose=True)




```python
auc_score(trees_corr170, X_ftrs_corr170_val, y_val)
```

    [Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
    [Parallel(n_jobs=20)]: Done   2 out of  20 | elapsed:    0.6s remaining:    5.3s
    [Parallel(n_jobs=20)]: Done  20 out of  20 | elapsed:    0.8s finished





    0.9144692038135699




```python
auc_score(trees_corr170, X_ftrs_corr170_impt_val, y_impt_val)
```

    [Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
    [Parallel(n_jobs=20)]: Done   2 out of  20 | elapsed:    0.0s remaining:    0.1s
    [Parallel(n_jobs=20)]: Done  20 out of  20 | elapsed:    0.0s finished





    0.8906275108468584



## Training on the 85 most important features
### Lots of estimators and high max_samples rate


```python
trees_impt85 = ExtraTreesClassifier(n_estimators=100, random_state=1, bootstrap=True,
                                    max_features=30, min_samples_leaf=1e-4, max_samples=0.7, n_jobs=30,
                                    oob_score=True, verbose=1)
trees_impt85.fit(X_impt_ftrs85_train, y_train, samples_weights[X_impt_ftrs85_train.index,])
```

    [Parallel(n_jobs=30)]: Using backend ThreadingBackend with 30 concurrent workers.
    [Parallel(n_jobs=30)]: Done 100 out of 100 | elapsed: 447.6min finished





    ExtraTreesClassifier(bootstrap=True, max_features=30, max_samples=0.7,
                         min_samples_leaf=0.0001, n_jobs=30, oob_score=True,
                         random_state=1, verbose=True)




```python
auc_score(trees_impt85, X_impt_ftrs85_val, y_val)
```

    [Parallel(n_jobs=30)]: Using backend ThreadingBackend with 30 concurrent workers.
    [Parallel(n_jobs=30)]: Done 100 out of 100 | elapsed:    4.8s finished





    0.9301058309562702




```python
auc_score(trees_impt85, X_impt_ftrs85_impt_val, y_impt_val)
```

    [Parallel(n_jobs=30)]: Using backend ThreadingBackend with 30 concurrent workers.
    [Parallel(n_jobs=30)]: Done 100 out of 100 | elapsed:    0.0s finished





    0.917754700305319




```python
X_impt_ftrs85_test = X_test.drop(not_impt_ftrs85, axis=1)
```


```python
predict_and_save_ans(trees_impt85, "ans_trees_impt85", X_impt_ftrs85_test)
```

    [Parallel(n_jobs=30)]: Using backend ThreadingBackend with 30 concurrent workers.
    [Parallel(n_jobs=30)]: Done 100 out of 100 | elapsed:   28.4s finished


### A few estimators with high max_samples rate but each feature is taken into account when splitting at every level


```python
trees_impt85_best_splitting = ExtraTreesClassifier(n_estimators=10, random_state=1, bootstrap=True,
                                    max_features=85, min_samples_leaf=1e-4, max_samples=0.95, n_jobs=30,
                                    oob_score=True, verbose=1)
trees_impt85_best_splitting.fit(X_impt_ftrs85_train, y_train, samples_weights[X_impt_ftrs85_train.index,])
```

    [Parallel(n_jobs=30)]: Using backend ThreadingBackend with 30 concurrent workers.
    [Parallel(n_jobs=30)]: Done   6 out of  10 | elapsed: 164.4min remaining: 109.6min
    [Parallel(n_jobs=30)]: Done  10 out of  10 | elapsed: 167.0min finished
    /home/krzpiesiewicz/venv/lib/python3.8/site-packages/sklearn/ensemble/_forest.py:541: UserWarning: Some inputs do not have OOB scores. This probably means too few trees were used to compute any reliable oob estimates.
      warn("Some inputs do not have OOB scores. "
    /home/krzpiesiewicz/venv/lib/python3.8/site-packages/sklearn/ensemble/_forest.py:545: RuntimeWarning: invalid value encountered in true_divide
      decision = (predictions[k] /





    ExtraTreesClassifier(bootstrap=True, max_features=85, max_samples=0.95,
                         min_samples_leaf=0.0001, n_estimators=10, n_jobs=30,
                         oob_score=True, random_state=1, verbose=True)




```python
auc_score(trees_impt85_best_splitting, X_impt_ftrs85_val, y_val)
```

    [Parallel(n_jobs=10)]: Using backend ThreadingBackend with 10 concurrent workers.
    [Parallel(n_jobs=10)]: Done   2 out of  10 | elapsed:    0.4s remaining:    1.6s
    [Parallel(n_jobs=10)]: Done  10 out of  10 | elapsed:    0.5s finished





    0.9374549951022586




```python
auc_score(trees_impt85_best_splitting, X_impt_ftrs85_impt_val, y_impt_val)
```

    [Parallel(n_jobs=10)]: Using backend ThreadingBackend with 10 concurrent workers.
    [Parallel(n_jobs=10)]: Done   2 out of  10 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=10)]: Done  10 out of  10 | elapsed:    0.0s finished





    0.9264924473726499




```python
predict_and_save_ans(trees_impt85_best_splitting, "ans_trees_impt85_best_splitting", X_impt_ftrs85_test)
```

    [Parallel(n_jobs=10)]: Using backend ThreadingBackend with 10 concurrent workers.
    [Parallel(n_jobs=10)]: Done   2 out of  10 | elapsed:    2.9s remaining:   11.5s
    [Parallel(n_jobs=10)]: Done  10 out of  10 | elapsed:    3.0s finished


### Let's fit an estimator for predicting sales at known locations
Its test predictions for known locations could be combined with predictions of more general estimator for unknown locations.

I add pos_ids features to the 85 most important ones.


```python
impt_ftrs455 = X_small.keys()[indices[:455]]
not_impt_ftrs455 = X_small.keys()[indices[455:]]
```


```python
impt_ftrs455
```




    Index(['sum_qty', 'week', 'avg_from_paypass_lag1', 'month',
           'available_products', 'days_since_prev_delivery', 'meanLastPeriod_lag1',
           'sdLastPeriod_lag1', 'meanLastPeriod_lag3', 'meanLastPeriod_lag2',
           ...
           'company_id_5bbdc6a015c00937a0810fc7',
           'pos_id_5a44d792f9cdb15f7f7d9d49',
           'company_id_5b35c7a90663ab48e335be72',
           'pos_id_5b713f4fd8f80d6ebf31d49f', 'pos_id_5ba0d8dd01f82e03b41313a0',
           'company_id_5bd018b451478a069f6c34fa', 'product_name_Gnocchi primavera',
           'pos_id_5c94d70e1032d642b63487c2',
           'company_id_5c8f6c59c667da0ef079d9fc',
           'pos_id_5cc040375e70463ad9d520dd'],
          dtype='object', length=455)




```python
X_impt_ftrs455_train = X_train.drop(not_impt_ftrs455, axis=1)
X_impt_ftrs455_val = X_val.drop(not_impt_ftrs455, axis=1)
X_impt_ftrs455_impt_val = X_impt_val.drop(not_impt_ftrs455, axis=1)
```


```python
trees_impt455 = ExtraTreesClassifier(n_estimators=20, random_state=2, bootstrap=True,
                                     max_features=30, min_samples_leaf=1e-5, max_samples=0.7, n_jobs=30,
                                     verbose=1)
trees_impt455.fit(X_impt_ftrs455_train, y_train)
```

    [Parallel(n_jobs=30)]: Using backend ThreadingBackend with 30 concurrent workers.
    [Parallel(n_jobs=30)]: Done   3 out of  20 | elapsed: 57.7min remaining: 326.8min
    [Parallel(n_jobs=30)]: Done  20 out of  20 | elapsed: 60.2min finished





    ExtraTreesClassifier(bootstrap=True, max_features=30, max_samples=0.7,
                         min_samples_leaf=1e-05, n_estimators=20, n_jobs=30,
                         random_state=2, verbose=True)




```python
auc_score(trees_impt455, X_impt_ftrs455_val, y_val)
```

    [Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
    [Parallel(n_jobs=20)]: Done   2 out of  20 | elapsed:    1.5s remaining:   13.9s
    [Parallel(n_jobs=20)]: Done  20 out of  20 | elapsed:    1.6s finished





    0.9314723596253695




```python
auc_score(trees_impt455, X_impt_ftrs455_impt_val, y_impt_val)
```

    [Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
    [Parallel(n_jobs=20)]: Done   2 out of  20 | elapsed:    0.0s remaining:    0.1s
    [Parallel(n_jobs=20)]: Done  20 out of  20 | elapsed:    0.0s finished





    0.8778924955809095



### Let's try with Gradient Boosting


```python
from sklearn.ensemble import GradientBoostingClassifier
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=8, min_samples_leaf=1e-4,
                                    subsample=0.01, validation_fraction=0.001,
                                    n_iter_no_change=3, random_state=3, verbose=3)
gb_clf.fit(X_impt_ftrs85_train, y_train)
```

          Iter       Train Loss      OOB Improve   Remaining Time 
             1           0.6617           0.0644           33.33m
             2           0.6287           0.0392           29.90m
             3           0.6041           0.0296           29.57m
             4           0.5772           0.0234           29.05m
             5           0.5567           0.0187           28.42m
             6           0.5374           0.0153           28.16m
             7           0.5212           0.0131           27.94m
             8           0.5193           0.0116           27.76m
             9           0.5130           0.0086           27.19m
            10           0.5037           0.0072           26.70m
            11           0.4985           0.0073           26.40m
            12           0.4888           0.0065           26.05m
            13           0.4850           0.0055           25.69m
            14           0.4833           0.0049           25.31m
            15           0.4709           0.0043           24.91m
            16           0.4638           0.0039           24.60m
            17           0.4640           0.0030           24.10m
            18           0.4633           0.0028           23.65m
            19           0.4560           0.0033           23.35m
            20           0.4456           0.0027           23.06m
            21           0.4501           0.0019           22.63m
            22           0.4474           0.0026           22.32m
            23           0.4413           0.0018           21.98m
            24           0.4443           0.0023           21.64m
            25           0.4513           0.0019           21.41m
            26           0.4445           0.0017           21.20m
            27           0.4447           0.0011           20.88m
            28           0.4377           0.0019           20.68m
            29           0.4346           0.0006           20.32m
            30           0.4406           0.0010           20.04m
            31           0.4352           0.0010           19.71m
            32           0.4315           0.0011           19.48m
            33           0.4300           0.0019           19.27m
            34           0.4381           0.0013           19.04m
            35           0.4388           0.0014           18.78m
            36           0.4266           0.0006           18.43m
            37           0.4304           0.0012           18.14m
            38           0.4191           0.0005           17.78m
            39           0.4266           0.0012           17.46m
            40           0.4247           0.0011           17.22m
            41           0.4224           0.0006           16.89m
            42           0.4296           0.0013           16.62m
            43           0.4281           0.0004           16.28m
            44           0.4230           0.0006           15.94m
            45           0.4274           0.0012           15.66m
            46           0.4194           0.0016           15.40m
            47           0.4301           0.0008           15.11m
            48           0.4262           0.0003           14.79m
            49           0.4154           0.0003           14.48m
            50           0.4151           0.0010           14.22m
            51           0.4196           0.0003           13.94m
            52           0.4207           0.0004           13.69m
            53           0.4138           0.0004           13.38m
            54           0.4143           0.0010           13.11m
            55           0.4185           0.0004           12.83m
            56           0.4233           0.0001           12.52m
            57           0.4176           0.0006           12.24m
            58           0.4068           0.0008           11.96m
            59           0.4087           0.0003           11.65m
            60           0.4108           0.0003           11.34m
            61           0.4080           0.0008           11.05m
            62           0.4205           0.0006           10.79m
            63           0.4081           0.0006           10.52m
            64           0.4193           0.0004           10.22m
            65           0.4076           0.0005            9.94m
            66           0.4084           0.0003            9.64m
            67           0.4043           0.0003            9.34m
            68           0.4130           0.0001            9.05m
            69           0.4072           0.0002            8.75m
            70           0.4085           0.0001            8.47m
            71           0.4058           0.0001            8.17m
            72           0.4232           0.0003            7.90m
            73           0.4122           0.0005            7.63m
            74           0.4102           0.0002            7.35m
            75           0.3991           0.0001            7.06m
            76           0.4070           0.0001            6.77m
            77           0.4090           0.0002            6.48m
            78           0.4119           0.0002            6.19m
            79           0.4104           0.0002            5.91m
            80           0.4091           0.0001            5.62m
            81           0.4162           0.0006            5.35m
            82           0.4179           0.0004            5.06m
            83           0.4214           0.0002            4.78m
            84           0.4069           0.0002            4.49m
            85           0.4065           0.0003            4.22m
            86           0.4128           0.0001            3.93m
            87           0.4052           0.0001            3.64m





    GradientBoostingClassifier(max_depth=8, min_samples_leaf=0.0001,
                               n_iter_no_change=3, random_state=3, subsample=0.01,
                               validation_fraction=0.001, verbose=3)




```python
auc_score(gb_clf, X_impt_ftrs85_val, y_val)
```




    0.9233331500706501




```python
auc_score(gb_clf, X_impt_ftrs85_impt_val, y_impt_val)
```




    0.9132552627350152




```python
predict_and_save_ans(gb_clf, "ans_gb_clf", X_impt_ftrs85_test)
```

## Let's consider time dependecies problem
Relating to features importance ranking, time features are at high positions. Note that test set samples are only from part of the year. I try to reach better generality by removing time/season information (with the except of weekdays).


```python
np.sort(pd.unique(df.month))
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])




```python
np.sort(pd.unique(X_test.month).to_dense())
```




    array([ 1.,  2., 11., 12.])




```python
np.sort(pd.unique(df.week))
```




    array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17,
           18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
           35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
           52])




```python
np.sort(pd.unique(X_test.week).to_dense())
```




    array([ 1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9., 48., 49., 50., 51.,
           52.])




```python
time_ftrs = ['quarter_Q1', 'quarter_Q2', 'quarter_Q3', 'quarter_Q4', 'month', 'week']
X_impt_ftrs85_no_time_train = X_impt_ftrs85_train.drop(time_ftrs, axis=1)
X_impt_ftrs85_no_time_val = X_impt_ftrs85_val.drop(time_ftrs, axis=1)
X_impt_ftrs85_no_time_impt_val = X_impt_ftrs85_impt_val.drop(time_ftrs, axis=1)
```


```python
X_impt_ftrs85_no_time_val.keys()
```




    Index(['bialko_100', 'weglow_100', 'cukry_calk', 'tluszcz_nasyc_calk',
           'energia_calk', 'energia_100', 'tluszcz_nasyc_100', 'blonnik_100',
           'tluszcz_calk', 'bialko_calk', 'sol_100', 'weglow_calk', 'blonnik_calk',
           'tluszcz_100', 'cukry_100', 'sol_calk', 'qty_lag1', 'qty_lag2',
           'qty_lag3', 'qty_lag4', 'qty_lag5', 'qty_lag6', 'qty_lag7', 'qty_lag9',
           'qty_lag10', 'qty_lag11', 'qty_lag12', 'qty_lag13', 'qty_lag14',
           'meanLastPeriod_lag1', 'meanLastPeriod_lag2', 'meanLastPeriod_lag3',
           'meanLastPeriod_lag4', 'meanLastPeriod_lag5', 'meanLastPeriod_lag6',
           'meanLastPeriod_lag7', 'meanLastPeriod_lag1_lag7_diff',
           'sdLastPeriod_lag1', 'sdLastPeriod_lag2', 'sdLastPeriod_lag3',
           'sdLastPeriod_lag4', 'sdLastPeriod_lag5', 'sdLastPeriod_lag6',
           'sdLastPeriod_lag7', 'maxLastPeriod_lag1', 'maxLastPeriod_lag7',
           'maxLastPeriod_lag1_lag7_diff', 'diff1_lag1', 'diff1_lag1_lag7_diff',
           'diffLagPeriod_lag7', 'diffLagPeriod_lag1_lag7_diff',
           'mean_diff1_lag1_lag7_diff', 'sum_qty', 'avg_from_blik_lag1',
           'avg_from_paypass_lag1', 'avg_from_payu_lag1', 'avg_total_lag1',
           'avg_total_to_discount_lag1', 'avg_total_base_lag1',
           'avg_transaction_discount_count_lag1', 'roc1_lag1', 'rocPeriod_lag1',
           'days_since_prev_delivery', 'sales_since_prev_delivery',
           'available_products', 'is_delivery_day', 'weight', 'stored_in_fridge',
           'category_name_Mr Thai', 'product_name_Dyniowe curry z indykiem',
           'address_city_Warszawa', 'address_city_Kraków', 'weekday_czwartek',
           'weekday_środa', 'weekday_wtorek', 'weekday_piątek', 'weekday_sobota',
           'weekday_niedziela', 'weekday_poniedziałek'],
          dtype='object')



### Benchmark of ExtraTreesClassifiers with season features and without
#### Benchmark on my validations sets
I will test dropping time information with the same fast classifier which I used with `X_impt_ftrs85` for the first time when it scored `0.92009` and `0.89389` AUC on my validation sets.


```python
trees_impt85_no_time_fast = ExtraTreesClassifier(n_estimators=20, random_state=1, bootstrap=True,
                                         max_features=30, min_samples_leaf=1e-4, max_samples=0.2, n_jobs=30,
                                         verbose=2)
trees_impt85_no_time_fast.fit(X_impt_ftrs85_no_time_train, y_train,
                              samples_weights[X_impt_ftrs85_no_time_train.index,])
```

    [Parallel(n_jobs=30)]: Using backend ThreadingBackend with 30 concurrent workers.


    building tree 1 of 20
    building tree 2 of 20
    building tree 3 of 20
    building tree 4 of 20
    building tree 5 of 20
    building tree 6 of 20
    building tree 7 of 20
    building tree 8 of 20
    building tree 9 of 20
    building tree 10 of 20building tree 11 of 20
    building tree 12 of 20
    building tree 13 of 20
    
    building tree 14 of 20
    building tree 15 of 20
    building tree 16 of 20
    building tree 17 of 20
    building tree 18 of 20
    building tree 19 of 20
    building tree 20 of 20


    [Parallel(n_jobs=30)]: Done   5 out of  20 | elapsed: 23.4min remaining: 70.3min
    [Parallel(n_jobs=30)]: Done  16 out of  20 | elapsed: 24.2min remaining:  6.1min
    [Parallel(n_jobs=30)]: Done  20 out of  20 | elapsed: 24.3min finished





    ExtraTreesClassifier(bootstrap=True, max_features=30, max_samples=0.2,
                         min_samples_leaf=0.0001, n_estimators=20, n_jobs=30,
                         random_state=1, verbose=2)




```python
auc_score(trees_impt85_no_time_fast, X_impt_ftrs85_no_time_val, y_val)
```

    [Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
    [Parallel(n_jobs=20)]: Done   3 out of  20 | elapsed:    0.8s remaining:    4.3s
    [Parallel(n_jobs=20)]: Done  14 out of  20 | elapsed:    0.8s remaining:    0.3s
    [Parallel(n_jobs=20)]: Done  20 out of  20 | elapsed:    0.8s finished





    0.9036358011163456




```python
auc_score(trees_impt85_no_time_fast, X_impt_ftrs85_no_time_impt_val, y_impt_val)
```

    [Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
    [Parallel(n_jobs=20)]: Done   3 out of  20 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=20)]: Done  14 out of  20 | elapsed:    0.0s remaining:    0.0s
    [Parallel(n_jobs=20)]: Done  20 out of  20 | elapsed:    0.0s finished





    0.8815985055439498




```python
X_impt_ftrs85_no_time_test = X_impt_ftrs85_test.drop(time_ftrs, axis=1)
```


```python
predict_and_save_ans(trees_impt85_no_time_fast, "trees_impt85_no_time_fast", X_impt_ftrs85_no_time_test)
```

    [Parallel(n_jobs=20)]: Using backend ThreadingBackend with 20 concurrent workers.
    [Parallel(n_jobs=20)]: Done   3 out of  20 | elapsed:    5.5s remaining:   31.3s
    [Parallel(n_jobs=20)]: Done  14 out of  20 | elapsed:    5.8s remaining:    2.5s
    [Parallel(n_jobs=20)]: Done  20 out of  20 | elapsed:    5.8s finished


These results are worse but both training and validation sets contain samples from each season. So the trees in previous model could be better fitted to specific time I should have split samples acca validation.
#### Benchmark on train and validation sets split by season


```python
df85_impt_ftrs85 = df.drop(not_impt_ftrs85, axis=1)

samples_months_3to10_indices = np.where(df85_impt_ftrs85.month.isin(range(3, 11)))[0]

samples_impt_ftrs85_months_3to10 = df85_impt_ftrs85.loc[samples_months_3to10_indices, ]
samples_impt_ftrs85_months_11to2 = df85_impt_ftrs85.drop(samples_months_3to10_indices, axis=0)

X_impt_ftrs85_months_3to10, y_months_3to10 = x_y_split(samples_impt_ftrs85_months_3to10)
X_impt_ftrs85_months_11to2, y_months_11to2 = x_y_split(samples_impt_ftrs85_months_11to2)
```


```python
X_impt_ftrs85_months_3to10 = as_sparse(X_impt_ftrs85_months_3to10)
y_months_3to10 = as_sparse(y_months_3to10)
X_impt_ftrs85_months_11to2 = as_sparse(X_impt_ftrs85_months_11to2)
y_months_11to2 = as_sparse(y_months_11to2)
```


```python
time_ftrs = ['quarter_Q1', 'quarter_Q2', 'quarter_Q3', 'quarter_Q4', 'month', 'week']
```


```python
X_impt_ftrs85_no_time_months_3to10 = X_impt_ftrs85_months_3to10.drop(time_ftrs, axis=1)
X_impt_ftrs85_no_time_months_11to2 = X_impt_ftrs85_months_11to2.drop(time_ftrs, axis=1)
```


```python
trees_impt85_very_fast = ExtraTreesClassifier(n_estimators=10, random_state=1, bootstrap=True,
                                              max_features=30, min_samples_leaf=1e-4, max_samples=0.1,
                                              n_jobs=30, verbose=2)
trees_impt85_very_fast.fit(X_impt_ftrs85_months_3to10, y_months_3to10,
                           samples_weights[X_impt_ftrs85_months_3to10.index,])
```

    [Parallel(n_jobs=30)]: Using backend ThreadingBackend with 30 concurrent workers.


    building tree 1 of 10building tree 2 of 10
    
    building tree 3 of 10
    building tree 4 of 10
    building tree 5 of 10
    building tree 6 of 10
    building tree 7 of 10
    building tree 8 of 10
    building tree 9 of 10
    building tree 10 of 10


    [Parallel(n_jobs=30)]: Done   5 out of  10 | elapsed:  4.4min remaining:  4.4min
    [Parallel(n_jobs=30)]: Done  10 out of  10 | elapsed:  4.4min finished





    ExtraTreesClassifier(bootstrap=True, max_features=30, max_samples=0.1,
                         min_samples_leaf=0.0001, n_estimators=10, n_jobs=30,
                         random_state=1, verbose=2)




```python
auc_score(trees_impt85_very_fast, X_impt_ftrs85_months_11to2, y_months_11to2)
```

    [Parallel(n_jobs=10)]: Using backend ThreadingBackend with 10 concurrent workers.
    [Parallel(n_jobs=10)]: Done   3 out of  10 | elapsed:    1.4s remaining:    3.4s
    [Parallel(n_jobs=10)]: Done  10 out of  10 | elapsed:    1.5s finished





    0.8418574024096729




```python
trees_impt85_no_time_very_fast = ExtraTreesClassifier(n_estimators=10, random_state=1, bootstrap=True,
                                                      max_features=30, min_samples_leaf=1e-4, max_samples=0.1,
                                                      n_jobs=30, verbose=2)
trees_impt85_no_time_very_fast.fit(X_impt_ftrs85_no_time_months_3to10, y_months_3to10,
                                   samples_weights[X_impt_ftrs85_no_time_months_3to10.index,])
```

    [Parallel(n_jobs=30)]: Using backend ThreadingBackend with 30 concurrent workers.


    building tree 1 of 10
    building tree 2 of 10
    building tree 3 of 10
    building tree 4 of 10
    building tree 5 of 10
    building tree 6 of 10
    building tree 7 of 10
    building tree 8 of 10
    building tree 9 of 10
    building tree 10 of 10


    [Parallel(n_jobs=30)]: Done   5 out of  10 | elapsed:  4.3min remaining:  4.3min
    [Parallel(n_jobs=30)]: Done  10 out of  10 | elapsed:  4.4min finished





    ExtraTreesClassifier(bootstrap=True, max_features=30, max_samples=0.1,
                         min_samples_leaf=0.0001, n_estimators=10, n_jobs=30,
                         random_state=1, verbose=2)




```python
auc_score(trees_impt85_no_time_very_fast, X_impt_ftrs85_no_time_months_11to2, y_months_11to2)
```

    [Parallel(n_jobs=10)]: Using backend ThreadingBackend with 10 concurrent workers.
    [Parallel(n_jobs=10)]: Done   3 out of  10 | elapsed:    1.4s remaining:    3.2s
    [Parallel(n_jobs=10)]: Done  10 out of  10 | elapsed:    1.4s finished





    0.8473781705546937



Training with no season features outperformed the one with them
## ExtraTreesClassifier and GradientBoosting with no season features trained on all data
### ExtraTreesClassifier


```python
not_impt_ftrs85_no_time = not_impt_ftrs85.append(pd.Index(time_ftrs))
```


```python
X, y = x_y_split(df.drop(not_impt_ftrs85_no_time, axis=1))
```


```python
X = as_sparse(X)
y = as_sparse(y)
```


```python
trees_impt85_no_time_best_splitting = ExtraTreesClassifier(n_estimators=10, random_state=1, bootstrap=True,
                                                           max_features=79, min_samples_leaf=1e-4,
                                                           max_samples=0.95, n_jobs=40, verbose=2)
trees_impt85_no_time_best_splitting.fit(X, y, samples_weights[X.index,])
```

    [Parallel(n_jobs=40)]: Using backend ThreadingBackend with 40 concurrent workers.


    building tree 1 of 10building tree 2 of 10
    
    building tree 3 of 10
    building tree 4 of 10
    building tree 5 of 10
    building tree 6 of 10
    building tree 7 of 10
    building tree 8 of 10
    building tree 9 of 10
    building tree 10 of 10


    [Parallel(n_jobs=40)]: Done   3 out of  10 | elapsed: 99.2min remaining: 231.4min
    [Parallel(n_jobs=40)]: Done  10 out of  10 | elapsed: 103.7min finished





    ExtraTreesClassifier(bootstrap=True, max_features=79, max_samples=0.95,
                         min_samples_leaf=0.0001, n_estimators=10, n_jobs=40,
                         random_state=1, verbose=2)




```python
X_test = X_test.drop(not_impt_ftrs85_no_time, axis=1)
```


```python
predict_and_save_ans(trees_impt85_no_time_best_splitting, "ans_trees_impt85_no_time_best_splitting", X_test)
```

    [Parallel(n_jobs=10)]: Using backend ThreadingBackend with 10 concurrent workers.
    [Parallel(n_jobs=10)]: Done   3 out of  10 | elapsed:    3.8s remaining:    8.9s
    [Parallel(n_jobs=10)]: Done  10 out of  10 | elapsed:    3.9s finished


### GradientBoosting


```python
from sklearn.ensemble import GradientBoostingClassifier
gb_clf_no_time = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=8,
                                            min_samples_leaf=1e-4,
                                            subsample=0.01, validation_fraction=0.001,
                                            n_iter_no_change=3, random_state=3, verbose=3)
gb_clf_no_time.fit(X, y)
```

          Iter       Train Loss      OOB Improve   Remaining Time 
             1           0.6674           0.0645           47.86m
             2           0.6276           0.0398           40.72m
             3           0.6057           0.0289           38.36m
             4           0.5750           0.0225           37.08m
             5           0.5641           0.0180           36.08m
             6           0.5410           0.0153           35.19m
             7           0.5314           0.0126           34.39m
             8           0.5122           0.0107           33.68m
             9           0.5075           0.0093           33.02m
            10           0.4979           0.0080           32.41m
            11           0.4934           0.0070           31.93m
            12           0.4910           0.0056           31.37m
            13           0.4863           0.0046           30.78m
            14           0.4795           0.0044           30.38m
            15           0.4741           0.0035           29.92m
            16           0.4799           0.0042           29.62m
            17           0.4725           0.0030           29.17m
            18           0.4737           0.0030           28.83m
            19           0.4630           0.0024           28.46m
            20           0.4610           0.0026           28.17m
            21           0.4633           0.0024           27.91m
            22           0.4602           0.0013           27.44m
            23           0.4538           0.0013           27.00m
            24           0.4591           0.0018           26.54m
            25           0.4502           0.0019           26.21m
            26           0.4488           0.0017           25.80m
            27           0.4573           0.0017           25.47m
            28           0.4522           0.0011           25.06m
            29           0.4491           0.0009           24.67m
            30           0.4541           0.0012           24.30m
            31           0.4417           0.0007           23.90m
            32           0.4460           0.0009           23.60m
            33           0.4408           0.0013           23.27m
            34           0.4450           0.0009           22.94m
            35           0.4412           0.0004           22.60m
            36           0.4443           0.0009           22.30m
            37           0.4469           0.0009           22.03m
            38           0.4310           0.0003           21.66m
            39           0.4396           0.0005           21.30m
            40           0.4396           0.0009           20.95m
            41           0.4416           0.0007           20.62m
            42           0.4375           0.0004           20.24m
            43           0.4372           0.0003           19.85m
            44           0.4385           0.0004           19.47m
            45           0.4382           0.0003           19.12m
            46           0.4284           0.0001           18.84m
            47           0.4352           0.0002           18.46m
            48           0.4363           0.0003           18.11m
            49           0.4347           0.0004           17.75m
            50           0.4304           0.0008           17.42m
            51           0.4368           0.0005           17.09m
            52           0.4322           0.0004           16.80m
            53           0.4322           0.0004           16.46m
            54           0.4352           0.0002           16.13m
            55           0.4380           0.0004           15.78m
            56           0.4379           0.0006           15.46m
            57           0.4344           0.0004           15.11m
            58           0.4347           0.0002           14.74m
            59           0.4346           0.0002           14.38m
            60           0.4372           0.0001           14.00m
            61           0.4206           0.0000           13.63m
            62           0.4345           0.0001           13.25m
            63           0.4370           0.0004           12.91m
            64           0.4240           0.0003           12.57m
            65           0.4349           0.0002           12.22m
            66           0.4343           0.0001           11.85m
            67           0.4390           0.0002           11.49m
            68           0.4365           0.0002           11.14m
            69           0.4385           0.0001           10.78m
            70           0.4331           0.0001           10.42m
            71           0.4286           0.0003           10.07m
            72           0.4301           0.0002            9.72m
            73           0.4285           0.0002            9.38m
            74           0.4361           0.0002            9.02m
            75           0.4303           0.0002            8.68m
            76           0.4306           0.0001            8.32m
            77           0.4362           0.0003            7.97m
            78           0.4231           0.0002            7.61m
            79           0.4279           0.0001            7.25m
            80           0.4275           0.0001            6.90m
            81           0.4305           0.0001            6.55m
            82           0.4239           0.0002            6.20m
            83           0.4297           0.0002            5.85m
            84           0.4283           0.0003            5.51m
            85           0.4255           0.0002            5.17m
            86           0.4263           0.0002            4.83m
            87           0.4211           0.0001            4.48m
            88           0.4239           0.0003            4.13m
            89           0.4354           0.0001            3.79m
            90           0.4291           0.0001            3.44m
            91           0.4282           0.0001            3.09m
            92           0.4305           0.0001            2.75m
            93           0.4205           0.0000            2.40m
            94           0.4291           0.0001            2.05m
            95           0.4201           0.0000            1.71m
            96           0.4233           0.0002            1.36m
            97           0.4247           0.0002            1.02m
            98           0.4213           0.0001           40.85s
            99           0.4316           0.0002           20.42s
           100           0.4195           0.0001            0.00s





    GradientBoostingClassifier(max_depth=8, min_samples_leaf=0.0001,
                               n_iter_no_change=3, random_state=3, subsample=0.01,
                               validation_fraction=0.001, verbose=3)




```python
predict_and_save_ans(gb_clf_no_time, "ans_gb_clf_no_time", X_test)
```

## Combining multiple answers into a final one


```python
def combine_answers(answers, weights, masks):
    cb_ans = np.zeros(answers[0].shape)
    weights_sum = np.zeros(answers[0].shape)
    for ans, w, mask in zip(answers, weights, masks):
        cb_ans += ans * w * mask
        weights_sum += w * mask
    return cb_ans / weights_sum
```


```python
def load_answer(file_name):
    ans = numpy.empty(X_test.shape[0])
    with open(file_name, "r") as f:
        for i, p in enumerate(f.readlines()):
            ans[i] = p
```

Finally I decided not to combine multiple answers...

## Summary - Lessons learned <a class="anchor" id="Summary-lessons-learned"></a>
To make a long story short I conclude that most important thing is creating not overfitted models that uses not too many features so that training samples could not be too finely split by irrelevant ones. Of course, there is also a need to set minimum number of samples required to be at a leaf node.

Due to the contest site evaluation my gradient boosting models outperforms my extra trees classifiers. My test does not confirm that but it is not trustworthy because my validation set is too similar to the training one. Another reason could be the fact I set max depth for gradient boosting trees.
