***
# Supervised Learning : Exploring Penalized Models for Predicting Numeric Responses

***
### [**John Pauline Pineda**](https://github.com/JohnPaulinePineda) <br> <br> *November 16, 2023*
***

* [**1. Table of Contents**](#TOC)
    * [1.1 Data Background](#1.1)
    * [1.2 Data Description](#1.2)
    * [1.3 Data Quality Assessment](#1.3)
    * [1.4 Data Preprocessing](#1.4)
        * [1.4.1 Data Cleaning](#1.4.1)
        * [1.4.2 Missing Data Imputation](#1.4.2)
        * [1.4.3 Outlier Treatment](#1.4.3)
        * [1.4.4 Collinearity](#1.4.4)
        * [1.4.5 Shape Transformation](#1.4.5)
        * [1.4.6 Centering and Scaling](#1.4.6)
        * [1.4.7 Data Encoding](#1.4.7)
        * [1.4.8 Preprocessed Data Description](#1.4.8)
    * [1.5 Data Exploration](#1.5)
        * [1.5.1 Exploratory Data Analysis](#1.5.1)
        * [1.5.2 Hypothesis Testing](#1.5.2)
    * [1.6 Model Development](#1.6)
        * [1.6.1 Premodelling Data Description](#1.6.1)
        * [1.6.2 Linear Regression](#1.6.2)
        * [1.6.3 Polynomial Regression](#1.6.3)
        * [1.6.4 Ridge Regression](#1.6.4)
        * [1.6.5 Least Absolute Shrinkage and Selection Operator Regression](#1.6.5)
        * [1.6.6 Elastic Net Regression](#1.6.6)
    * [1.7 Consolidated Findings](#1.7)   
* [**2. Summary**](#Summary)   
* [**3. References**](#References)

***

# 1. Table of Contents <a class="anchor" id="TOC"></a>

This project implements different penalized regression modelling procedures for numeric responses using various helpful packages in <mark style="background-color: #CCECFF"><b>Python</b></mark>. Models applied in the analysis to predict numeric responses included the **Linear Regression**, **Polynomial Regression**, **Ridge Regression**, **Least Absolute Shrinkage and Selection Operator Regression** and **Elastic Net Regression** algorithms. The resulting predictions derived from the candidate models were evaluated in terms of their model fit using the r-squared, mean squared error (MSE) and mean absolute error (MAE) metrics. All results were consolidated in a [<span style="color: #FF0000"><b>Summary</b></span>](#Summary) presented at the end of the document.

[Penalized regression](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) is a form of variable selection method aimed at reducing model complexity by seeking a smaller subset of informative variables. This process keeps all the predictor variables in the model but constrains (regularizes) the regression coefficients by shrinking them toward zero, in addition to setting some coefficients exactly equal to zero, and is directly built as extensions of standard regression coefficient estimation by incorporating a penalty in the loss function. The algorithms applied in this study attempt to evaluate various penalties for over-confidence in the parameter estimates and accept a slightly worse fit in order to have a more parsimonious model.

## 1.1. Data Background <a class="anchor" id="1.1"></a>

Datasets used for the analysis were separately gathered and consolidated from various sources including: 
1. Cancer Rates from [World Population Review](https://worldpopulationreview.com/country-rankings/cancer-rates-by-country)
2. Social Protection and Labor Indicator from [World Bank](https://data.worldbank.org/topic/social-protection-and-labor?view=chart)
3. Education Indicator from [World Bank](https://data.worldbank.org/topic/education?view=chart)
4. Economy and Growth Indicator from [World Bank](https://data.worldbank.org/topic/economy-and-growth?view=chart)
5. Environment Indicator from [World Bank](https://data.worldbank.org/topic/environment?view=chart)
6. Climate Change Indicator from [World Bank](https://data.worldbank.org/topic/climate-change?view=chart)
7. Agricultural and Rural Development Indicator from [World Bank](https://data.worldbank.org/topic/agriculture-and-rural-development?view=chart)
8. Social Development Indicator from [World Bank](https://data.worldbank.org/topic/social-development?view=chart)
9. Health Indicator from [World Bank](https://data.worldbank.org/topic/health?view=chart)
10. Science and Technology Indicator from [World Bank](https://data.worldbank.org/topic/science-and-technology?view=chart)
11. Urban Development Indicator from [World Bank](https://data.worldbank.org/topic/urban-development?view=chart)
12. Human Development Indices from [Human Development Reports](https://hdr.undp.org/data-center/human-development-index#/indicies/HDI)
13. Environmental Performance Indices from [Yale Center for Environmental Law and Policy](https://epi.yale.edu/epi-results/2022/component/epi)

This study hypothesized that various global development indicators and indices influence cancer rates across countries.

The target variable for the study is:
* <span style="color: #FF0000">CANRAT</span> - Age-standardized cancer rates, per 100K population (2022)

The predictor variables for the study are:
* <span style="color: #FF0000">GDPPER</span> - GDP per person employed, current US Dollars (2020)
* <span style="color: #FF0000">URBPOP</span> - Urban population, % of total population (2020)
* <span style="color: #FF0000">PATRES</span> - Patent applications by residents, total count (2020)
* <span style="color: #FF0000">RNDGDP</span> - Research and development expenditure, % of GDP (2020)
* <span style="color: #FF0000">POPGRO</span> - Population growth, annual % (2020)
* <span style="color: #FF0000">LIFEXP</span> - Life expectancy at birth, total in years (2020)
* <span style="color: #FF0000">TUBINC</span> - Incidence of tuberculosis, per 100K population (2020)
* <span style="color: #FF0000">DTHCMD</span> - Cause of death by communicable diseases and maternal, prenatal and nutrition conditions,  % of total (2019)
* <span style="color: #FF0000">AGRLND</span> - Agricultural land,  % of land area (2020)
* <span style="color: #FF0000">GHGEMI</span> - Total greenhouse gas emissions, kt of CO2 equivalent (2020)
* <span style="color: #FF0000">RELOUT</span> - Renewable electricity output, % of total electricity output (2015)
* <span style="color: #FF0000">METEMI</span> - Methane emissions, kt of CO2 equivalent (2020)
* <span style="color: #FF0000">FORARE</span> - Forest area, % of land area (2020)
* <span style="color: #FF0000">CO2EMI</span> - CO2 emissions, metric tons per capita (2020)
* <span style="color: #FF0000">PM2EXP</span> - PM2.5 air pollution, population exposed to levels exceeding WHO guideline value,  % of total (2017)
* <span style="color: #FF0000">POPDEN</span> - Population density, people per sq. km of land area (2020)
* <span style="color: #FF0000">GDPCAP</span> - GDP per capita, current US Dollars (2020)
* <span style="color: #FF0000">ENRTER</span> - Tertiary school enrollment, % gross (2020)
* <span style="color: #FF0000">HDICAT</span> - Human development index, ordered category (2020)
* <span style="color: #FF0000">EPISCO</span> - Environment performance index , score (2022)


## 1.2. Data Description <a class="anchor" id="1.2"></a>

1. The dataset is comprised of:
    * **177 rows** (observations)
    * **22 columns** (variables)
        * **1/22 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/22 target** (numeric)
             * <span style="color: #FF0000">CANRAT</span>
        * **19/22 predictor** (numeric)
             * <span style="color: #FF0000">GDPPER</span>
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">PATRES</span>
             * <span style="color: #FF0000">RNDGDP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">RELOUT</span>
             * <span style="color: #FF0000">METEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">PM2EXP</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">ENRTER</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/22 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Loading Python Libraries
##################################
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import os
%matplotlib inline

from operator import add,mul,truediv
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PowerTransformer, StandardScaler
from scipy import stats

from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.pipeline import Pipeline
```


```python
##################################
# Setting Global Options
##################################
np.set_printoptions(suppress=True, precision=4)
pd.options.display.float_format = '{:.4f}'.format
```


```python
##################################
# Defining file paths
##################################
DATASETS_ORIGINAL_PATH = r"datasets\original"
```


```python
##################################
# Loading the dataset
# from the DATASETS_ORIGINAL_PATH
##################################
cancer_rate = pd.read_csv(os.path.join("..", DATASETS_ORIGINAL_PATH, "NumericCancerRates.csv"))
```


```python
##################################
# Performing a general exploration of the dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate.shape)
```

    Dataset Dimensions: 
    


    (177, 22)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(cancer_rate.dtypes)
```

    Column Names and Data Types:
    


    COUNTRY     object
    CANRAT     float64
    GDPPER     float64
    URBPOP     float64
    PATRES     float64
    RNDGDP     float64
    POPGRO     float64
    LIFEXP     float64
    TUBINC     float64
    DTHCMD     float64
    AGRLND     float64
    GHGEMI     float64
    RELOUT     float64
    METEMI     float64
    FORARE     float64
    CO2EMI     float64
    PM2EXP     float64
    POPDEN     float64
    ENRTER     float64
    GDPCAP     float64
    HDICAT      object
    EPISCO     float64
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
cancer_rate.head()
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
      <th>COUNTRY</th>
      <th>CANRAT</th>
      <th>GDPPER</th>
      <th>URBPOP</th>
      <th>PATRES</th>
      <th>RNDGDP</th>
      <th>POPGRO</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>...</th>
      <th>RELOUT</th>
      <th>METEMI</th>
      <th>FORARE</th>
      <th>CO2EMI</th>
      <th>PM2EXP</th>
      <th>POPDEN</th>
      <th>ENRTER</th>
      <th>GDPCAP</th>
      <th>HDICAT</th>
      <th>EPISCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>452.4000</td>
      <td>98380.6360</td>
      <td>86.2410</td>
      <td>2368.0000</td>
      <td>NaN</td>
      <td>1.2357</td>
      <td>83.2000</td>
      <td>7.2000</td>
      <td>4.9411</td>
      <td>...</td>
      <td>13.6378</td>
      <td>131484.7632</td>
      <td>17.4213</td>
      <td>14.7727</td>
      <td>24.8936</td>
      <td>3.3353</td>
      <td>110.1392</td>
      <td>51722.0690</td>
      <td>VH</td>
      <td>60.1000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New Zealand</td>
      <td>422.9000</td>
      <td>77541.7644</td>
      <td>86.6990</td>
      <td>348.0000</td>
      <td>NaN</td>
      <td>2.2048</td>
      <td>82.2561</td>
      <td>7.2000</td>
      <td>4.3547</td>
      <td>...</td>
      <td>80.0814</td>
      <td>32241.9370</td>
      <td>37.5701</td>
      <td>6.1608</td>
      <td>NaN</td>
      <td>19.3316</td>
      <td>75.7348</td>
      <td>41760.5948</td>
      <td>VH</td>
      <td>56.7000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>372.8000</td>
      <td>198405.8750</td>
      <td>63.6530</td>
      <td>75.0000</td>
      <td>1.2324</td>
      <td>1.0291</td>
      <td>82.5561</td>
      <td>5.3000</td>
      <td>5.6846</td>
      <td>...</td>
      <td>27.9654</td>
      <td>15252.8246</td>
      <td>11.3517</td>
      <td>6.7682</td>
      <td>0.2741</td>
      <td>72.3673</td>
      <td>74.6803</td>
      <td>85420.1909</td>
      <td>VH</td>
      <td>57.4000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>362.2000</td>
      <td>130941.6369</td>
      <td>82.6640</td>
      <td>269586.0000</td>
      <td>3.4229</td>
      <td>0.9643</td>
      <td>76.9805</td>
      <td>2.3000</td>
      <td>5.3021</td>
      <td>...</td>
      <td>13.2286</td>
      <td>748241.4029</td>
      <td>33.8669</td>
      <td>13.0328</td>
      <td>3.3432</td>
      <td>36.2410</td>
      <td>87.5677</td>
      <td>63528.6343</td>
      <td>VH</td>
      <td>51.1000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Denmark</td>
      <td>351.1000</td>
      <td>113300.6011</td>
      <td>88.1160</td>
      <td>1261.0000</td>
      <td>2.9687</td>
      <td>0.2916</td>
      <td>81.6024</td>
      <td>4.1000</td>
      <td>6.8261</td>
      <td>...</td>
      <td>65.5059</td>
      <td>7778.7739</td>
      <td>15.7110</td>
      <td>4.6912</td>
      <td>56.9145</td>
      <td>145.7851</td>
      <td>82.6643</td>
      <td>60915.4244</td>
      <td>VH</td>
      <td>77.9000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
##################################
# Setting the levels of the categorical variables
##################################
cancer_rate['HDICAT'] = cancer_rate['HDICAT'].astype('category')
cancer_rate['HDICAT'] = cancer_rate['HDICAT'].cat.set_categories(['L', 'M', 'H', 'VH'], ordered=True)
```


```python
##################################
# Performing a general exploration of the numeric variables
##################################
print('Numeric Variable Summary:')
display(cancer_rate.describe(include='number').transpose())
```

    Numeric Variable Summary:
    


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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CANRAT</th>
      <td>177.0000</td>
      <td>183.8294</td>
      <td>79.7434</td>
      <td>78.4000</td>
      <td>118.1000</td>
      <td>155.3000</td>
      <td>240.4000</td>
      <td>452.4000</td>
    </tr>
    <tr>
      <th>GDPPER</th>
      <td>165.0000</td>
      <td>45284.4243</td>
      <td>39417.9404</td>
      <td>1718.8049</td>
      <td>13545.2545</td>
      <td>34024.9009</td>
      <td>66778.4160</td>
      <td>234646.9049</td>
    </tr>
    <tr>
      <th>URBPOP</th>
      <td>174.0000</td>
      <td>59.7881</td>
      <td>22.8064</td>
      <td>13.3450</td>
      <td>42.4327</td>
      <td>61.7015</td>
      <td>79.1865</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>PATRES</th>
      <td>108.0000</td>
      <td>20607.3889</td>
      <td>134068.3101</td>
      <td>1.0000</td>
      <td>35.2500</td>
      <td>244.5000</td>
      <td>1297.7500</td>
      <td>1344817.0000</td>
    </tr>
    <tr>
      <th>RNDGDP</th>
      <td>74.0000</td>
      <td>1.1975</td>
      <td>1.1900</td>
      <td>0.0398</td>
      <td>0.2564</td>
      <td>0.8737</td>
      <td>1.6088</td>
      <td>5.3545</td>
    </tr>
    <tr>
      <th>POPGRO</th>
      <td>174.0000</td>
      <td>1.1270</td>
      <td>1.1977</td>
      <td>-2.0793</td>
      <td>0.2369</td>
      <td>1.1800</td>
      <td>2.0312</td>
      <td>3.7271</td>
    </tr>
    <tr>
      <th>LIFEXP</th>
      <td>174.0000</td>
      <td>71.7461</td>
      <td>7.6062</td>
      <td>52.7770</td>
      <td>65.9075</td>
      <td>72.4646</td>
      <td>77.5235</td>
      <td>84.5600</td>
    </tr>
    <tr>
      <th>TUBINC</th>
      <td>174.0000</td>
      <td>105.0059</td>
      <td>136.7229</td>
      <td>0.7700</td>
      <td>12.0000</td>
      <td>44.5000</td>
      <td>147.7500</td>
      <td>592.0000</td>
    </tr>
    <tr>
      <th>DTHCMD</th>
      <td>170.0000</td>
      <td>21.2605</td>
      <td>19.2733</td>
      <td>1.2836</td>
      <td>6.0780</td>
      <td>12.4563</td>
      <td>36.9805</td>
      <td>65.2079</td>
    </tr>
    <tr>
      <th>AGRLND</th>
      <td>174.0000</td>
      <td>38.7935</td>
      <td>21.7155</td>
      <td>0.5128</td>
      <td>20.1303</td>
      <td>40.3866</td>
      <td>54.0138</td>
      <td>80.8411</td>
    </tr>
    <tr>
      <th>GHGEMI</th>
      <td>170.0000</td>
      <td>259582.7099</td>
      <td>1118549.7301</td>
      <td>179.7252</td>
      <td>12527.4874</td>
      <td>41009.2760</td>
      <td>116482.5786</td>
      <td>12942868.3400</td>
    </tr>
    <tr>
      <th>RELOUT</th>
      <td>153.0000</td>
      <td>39.7600</td>
      <td>31.9149</td>
      <td>0.0003</td>
      <td>10.5827</td>
      <td>32.3817</td>
      <td>63.0115</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>METEMI</th>
      <td>170.0000</td>
      <td>47876.1336</td>
      <td>134661.0653</td>
      <td>11.5961</td>
      <td>3662.8849</td>
      <td>11118.9760</td>
      <td>32368.9090</td>
      <td>1186285.1810</td>
    </tr>
    <tr>
      <th>FORARE</th>
      <td>173.0000</td>
      <td>32.2182</td>
      <td>23.1200</td>
      <td>0.0081</td>
      <td>11.6044</td>
      <td>31.5090</td>
      <td>49.0718</td>
      <td>97.4121</td>
    </tr>
    <tr>
      <th>CO2EMI</th>
      <td>170.0000</td>
      <td>3.7511</td>
      <td>4.6065</td>
      <td>0.0326</td>
      <td>0.6319</td>
      <td>2.2984</td>
      <td>4.8235</td>
      <td>31.7268</td>
    </tr>
    <tr>
      <th>PM2EXP</th>
      <td>167.0000</td>
      <td>91.9406</td>
      <td>22.0600</td>
      <td>0.2741</td>
      <td>99.6271</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>100.0000</td>
    </tr>
    <tr>
      <th>POPDEN</th>
      <td>174.0000</td>
      <td>200.8868</td>
      <td>645.3834</td>
      <td>2.1151</td>
      <td>27.4545</td>
      <td>77.9831</td>
      <td>153.9936</td>
      <td>7918.9513</td>
    </tr>
    <tr>
      <th>ENRTER</th>
      <td>116.0000</td>
      <td>49.9950</td>
      <td>29.7062</td>
      <td>2.4326</td>
      <td>22.1072</td>
      <td>53.3925</td>
      <td>71.0575</td>
      <td>143.3107</td>
    </tr>
    <tr>
      <th>GDPCAP</th>
      <td>170.0000</td>
      <td>13992.0956</td>
      <td>19579.5430</td>
      <td>216.8274</td>
      <td>1870.5030</td>
      <td>5348.1929</td>
      <td>17421.1162</td>
      <td>117370.4969</td>
    </tr>
    <tr>
      <th>EPISCO</th>
      <td>165.0000</td>
      <td>42.9467</td>
      <td>12.4909</td>
      <td>18.9000</td>
      <td>33.0000</td>
      <td>40.9000</td>
      <td>50.5000</td>
      <td>77.9000</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the object variable
##################################
print('Object Variable Summary:')
display(cancer_rate.describe(include='object').transpose())
```

    Object Variable Summary:
    


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
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>COUNTRY</th>
      <td>177</td>
      <td>177</td>
      <td>Australia</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Performing a general exploration of the categorical variable
##################################
print('Categorical Variable Summary:')
display(cancer_rate.describe(include='category').transpose())
```

    Categorical Variable Summary:
    


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
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>HDICAT</th>
      <td>167</td>
      <td>4</td>
      <td>VH</td>
      <td>59</td>
    </tr>
  </tbody>
</table>
</div>


## 1.3. Data Quality Assessment <a class="anchor" id="1.3"></a>

Data quality findings based on assessment are as follows:
1. No duplicated rows observed.
2. Missing data noted for 20 variables with Null.Count>0 and Fill.Rate<1.0.
    * <span style="color: #FF0000">RNDGDP</span>: Null.Count = 103, Fill.Rate = 0.418
    * <span style="color: #FF0000">PATRES</span>: Null.Count = 69, Fill.Rate = 0.610
    * <span style="color: #FF0000">ENRTER</span>: Null.Count = 61, Fill.Rate = 0.655
    * <span style="color: #FF0000">RELOUT</span>: Null.Count = 24, Fill.Rate = 0.864
    * <span style="color: #FF0000">GDPPER</span>: Null.Count = 12, Fill.Rate = 0.932
    * <span style="color: #FF0000">EPISCO</span>: Null.Count = 12, Fill.Rate = 0.932
    * <span style="color: #FF0000">HDICAT</span>: Null.Count = 10, Fill.Rate = 0.943
    * <span style="color: #FF0000">PM2EXP</span>: Null.Count = 10, Fill.Rate = 0.943
    * <span style="color: #FF0000">DTHCMD</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">METEMI</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">CO2EMI</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">GDPCAP</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">GHGEMI</span>: Null.Count = 7, Fill.Rate = 0.960
    * <span style="color: #FF0000">FORARE</span>: Null.Count = 4, Fill.Rate = 0.977
    * <span style="color: #FF0000">TUBINC</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">AGRLND</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">POPGRO</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">POPDEN</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">URBPOP</span>: Null.Count = 3, Fill.Rate = 0.983
    * <span style="color: #FF0000">LIFEXP</span>: Null.Count = 3, Fill.Rate = 0.983
3. 120 observations noted with at least 1 missing data. From this number, 14 observations reported high Missing.Rate>0.2.
    * <span style="color: #FF0000">COUNTRY=Guadeloupe</span>: Missing.Rate= 0.909
    * <span style="color: #FF0000">COUNTRY=Martinique</span>: Missing.Rate= 0.909
    * <span style="color: #FF0000">COUNTRY=French Guiana</span>: Missing.Rate= 0.909
    * <span style="color: #FF0000">COUNTRY=New Caledonia</span>: Missing.Rate= 0.500
    * <span style="color: #FF0000">COUNTRY=French Polynesia</span>: Missing.Rate= 0.500
    * <span style="color: #FF0000">COUNTRY=Guam</span>: Missing.Rate= 0.500
    * <span style="color: #FF0000">COUNTRY=Puerto Rico</span>: Missing.Rate= 0.409
    * <span style="color: #FF0000">COUNTRY=North Korea</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Somalia</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=South Sudan</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Venezuela</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Libya</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Eritrea</span>: Missing.Rate= 0.227
    * <span style="color: #FF0000">COUNTRY=Yemen</span>: Missing.Rate= 0.227
4. Low variance observed for 1 variable with First.Second.Mode.Ratio>5.
    * <span style="color: #FF0000">PM2EXP</span>: First.Second.Mode.Ratio = 53.000
5. No low variance observed for any variable with Unique.Count.Ratio>10.
6. High skewness observed for 5 variables with Skewness>3 or Skewness<(-3).
    * <span style="color: #FF0000">POPDEN</span>: Skewness = +10.267
    * <span style="color: #FF0000">GHGEMI</span>: Skewness = +9.496
    * <span style="color: #FF0000">PATRES</span>: Skewness = +9.284
    * <span style="color: #FF0000">METEMI</span>: Skewness = +5.801
    * <span style="color: #FF0000">PM2EXP</span>: Skewness = -3.141


```python
##################################
# Counting the number of duplicated rows
##################################
cancer_rate.duplicated().sum()
```




    np.int64(0)




```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(cancer_rate.dtypes)
```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(cancer_rate.columns)
```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(cancer_rate)] * len(cancer_rate.columns))
```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(cancer_rate.isna().sum(axis=0))
```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(cancer_rate.count())
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
```


```python
##################################
# Formulating the summary
# for all columns
##################################
all_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                              data_type_list,
                                              row_count_list,
                                              non_null_count_list,
                                              null_count_list,
                                              fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(all_column_quality_summary)
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
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COUNTRY</td>
      <td>object</td>
      <td>177</td>
      <td>177</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CANRAT</td>
      <td>float64</td>
      <td>177</td>
      <td>177</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.9322</td>
    </tr>
    <tr>
      <th>3</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.9831</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PATRES</td>
      <td>float64</td>
      <td>177</td>
      <td>108</td>
      <td>69</td>
      <td>0.6102</td>
    </tr>
    <tr>
      <th>5</th>
      <td>RNDGDP</td>
      <td>float64</td>
      <td>177</td>
      <td>74</td>
      <td>103</td>
      <td>0.4181</td>
    </tr>
    <tr>
      <th>6</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.9831</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.9831</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.9831</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.9605</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.9831</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.9605</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RELOUT</td>
      <td>float64</td>
      <td>177</td>
      <td>153</td>
      <td>24</td>
      <td>0.8644</td>
    </tr>
    <tr>
      <th>13</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.9605</td>
    </tr>
    <tr>
      <th>14</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>177</td>
      <td>173</td>
      <td>4</td>
      <td>0.9774</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.9605</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.9435</td>
    </tr>
    <tr>
      <th>17</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.9831</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ENRTER</td>
      <td>float64</td>
      <td>177</td>
      <td>116</td>
      <td>61</td>
      <td>0.6554</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.9605</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.9435</td>
    </tr>
    <tr>
      <th>21</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.9322</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of columns
# with Fill.Rate < 1.00
##################################
len(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)])
```




    20




```python
##################################
# Identifying the columns
# with Fill.Rate < 1.00
##################################
display(all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<1)].sort_values(by=['Fill.Rate'], ascending=True))
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
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5</th>
      <td>RNDGDP</td>
      <td>float64</td>
      <td>177</td>
      <td>74</td>
      <td>103</td>
      <td>0.4181</td>
    </tr>
    <tr>
      <th>4</th>
      <td>PATRES</td>
      <td>float64</td>
      <td>177</td>
      <td>108</td>
      <td>69</td>
      <td>0.6102</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ENRTER</td>
      <td>float64</td>
      <td>177</td>
      <td>116</td>
      <td>61</td>
      <td>0.6554</td>
    </tr>
    <tr>
      <th>12</th>
      <td>RELOUT</td>
      <td>float64</td>
      <td>177</td>
      <td>153</td>
      <td>24</td>
      <td>0.8644</td>
    </tr>
    <tr>
      <th>21</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.9322</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>177</td>
      <td>165</td>
      <td>12</td>
      <td>0.9322</td>
    </tr>
    <tr>
      <th>16</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.9435</td>
    </tr>
    <tr>
      <th>20</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>177</td>
      <td>167</td>
      <td>10</td>
      <td>0.9435</td>
    </tr>
    <tr>
      <th>15</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.9605</td>
    </tr>
    <tr>
      <th>13</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.9605</td>
    </tr>
    <tr>
      <th>11</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.9605</td>
    </tr>
    <tr>
      <th>9</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.9605</td>
    </tr>
    <tr>
      <th>19</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>177</td>
      <td>170</td>
      <td>7</td>
      <td>0.9605</td>
    </tr>
    <tr>
      <th>14</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>177</td>
      <td>173</td>
      <td>4</td>
      <td>0.9774</td>
    </tr>
    <tr>
      <th>6</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.9831</td>
    </tr>
    <tr>
      <th>3</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.9831</td>
    </tr>
    <tr>
      <th>17</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.9831</td>
    </tr>
    <tr>
      <th>10</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.9831</td>
    </tr>
    <tr>
      <th>7</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.9831</td>
    </tr>
    <tr>
      <th>8</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>177</td>
      <td>174</td>
      <td>3</td>
      <td>0.9831</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Identifying the rows
# with Fill.Rate < 0.90
##################################
column_low_fill_rate = all_column_quality_summary[(all_column_quality_summary['Fill.Rate']<0.90)]
```


```python
##################################
# Gathering the metadata labels for each observation
##################################
row_metadata_list = cancer_rate["COUNTRY"].values.tolist()
```


```python
##################################
# Gathering the number of columns for each observation
##################################
column_count_list = list([len(cancer_rate.columns)] * len(cancer_rate))
```


```python
##################################
# Gathering the number of missing data for each row
##################################
null_row_list = list(cancer_rate.isna().sum(axis=1))
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
missing_rate_list = map(truediv, null_row_list, column_count_list)
```


```python
##################################
# Identifying the rows
# with missing data
##################################
all_row_quality_summary = pd.DataFrame(zip(row_metadata_list,
                                           column_count_list,
                                           null_row_list,
                                           missing_rate_list), 
                                        columns=['Row.Name',
                                                 'Column.Count',
                                                 'Null.Count',                                                 
                                                 'Missing.Rate'])
display(all_row_quality_summary)
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
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Australia</td>
      <td>22</td>
      <td>1</td>
      <td>0.0455</td>
    </tr>
    <tr>
      <th>1</th>
      <td>New Zealand</td>
      <td>22</td>
      <td>2</td>
      <td>0.0909</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Ireland</td>
      <td>22</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>United States</td>
      <td>22</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Denmark</td>
      <td>22</td>
      <td>0</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>172</th>
      <td>Congo Republic</td>
      <td>22</td>
      <td>3</td>
      <td>0.1364</td>
    </tr>
    <tr>
      <th>173</th>
      <td>Bhutan</td>
      <td>22</td>
      <td>2</td>
      <td>0.0909</td>
    </tr>
    <tr>
      <th>174</th>
      <td>Nepal</td>
      <td>22</td>
      <td>2</td>
      <td>0.0909</td>
    </tr>
    <tr>
      <th>175</th>
      <td>Gambia</td>
      <td>22</td>
      <td>4</td>
      <td>0.1818</td>
    </tr>
    <tr>
      <th>176</th>
      <td>Niger</td>
      <td>22</td>
      <td>2</td>
      <td>0.0909</td>
    </tr>
  </tbody>
</table>
<p>177 rows × 4 columns</p>
</div>



```python
##################################
# Counting the number of rows
# with Missing.Rate > 0.00
##################################
len(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.00)])
```




    120




```python
##################################
# Counting the number of rows
# with Missing.Rate > 0.20
##################################
len(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.20)])
```




    14




```python
##################################
# Identifying the rows
# with Missing.Rate > 0.20
##################################
row_high_missing_rate = all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.20)]
```


```python
##################################
# Identifying the rows
# with Missing.Rate > 0.20
##################################
display(all_row_quality_summary[(all_row_quality_summary['Missing.Rate']>0.20)].sort_values(by=['Missing.Rate'], ascending=False))
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
      <th>Row.Name</th>
      <th>Column.Count</th>
      <th>Null.Count</th>
      <th>Missing.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>35</th>
      <td>Guadeloupe</td>
      <td>22</td>
      <td>20</td>
      <td>0.9091</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Martinique</td>
      <td>22</td>
      <td>20</td>
      <td>0.9091</td>
    </tr>
    <tr>
      <th>56</th>
      <td>French Guiana</td>
      <td>22</td>
      <td>20</td>
      <td>0.9091</td>
    </tr>
    <tr>
      <th>13</th>
      <td>New Caledonia</td>
      <td>22</td>
      <td>11</td>
      <td>0.5000</td>
    </tr>
    <tr>
      <th>44</th>
      <td>French Polynesia</td>
      <td>22</td>
      <td>11</td>
      <td>0.5000</td>
    </tr>
    <tr>
      <th>75</th>
      <td>Guam</td>
      <td>22</td>
      <td>11</td>
      <td>0.5000</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Puerto Rico</td>
      <td>22</td>
      <td>9</td>
      <td>0.4091</td>
    </tr>
    <tr>
      <th>85</th>
      <td>North Korea</td>
      <td>22</td>
      <td>6</td>
      <td>0.2727</td>
    </tr>
    <tr>
      <th>168</th>
      <td>South Sudan</td>
      <td>22</td>
      <td>6</td>
      <td>0.2727</td>
    </tr>
    <tr>
      <th>132</th>
      <td>Somalia</td>
      <td>22</td>
      <td>6</td>
      <td>0.2727</td>
    </tr>
    <tr>
      <th>117</th>
      <td>Libya</td>
      <td>22</td>
      <td>5</td>
      <td>0.2273</td>
    </tr>
    <tr>
      <th>73</th>
      <td>Venezuela</td>
      <td>22</td>
      <td>5</td>
      <td>0.2273</td>
    </tr>
    <tr>
      <th>161</th>
      <td>Eritrea</td>
      <td>22</td>
      <td>5</td>
      <td>0.2273</td>
    </tr>
    <tr>
      <th>164</th>
      <td>Yemen</td>
      <td>22</td>
      <td>5</td>
      <td>0.2273</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the dataset
# with numeric columns only
##################################
cancer_rate_numeric = cancer_rate.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = cancer_rate_numeric.columns
```


```python
##################################
# Gathering the minimum value for each numeric column
##################################
numeric_minimum_list = cancer_rate_numeric.min()
```


```python
##################################
# Gathering the mean value for each numeric column
##################################
numeric_mean_list = cancer_rate_numeric.mean()
```


```python
##################################
# Gathering the median value for each numeric column
##################################
numeric_median_list = cancer_rate_numeric.median()
```


```python
##################################
# Gathering the maximum value for each numeric column
##################################
numeric_maximum_list = cancer_rate_numeric.max()
```


```python
##################################
# Gathering the first mode values for each numeric column
##################################
numeric_first_mode_list = [cancer_rate[x].value_counts(dropna=True).index.tolist()[0] for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the second mode values for each numeric column
##################################
numeric_second_mode_list = [cancer_rate[x].value_counts(dropna=True).index.tolist()[1] for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the count of first mode values for each numeric column
##################################
numeric_first_mode_count_list = [cancer_rate_numeric[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the count of second mode values for each numeric column
##################################
numeric_second_mode_count_list = [cancer_rate_numeric[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cancer_rate_numeric]
```


```python
##################################
# Gathering the first mode to second mode ratio for each numeric column
##################################
numeric_first_second_mode_ratio_list = map(truediv, numeric_first_mode_count_list, numeric_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each numeric column
##################################
numeric_unique_count_list = cancer_rate_numeric.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each numeric column
##################################
numeric_row_count_list = list([len(cancer_rate_numeric)] * len(cancer_rate_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each numeric column
##################################
numeric_unique_count_ratio_list = map(truediv, numeric_unique_count_list, numeric_row_count_list)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = cancer_rate_numeric.skew()
```


```python
##################################
# Gathering the kurtosis value for each numeric column
##################################
numeric_kurtosis_list = cancer_rate_numeric.kurtosis()
```


```python
numeric_column_quality_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                numeric_minimum_list,
                                                numeric_mean_list,
                                                numeric_median_list,
                                                numeric_maximum_list,
                                                numeric_first_mode_list,
                                                numeric_second_mode_list,
                                                numeric_first_mode_count_list,
                                                numeric_second_mode_count_list,
                                                numeric_first_second_mode_ratio_list,
                                                numeric_unique_count_list,
                                                numeric_row_count_list,
                                                numeric_unique_count_ratio_list,
                                                numeric_skewness_list,
                                                numeric_kurtosis_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Minimum',
                                                 'Mean',
                                                 'Median',
                                                 'Maximum',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio',
                                                 'Skewness',
                                                 'Kurtosis'])
display(numeric_column_quality_summary)
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
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CANRAT</td>
      <td>78.4000</td>
      <td>183.8294</td>
      <td>155.3000</td>
      <td>452.4000</td>
      <td>135.3000</td>
      <td>130.6000</td>
      <td>3</td>
      <td>2</td>
      <td>1.5000</td>
      <td>167</td>
      <td>177</td>
      <td>0.9435</td>
      <td>0.8818</td>
      <td>0.0635</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GDPPER</td>
      <td>1718.8049</td>
      <td>45284.4243</td>
      <td>34024.9009</td>
      <td>234646.9049</td>
      <td>98380.6360</td>
      <td>77541.7644</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>165</td>
      <td>177</td>
      <td>0.9322</td>
      <td>1.5176</td>
      <td>3.4720</td>
    </tr>
    <tr>
      <th>2</th>
      <td>URBPOP</td>
      <td>13.3450</td>
      <td>59.7881</td>
      <td>61.7015</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>86.6990</td>
      <td>2</td>
      <td>1</td>
      <td>2.0000</td>
      <td>173</td>
      <td>177</td>
      <td>0.9774</td>
      <td>-0.2107</td>
      <td>-0.9628</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PATRES</td>
      <td>1.0000</td>
      <td>20607.3889</td>
      <td>244.5000</td>
      <td>1344817.0000</td>
      <td>6.0000</td>
      <td>2.0000</td>
      <td>4</td>
      <td>3</td>
      <td>1.3333</td>
      <td>97</td>
      <td>177</td>
      <td>0.5480</td>
      <td>9.2844</td>
      <td>91.1872</td>
    </tr>
    <tr>
      <th>4</th>
      <td>RNDGDP</td>
      <td>0.0398</td>
      <td>1.1975</td>
      <td>0.8737</td>
      <td>5.3545</td>
      <td>1.2324</td>
      <td>3.4229</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>74</td>
      <td>177</td>
      <td>0.4181</td>
      <td>1.3967</td>
      <td>1.6960</td>
    </tr>
    <tr>
      <th>5</th>
      <td>POPGRO</td>
      <td>-2.0793</td>
      <td>1.1270</td>
      <td>1.1800</td>
      <td>3.7271</td>
      <td>1.2357</td>
      <td>2.2048</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>174</td>
      <td>177</td>
      <td>0.9831</td>
      <td>-0.1952</td>
      <td>-0.4236</td>
    </tr>
    <tr>
      <th>6</th>
      <td>LIFEXP</td>
      <td>52.7770</td>
      <td>71.7461</td>
      <td>72.4646</td>
      <td>84.5600</td>
      <td>83.2000</td>
      <td>82.2561</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>174</td>
      <td>177</td>
      <td>0.9831</td>
      <td>-0.3580</td>
      <td>-0.6496</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TUBINC</td>
      <td>0.7700</td>
      <td>105.0059</td>
      <td>44.5000</td>
      <td>592.0000</td>
      <td>12.0000</td>
      <td>4.1000</td>
      <td>4</td>
      <td>3</td>
      <td>1.3333</td>
      <td>131</td>
      <td>177</td>
      <td>0.7401</td>
      <td>1.7463</td>
      <td>2.4294</td>
    </tr>
    <tr>
      <th>8</th>
      <td>DTHCMD</td>
      <td>1.2836</td>
      <td>21.2605</td>
      <td>12.4563</td>
      <td>65.2079</td>
      <td>4.9411</td>
      <td>4.3547</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>170</td>
      <td>177</td>
      <td>0.9605</td>
      <td>0.9005</td>
      <td>-0.6915</td>
    </tr>
    <tr>
      <th>9</th>
      <td>AGRLND</td>
      <td>0.5128</td>
      <td>38.7935</td>
      <td>40.3866</td>
      <td>80.8411</td>
      <td>46.2525</td>
      <td>38.5629</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>174</td>
      <td>177</td>
      <td>0.9831</td>
      <td>0.0740</td>
      <td>-0.9262</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GHGEMI</td>
      <td>179.7252</td>
      <td>259582.7099</td>
      <td>41009.2760</td>
      <td>12942868.3400</td>
      <td>571903.1199</td>
      <td>80158.0258</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>170</td>
      <td>177</td>
      <td>0.9605</td>
      <td>9.4961</td>
      <td>101.6373</td>
    </tr>
    <tr>
      <th>11</th>
      <td>RELOUT</td>
      <td>0.0003</td>
      <td>39.7600</td>
      <td>32.3817</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>80.0814</td>
      <td>3</td>
      <td>1</td>
      <td>3.0000</td>
      <td>151</td>
      <td>177</td>
      <td>0.8531</td>
      <td>0.5011</td>
      <td>-0.9818</td>
    </tr>
    <tr>
      <th>12</th>
      <td>METEMI</td>
      <td>11.5961</td>
      <td>47876.1336</td>
      <td>11118.9760</td>
      <td>1186285.1810</td>
      <td>131484.7632</td>
      <td>32241.9370</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>170</td>
      <td>177</td>
      <td>0.9605</td>
      <td>5.8010</td>
      <td>38.6614</td>
    </tr>
    <tr>
      <th>13</th>
      <td>FORARE</td>
      <td>0.0081</td>
      <td>32.2182</td>
      <td>31.5090</td>
      <td>97.4121</td>
      <td>17.4213</td>
      <td>37.5701</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>173</td>
      <td>177</td>
      <td>0.9774</td>
      <td>0.5193</td>
      <td>-0.3226</td>
    </tr>
    <tr>
      <th>14</th>
      <td>CO2EMI</td>
      <td>0.0326</td>
      <td>3.7511</td>
      <td>2.2984</td>
      <td>31.7268</td>
      <td>14.7727</td>
      <td>6.1608</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>170</td>
      <td>177</td>
      <td>0.9605</td>
      <td>2.7216</td>
      <td>10.3116</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PM2EXP</td>
      <td>0.2741</td>
      <td>91.9406</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>106</td>
      <td>2</td>
      <td>53.0000</td>
      <td>61</td>
      <td>177</td>
      <td>0.3446</td>
      <td>-3.1416</td>
      <td>9.0324</td>
    </tr>
    <tr>
      <th>16</th>
      <td>POPDEN</td>
      <td>2.1151</td>
      <td>200.8868</td>
      <td>77.9831</td>
      <td>7918.9513</td>
      <td>3.3353</td>
      <td>19.3316</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>174</td>
      <td>177</td>
      <td>0.9831</td>
      <td>10.2678</td>
      <td>119.9953</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ENRTER</td>
      <td>2.4326</td>
      <td>49.9950</td>
      <td>53.3925</td>
      <td>143.3107</td>
      <td>110.1392</td>
      <td>75.7348</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>116</td>
      <td>177</td>
      <td>0.6554</td>
      <td>0.2759</td>
      <td>-0.3929</td>
    </tr>
    <tr>
      <th>18</th>
      <td>GDPCAP</td>
      <td>216.8274</td>
      <td>13992.0956</td>
      <td>5348.1929</td>
      <td>117370.4969</td>
      <td>51722.0690</td>
      <td>41760.5948</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>170</td>
      <td>177</td>
      <td>0.9605</td>
      <td>2.2586</td>
      <td>5.9387</td>
    </tr>
    <tr>
      <th>19</th>
      <td>EPISCO</td>
      <td>18.9000</td>
      <td>42.9467</td>
      <td>40.9000</td>
      <td>77.9000</td>
      <td>29.6000</td>
      <td>43.6000</td>
      <td>3</td>
      <td>3</td>
      <td>1.0000</td>
      <td>137</td>
      <td>177</td>
      <td>0.7740</td>
      <td>0.6418</td>
      <td>0.0352</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of numeric columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    1




```python
##################################
# Identifying the numeric columns
# with First.Second.Mode.Ratio > 5.00
##################################
display(numeric_column_quality_summary[(numeric_column_quality_summary['First.Second.Mode.Ratio']>5)].sort_values(by=['First.Second.Mode.Ratio'], ascending=False))
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
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>15</th>
      <td>PM2EXP</td>
      <td>0.2741</td>
      <td>91.9406</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>106</td>
      <td>2</td>
      <td>53.0000</td>
      <td>61</td>
      <td>177</td>
      <td>0.3446</td>
      <td>-3.1416</td>
      <td>9.0324</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of numeric columns
# with Unique.Count.Ratio > 10.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0




```python
##################################
# Counting the number of numeric columns
# with Skewness > 3.00 or Skewness < -3.00
##################################
len(numeric_column_quality_summary[(numeric_column_quality_summary['Skewness']>3) | (numeric_column_quality_summary['Skewness']<(-3))])
```




    5




```python
##################################
# Identifying the numeric columns
# with Skewness > 3.00 or Skewness < -3.00
##################################
display(numeric_column_quality_summary[(numeric_column_quality_summary['Skewness']>3) | (numeric_column_quality_summary['Skewness']<(-3))].sort_values(by=['Skewness'], ascending=False))
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
      <th>Numeric.Column.Name</th>
      <th>Minimum</th>
      <th>Mean</th>
      <th>Median</th>
      <th>Maximum</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
      <th>Skewness</th>
      <th>Kurtosis</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>16</th>
      <td>POPDEN</td>
      <td>2.1151</td>
      <td>200.8868</td>
      <td>77.9831</td>
      <td>7918.9513</td>
      <td>3.3353</td>
      <td>19.3316</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>174</td>
      <td>177</td>
      <td>0.9831</td>
      <td>10.2678</td>
      <td>119.9953</td>
    </tr>
    <tr>
      <th>10</th>
      <td>GHGEMI</td>
      <td>179.7252</td>
      <td>259582.7099</td>
      <td>41009.2760</td>
      <td>12942868.3400</td>
      <td>571903.1199</td>
      <td>80158.0258</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>170</td>
      <td>177</td>
      <td>0.9605</td>
      <td>9.4961</td>
      <td>101.6373</td>
    </tr>
    <tr>
      <th>3</th>
      <td>PATRES</td>
      <td>1.0000</td>
      <td>20607.3889</td>
      <td>244.5000</td>
      <td>1344817.0000</td>
      <td>6.0000</td>
      <td>2.0000</td>
      <td>4</td>
      <td>3</td>
      <td>1.3333</td>
      <td>97</td>
      <td>177</td>
      <td>0.5480</td>
      <td>9.2844</td>
      <td>91.1872</td>
    </tr>
    <tr>
      <th>12</th>
      <td>METEMI</td>
      <td>11.5961</td>
      <td>47876.1336</td>
      <td>11118.9760</td>
      <td>1186285.1810</td>
      <td>131484.7632</td>
      <td>32241.9370</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>170</td>
      <td>177</td>
      <td>0.9605</td>
      <td>5.8010</td>
      <td>38.6614</td>
    </tr>
    <tr>
      <th>15</th>
      <td>PM2EXP</td>
      <td>0.2741</td>
      <td>91.9406</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>100.0000</td>
      <td>106</td>
      <td>2</td>
      <td>53.0000</td>
      <td>61</td>
      <td>177</td>
      <td>0.3446</td>
      <td>-3.1416</td>
      <td>9.0324</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the dataset
# with object column only
##################################
cancer_rate_object = cancer_rate.select_dtypes(include='object')
```


```python
##################################
# Gathering the variable names for the object column
##################################
object_variable_name_list = cancer_rate_object.columns
```


```python
##################################
# Gathering the first mode values for the object column
##################################
object_first_mode_list = [cancer_rate[x].value_counts().index.tolist()[0] for x in cancer_rate_object]
```


```python
##################################
# Gathering the second mode values for each object column
##################################
object_second_mode_list = [cancer_rate[x].value_counts().index.tolist()[1] for x in cancer_rate_object]
```


```python
##################################
# Gathering the count of first mode values for each object column
##################################
object_first_mode_count_list = [cancer_rate_object[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cancer_rate_object]
```


```python
##################################
# Gathering the count of second mode values for each object column
##################################
object_second_mode_count_list = [cancer_rate_object[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cancer_rate_object]
```


```python
##################################
# Gathering the first mode to second mode ratio for each object column
##################################
object_first_second_mode_ratio_list = map(truediv, object_first_mode_count_list, object_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each object column
##################################
object_unique_count_list = cancer_rate_object.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each object column
##################################
object_row_count_list = list([len(cancer_rate_object)] * len(cancer_rate_object.columns))
```


```python
##################################
# Gathering the unique to count ratio for each object column
##################################
object_unique_count_ratio_list = map(truediv, object_unique_count_list, object_row_count_list)
```


```python
object_column_quality_summary = pd.DataFrame(zip(object_variable_name_list,
                                                 object_first_mode_list,
                                                 object_second_mode_list,
                                                 object_first_mode_count_list,
                                                 object_second_mode_count_list,
                                                 object_first_second_mode_ratio_list,
                                                 object_unique_count_list,
                                                 object_row_count_list,
                                                 object_unique_count_ratio_list), 
                                        columns=['Object.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(object_column_quality_summary)
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
      <th>Object.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COUNTRY</td>
      <td>Australia</td>
      <td>New Zealand</td>
      <td>1</td>
      <td>1</td>
      <td>1.0000</td>
      <td>177</td>
      <td>177</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of object columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    0




```python
##################################
# Counting the number of object columns
# with Unique.Count.Ratio > 10.00
##################################
len(object_column_quality_summary[(object_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0




```python
##################################
# Formulating the dataset
# with categorical columns only
##################################
cancer_rate_categorical = cancer_rate.select_dtypes(include='category')
```


```python
##################################
# Gathering the variable names for the categorical column
##################################
categorical_variable_name_list = cancer_rate_categorical.columns
```


```python
##################################
# Gathering the first mode values for each categorical column
##################################
categorical_first_mode_list = [cancer_rate[x].value_counts().index.tolist()[0] for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the second mode values for each categorical column
##################################
categorical_second_mode_list = [cancer_rate[x].value_counts().index.tolist()[1] for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the count of first mode values for each categorical column
##################################
categorical_first_mode_count_list = [cancer_rate_categorical[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[0]]).sum() for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the count of second mode values for each categorical column
##################################
categorical_second_mode_count_list = [cancer_rate_categorical[x].isin([cancer_rate[x].value_counts(dropna=True).index.tolist()[1]]).sum() for x in cancer_rate_categorical]
```


```python
##################################
# Gathering the first mode to second mode ratio for each categorical column
##################################
categorical_first_second_mode_ratio_list = map(truediv, categorical_first_mode_count_list, categorical_second_mode_count_list)
```


```python
##################################
# Gathering the count of unique values for each categorical column
##################################
categorical_unique_count_list = cancer_rate_categorical.nunique(dropna=True)
```


```python
##################################
# Gathering the number of observations for each categorical column
##################################
categorical_row_count_list = list([len(cancer_rate_categorical)] * len(cancer_rate_categorical.columns))
```


```python
##################################
# Gathering the unique to count ratio for each categorical column
##################################
categorical_unique_count_ratio_list = map(truediv, categorical_unique_count_list, categorical_row_count_list)
```


```python
categorical_column_quality_summary = pd.DataFrame(zip(categorical_variable_name_list,
                                                    categorical_first_mode_list,
                                                    categorical_second_mode_list,
                                                    categorical_first_mode_count_list,
                                                    categorical_second_mode_count_list,
                                                    categorical_first_second_mode_ratio_list,
                                                    categorical_unique_count_list,
                                                    categorical_row_count_list,
                                                    categorical_unique_count_ratio_list), 
                                        columns=['Categorical.Column.Name',
                                                 'First.Mode',
                                                 'Second.Mode',
                                                 'First.Mode.Count',
                                                 'Second.Mode.Count',
                                                 'First.Second.Mode.Ratio',
                                                 'Unique.Count',
                                                 'Row.Count',
                                                 'Unique.Count.Ratio'])
display(categorical_column_quality_summary)
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
      <th>Categorical.Column.Name</th>
      <th>First.Mode</th>
      <th>Second.Mode</th>
      <th>First.Mode.Count</th>
      <th>Second.Mode.Count</th>
      <th>First.Second.Mode.Ratio</th>
      <th>Unique.Count</th>
      <th>Row.Count</th>
      <th>Unique.Count.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>HDICAT</td>
      <td>VH</td>
      <td>H</td>
      <td>59</td>
      <td>39</td>
      <td>1.5128</td>
      <td>4</td>
      <td>177</td>
      <td>0.0226</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Counting the number of categorical columns
# with First.Second.Mode.Ratio > 5.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['First.Second.Mode.Ratio']>5)])
```




    0




```python
##################################
# Counting the number of categorical columns
# with Unique.Count.Ratio > 10.00
##################################
len(categorical_column_quality_summary[(categorical_column_quality_summary['Unique.Count.Ratio']>10)])
```




    0



## 1.4. Data Preprocessing <a class="anchor" id="1.4"></a>


### 1.4.1 Data Cleaning <a class="anchor" id="1.4.1"></a>

1. Subsets of rows and columns with high rates of missing data were removed from the dataset:
    * 4 variables with Fill.Rate<0.9 were excluded for subsequent analysis.
        * <span style="color: #FF0000">RNDGDP</span>: Null.Count = 103, Fill.Rate = 0.418
        * <span style="color: #FF0000">PATRES</span>: Null.Count = 69, Fill.Rate = 0.610
        * <span style="color: #FF0000">ENRTER</span>: Null.Count = 61, Fill.Rate = 0.655
        * <span style="color: #FF0000">RELOUT</span>: Null.Count = 24, Fill.Rate = 0.864
    * 14 rows with Missing.Rate>0.2 were exluded for subsequent analysis.
        * <span style="color: #FF0000">COUNTRY=Guadeloupe</span>: Missing.Rate= 0.909
        * <span style="color: #FF0000">COUNTRY=Martinique</span>: Missing.Rate= 0.909
        * <span style="color: #FF0000">COUNTRY=French Guiana</span>: Missing.Rate= 0.909
        * <span style="color: #FF0000">COUNTRY=New Caledonia</span>: Missing.Rate= 0.500
        * <span style="color: #FF0000">COUNTRY=French Polynesia</span>: Missing.Rate= 0.500
        * <span style="color: #FF0000">COUNTRY=Guam</span>: Missing.Rate= 0.500
        * <span style="color: #FF0000">COUNTRY=Puerto Rico</span>: Missing.Rate= 0.409
        * <span style="color: #FF0000">COUNTRY=North Korea</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Somalia</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=South Sudan</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Venezuela</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Libya</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Eritrea</span>: Missing.Rate= 0.227
        * <span style="color: #FF0000">COUNTRY=Yemen</span>: Missing.Rate= 0.227  
2. No variables were removed due to zero or near-zero variance.
3. The cleaned dataset is comprised of:
    * **163 rows** (observations)
    * **18 columns** (variables)
        * **1/18 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/18 target** (numeric)
             * <span style="color: #FF0000">CANRAT</span>
        * **15/18 predictor** (numeric)
             * <span style="color: #FF0000">GDPPER</span>
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">METEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">PM2EXP</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/18 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Performing a general exploration of the original dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate.shape)
```

    Dataset Dimensions: 
    


    (177, 22)



```python
##################################
# Filtering out the rows with
# with Missing.Rate > 0.20
##################################
cancer_rate_filtered_row = cancer_rate.drop(cancer_rate[cancer_rate.COUNTRY.isin(row_high_missing_rate['Row.Name'].values.tolist())].index)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_filtered_row.shape)
```

    Dataset Dimensions: 
    


    (163, 22)



```python
##################################
# Filtering out the columns with
# with Fill.Rate < 0.90
##################################
cancer_rate_filtered_row_column = cancer_rate_filtered_row.drop(column_low_fill_rate['Column.Name'].values.tolist(), axis=1)
```


```python
##################################
# Formulating a new dataset object
# for the cleaned data
##################################
cancer_rate_cleaned = cancer_rate_filtered_row_column
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_cleaned.shape)
```

    Dataset Dimensions: 
    


    (163, 18)


### 1.4.2 Missing Data Imputation <a class="anchor" id="1.4.2"></a>

1. Missing data for numeric variables were imputed using the iterative imputer algorithm with a  linear regression estimator.
    * <span style="color: #FF0000">GDPPER</span>: Null.Count = 1
    * <span style="color: #FF0000">FORARE</span>: Null.Count = 1
    * <span style="color: #FF0000">PM2EXP</span>: Null.Count = 5
2. Missing data for categorical variables were imputed using the most frequent value.
    * <span style="color: #FF0000">HDICAP</span>: Null.Count = 1


```python
##################################
# Formulating the summary
# for all cleaned columns
##################################
cleaned_column_quality_summary = pd.DataFrame(zip(list(cancer_rate_cleaned.columns),
                                                  list(cancer_rate_cleaned.dtypes),
                                                  list([len(cancer_rate_cleaned)] * len(cancer_rate_cleaned.columns)),
                                                  list(cancer_rate_cleaned.count()),
                                                  list(cancer_rate_cleaned.isna().sum(axis=0))), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count'])
display(cleaned_column_quality_summary)
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
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>COUNTRY</td>
      <td>object</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>CANRAT</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>163</td>
      <td>162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>163</td>
      <td>162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>163</td>
      <td>158</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
    <tr>
      <th>16</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>163</td>
      <td>162</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the cleaned dataset
# with categorical columns only
##################################
cancer_rate_cleaned_categorical = cancer_rate_cleaned.select_dtypes(include='object')
```


```python
##################################
# Formulating the cleaned dataset
# with numeric columns only
##################################
cancer_rate_cleaned_numeric = cancer_rate_cleaned.select_dtypes(include='number')
```


```python
##################################
# Taking a snapshot of the cleaned dataset
##################################
cancer_rate_cleaned_numeric.head()
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
      <th>CANRAT</th>
      <th>GDPPER</th>
      <th>URBPOP</th>
      <th>POPGRO</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>AGRLND</th>
      <th>GHGEMI</th>
      <th>METEMI</th>
      <th>FORARE</th>
      <th>CO2EMI</th>
      <th>PM2EXP</th>
      <th>POPDEN</th>
      <th>GDPCAP</th>
      <th>EPISCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>452.4000</td>
      <td>98380.6360</td>
      <td>86.2410</td>
      <td>1.2357</td>
      <td>83.2000</td>
      <td>7.2000</td>
      <td>4.9411</td>
      <td>46.2525</td>
      <td>571903.1199</td>
      <td>131484.7632</td>
      <td>17.4213</td>
      <td>14.7727</td>
      <td>24.8936</td>
      <td>3.3353</td>
      <td>51722.0690</td>
      <td>60.1000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>422.9000</td>
      <td>77541.7644</td>
      <td>86.6990</td>
      <td>2.2048</td>
      <td>82.2561</td>
      <td>7.2000</td>
      <td>4.3547</td>
      <td>38.5629</td>
      <td>80158.0258</td>
      <td>32241.9370</td>
      <td>37.5701</td>
      <td>6.1608</td>
      <td>NaN</td>
      <td>19.3316</td>
      <td>41760.5948</td>
      <td>56.7000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>372.8000</td>
      <td>198405.8750</td>
      <td>63.6530</td>
      <td>1.0291</td>
      <td>82.5561</td>
      <td>5.3000</td>
      <td>5.6846</td>
      <td>65.4957</td>
      <td>59497.7347</td>
      <td>15252.8246</td>
      <td>11.3517</td>
      <td>6.7682</td>
      <td>0.2741</td>
      <td>72.3673</td>
      <td>85420.1909</td>
      <td>57.4000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>362.2000</td>
      <td>130941.6369</td>
      <td>82.6640</td>
      <td>0.9643</td>
      <td>76.9805</td>
      <td>2.3000</td>
      <td>5.3021</td>
      <td>44.3634</td>
      <td>5505180.8900</td>
      <td>748241.4029</td>
      <td>33.8669</td>
      <td>13.0328</td>
      <td>3.3432</td>
      <td>36.2410</td>
      <td>63528.6343</td>
      <td>51.1000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>351.1000</td>
      <td>113300.6011</td>
      <td>88.1160</td>
      <td>0.2916</td>
      <td>81.6024</td>
      <td>4.1000</td>
      <td>6.8261</td>
      <td>65.4997</td>
      <td>41135.5545</td>
      <td>7778.7739</td>
      <td>15.7110</td>
      <td>4.6912</td>
      <td>56.9145</td>
      <td>145.7851</td>
      <td>60915.4244</td>
      <td>77.9000</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Defining the estimator to be used
# at each step of the round-robin imputation
##################################
lr = LinearRegression()
```


```python
##################################
# Defining the parameter of the
# iterative imputer which will estimate 
# the columns with missing values
# as a function of the other columns
# in a round-robin fashion
##################################
iterative_imputer = IterativeImputer(
    estimator = lr,
    max_iter = 10,
    tol = 1e-10,
    imputation_order = 'ascending',
    random_state=88888888
)
```


```python
##################################
# Implementing the iterative imputer 
##################################
cancer_rate_imputed_numeric_array = iterative_imputer.fit_transform(cancer_rate_cleaned_numeric)
```


```python
##################################
# Transforming the imputed data
# from an array to a dataframe
##################################
cancer_rate_imputed_numeric = pd.DataFrame(cancer_rate_imputed_numeric_array, 
                                           columns = cancer_rate_cleaned_numeric.columns)
```


```python
##################################
# Taking a snapshot of the imputed dataset
##################################
cancer_rate_imputed_numeric.head()
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
      <th>CANRAT</th>
      <th>GDPPER</th>
      <th>URBPOP</th>
      <th>POPGRO</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>AGRLND</th>
      <th>GHGEMI</th>
      <th>METEMI</th>
      <th>FORARE</th>
      <th>CO2EMI</th>
      <th>PM2EXP</th>
      <th>POPDEN</th>
      <th>GDPCAP</th>
      <th>EPISCO</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>452.4000</td>
      <td>98380.6360</td>
      <td>86.2410</td>
      <td>1.2357</td>
      <td>83.2000</td>
      <td>7.2000</td>
      <td>4.9411</td>
      <td>46.2525</td>
      <td>571903.1199</td>
      <td>131484.7632</td>
      <td>17.4213</td>
      <td>14.7727</td>
      <td>24.8936</td>
      <td>3.3353</td>
      <td>51722.0690</td>
      <td>60.1000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>422.9000</td>
      <td>77541.7644</td>
      <td>86.6990</td>
      <td>2.2048</td>
      <td>82.2561</td>
      <td>7.2000</td>
      <td>4.3547</td>
      <td>38.5629</td>
      <td>80158.0258</td>
      <td>32241.9370</td>
      <td>37.5701</td>
      <td>6.1608</td>
      <td>59.4755</td>
      <td>19.3316</td>
      <td>41760.5948</td>
      <td>56.7000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>372.8000</td>
      <td>198405.8750</td>
      <td>63.6530</td>
      <td>1.0291</td>
      <td>82.5561</td>
      <td>5.3000</td>
      <td>5.6846</td>
      <td>65.4957</td>
      <td>59497.7347</td>
      <td>15252.8246</td>
      <td>11.3517</td>
      <td>6.7682</td>
      <td>0.2741</td>
      <td>72.3673</td>
      <td>85420.1909</td>
      <td>57.4000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>362.2000</td>
      <td>130941.6369</td>
      <td>82.6640</td>
      <td>0.9643</td>
      <td>76.9805</td>
      <td>2.3000</td>
      <td>5.3021</td>
      <td>44.3634</td>
      <td>5505180.8900</td>
      <td>748241.4029</td>
      <td>33.8669</td>
      <td>13.0328</td>
      <td>3.3432</td>
      <td>36.2410</td>
      <td>63528.6343</td>
      <td>51.1000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>351.1000</td>
      <td>113300.6011</td>
      <td>88.1160</td>
      <td>0.2916</td>
      <td>81.6024</td>
      <td>4.1000</td>
      <td>6.8261</td>
      <td>65.4997</td>
      <td>41135.5545</td>
      <td>7778.7739</td>
      <td>15.7110</td>
      <td>4.6912</td>
      <td>56.9145</td>
      <td>145.7851</td>
      <td>60915.4244</td>
      <td>77.9000</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Formulating the cleaned dataset
# with categorical columns only
##################################
cancer_rate_cleaned_categorical = cancer_rate_cleaned.select_dtypes(include='category')
```


```python
##################################
# Imputing the missing data
# for categorical columns with
# the most frequent category
##################################
cancer_rate_cleaned_categorical['HDICAT'] = cancer_rate_cleaned_categorical['HDICAT'].fillna(cancer_rate_cleaned_categorical['HDICAT'].mode()[0])
cancer_rate_imputed_categorical = cancer_rate_cleaned_categorical.reset_index(drop=True)
```


```python
##################################
# Formulating the imputed dataset
##################################
cancer_rate_imputed = pd.concat([cancer_rate_imputed_numeric,cancer_rate_imputed_categorical], axis=1, join='inner')  
```


```python
##################################
# Gathering the data types for each column
##################################
data_type_list = list(cancer_rate_imputed.dtypes)
```


```python
##################################
# Gathering the variable names for each column
##################################
variable_name_list = list(cancer_rate_imputed.columns)
```


```python
##################################
# Gathering the number of observations for each column
##################################
row_count_list = list([len(cancer_rate_imputed)] * len(cancer_rate_imputed.columns))
```


```python
##################################
# Gathering the number of missing data for each column
##################################
null_count_list = list(cancer_rate_imputed.isna().sum(axis=0))
```


```python
##################################
# Gathering the number of non-missing data for each column
##################################
non_null_count_list = list(cancer_rate_imputed.count())
```


```python
##################################
# Gathering the missing data percentage for each column
##################################
fill_rate_list = map(truediv, non_null_count_list, row_count_list)
```


```python
##################################
# Formulating the summary
# for all imputed columns
##################################
imputed_column_quality_summary = pd.DataFrame(zip(variable_name_list,
                                                  data_type_list,
                                                  row_count_list,
                                                  non_null_count_list,
                                                  null_count_list,
                                                  fill_rate_list), 
                                        columns=['Column.Name',
                                                 'Column.Type',
                                                 'Row.Count',
                                                 'Non.Null.Count',
                                                 'Null.Count',                                                 
                                                 'Fill.Rate'])
display(imputed_column_quality_summary)
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
      <th>Column.Name</th>
      <th>Column.Type</th>
      <th>Row.Count</th>
      <th>Non.Null.Count</th>
      <th>Null.Count</th>
      <th>Fill.Rate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CANRAT</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GDPPER</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>URBPOP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>POPGRO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LIFEXP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TUBINC</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DTHCMD</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AGRLND</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GHGEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>9</th>
      <td>METEMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FORARE</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CO2EMI</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PM2EXP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>13</th>
      <td>POPDEN</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GDPCAP</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>15</th>
      <td>EPISCO</td>
      <td>float64</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
    <tr>
      <th>16</th>
      <td>HDICAT</td>
      <td>category</td>
      <td>163</td>
      <td>163</td>
      <td>0</td>
      <td>1.0000</td>
    </tr>
  </tbody>
</table>
</div>


### 1.4.3 Outlier Detection <a class="anchor" id="1.4.3"></a>

1. High number of outliers observed for 5 numeric variables with Outlier.Ratio>0.10 and marginal to high Skewness.
    * <span style="color: #FF0000">PM2EXP</span>: Outlier.Count = 37, Outlier.Ratio = 0.226, Skewness=-3.061
    * <span style="color: #FF0000">GHGEMI</span>: Outlier.Count = 27, Outlier.Ratio = 0.165, Skewness=+9.299
    * <span style="color: #FF0000">GDPCAP</span>: Outlier.Count = 22, Outlier.Ratio = 0.134, Skewness=+2.311
    * <span style="color: #FF0000">POPDEN</span>: Outlier.Count = 20, Outlier.Ratio = 0.122, Skewness=+9.972
    * <span style="color: #FF0000">METEMI</span>: Outlier.Count = 20, Outlier.Ratio = 0.122, Skewness=+5.688
2. Minimal number of outliers observed for 5 numeric variables with Outlier.Ratio<0.10 and normal Skewness.
    * <span style="color: #FF0000">TUBINC</span>: Outlier.Count = 12, Outlier.Ratio = 0.073, Skewness=+1.747
    * <span style="color: #FF0000">CO2EMI</span>: Outlier.Count = 11, Outlier.Ratio = 0.067, Skewness=+2.693
    * <span style="color: #FF0000">GDPPER</span>: Outlier.Count = 3, Outlier.Ratio = 0.018, Skewness=+1.554
    * <span style="color: #FF0000">EPISCO</span>: Outlier.Count = 3, Outlier.Ratio = 0.018, Skewness=+0.635
    * <span style="color: #FF0000">CANRAT</span>: Outlier.Count = 2, Outlier.Ratio = 0.012, Skewness=+0.910


```python
##################################
# Formulating the imputed dataset
# with numeric columns only
##################################
cancer_rate_imputed_numeric = cancer_rate_imputed.select_dtypes(include='number')
```


```python
##################################
# Gathering the variable names for each numeric column
##################################
numeric_variable_name_list = list(cancer_rate_imputed_numeric.columns)
```


```python
##################################
# Gathering the skewness value for each numeric column
##################################
numeric_skewness_list = cancer_rate_imputed_numeric.skew()
```


```python
##################################
# Computing the interquartile range
# for all columns
##################################
cancer_rate_imputed_numeric_q1 = cancer_rate_imputed_numeric.quantile(0.25)
cancer_rate_imputed_numeric_q3 = cancer_rate_imputed_numeric.quantile(0.75)
cancer_rate_imputed_numeric_iqr = cancer_rate_imputed_numeric_q3 - cancer_rate_imputed_numeric_q1
```


```python
##################################
# Gathering the outlier count for each numeric column
# based on the interquartile range criterion
##################################
numeric_outlier_count_list = ((cancer_rate_imputed_numeric < (cancer_rate_imputed_numeric_q1 - 1.5 * cancer_rate_imputed_numeric_iqr)) | (cancer_rate_imputed_numeric > (cancer_rate_imputed_numeric_q3 + 1.5 * cancer_rate_imputed_numeric_iqr))).sum()
```


```python
##################################
# Gathering the number of observations for each column
##################################
numeric_row_count_list = list([len(cancer_rate_imputed_numeric)] * len(cancer_rate_imputed_numeric.columns))
```


```python
##################################
# Gathering the unique to count ratio for each categorical column
##################################
numeric_outlier_ratio_list = map(truediv, numeric_outlier_count_list, numeric_row_count_list)
```


```python
##################################
# Formulating the outlier summary
# for all numeric columns
##################################
numeric_column_outlier_summary = pd.DataFrame(zip(numeric_variable_name_list,
                                                  numeric_skewness_list,
                                                  numeric_outlier_count_list,
                                                  numeric_row_count_list,
                                                  numeric_outlier_ratio_list), 
                                        columns=['Numeric.Column.Name',
                                                 'Skewness',
                                                 'Outlier.Count',
                                                 'Row.Count',
                                                 'Outlier.Ratio'])
display(numeric_column_outlier_summary)
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
      <th>Numeric.Column.Name</th>
      <th>Skewness</th>
      <th>Outlier.Count</th>
      <th>Row.Count</th>
      <th>Outlier.Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>CANRAT</td>
      <td>0.9101</td>
      <td>2</td>
      <td>163</td>
      <td>0.0123</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GDPPER</td>
      <td>1.5544</td>
      <td>3</td>
      <td>163</td>
      <td>0.0184</td>
    </tr>
    <tr>
      <th>2</th>
      <td>URBPOP</td>
      <td>-0.2123</td>
      <td>0</td>
      <td>163</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>POPGRO</td>
      <td>-0.1817</td>
      <td>0</td>
      <td>163</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>LIFEXP</td>
      <td>-0.3297</td>
      <td>0</td>
      <td>163</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>TUBINC</td>
      <td>1.7480</td>
      <td>12</td>
      <td>163</td>
      <td>0.0736</td>
    </tr>
    <tr>
      <th>6</th>
      <td>DTHCMD</td>
      <td>0.9307</td>
      <td>0</td>
      <td>163</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>AGRLND</td>
      <td>0.0353</td>
      <td>0</td>
      <td>163</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GHGEMI</td>
      <td>9.3000</td>
      <td>27</td>
      <td>163</td>
      <td>0.1656</td>
    </tr>
    <tr>
      <th>9</th>
      <td>METEMI</td>
      <td>5.6887</td>
      <td>20</td>
      <td>163</td>
      <td>0.1227</td>
    </tr>
    <tr>
      <th>10</th>
      <td>FORARE</td>
      <td>0.5562</td>
      <td>0</td>
      <td>163</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>11</th>
      <td>CO2EMI</td>
      <td>2.6936</td>
      <td>11</td>
      <td>163</td>
      <td>0.0675</td>
    </tr>
    <tr>
      <th>12</th>
      <td>PM2EXP</td>
      <td>-3.0616</td>
      <td>37</td>
      <td>163</td>
      <td>0.2270</td>
    </tr>
    <tr>
      <th>13</th>
      <td>POPDEN</td>
      <td>9.9728</td>
      <td>20</td>
      <td>163</td>
      <td>0.1227</td>
    </tr>
    <tr>
      <th>14</th>
      <td>GDPCAP</td>
      <td>2.3111</td>
      <td>22</td>
      <td>163</td>
      <td>0.1350</td>
    </tr>
    <tr>
      <th>15</th>
      <td>EPISCO</td>
      <td>0.6360</td>
      <td>3</td>
      <td>163</td>
      <td>0.0184</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Formulating the individual boxplots
# for all numeric columns
##################################
for column in cancer_rate_imputed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cancer_rate_imputed_numeric, x=column)
```


    
![png](output_124_0.png)
    



    
![png](output_124_1.png)
    



    
![png](output_124_2.png)
    



    
![png](output_124_3.png)
    



    
![png](output_124_4.png)
    



    
![png](output_124_5.png)
    



    
![png](output_124_6.png)
    



    
![png](output_124_7.png)
    



    
![png](output_124_8.png)
    



    
![png](output_124_9.png)
    



    
![png](output_124_10.png)
    



    
![png](output_124_11.png)
    



    
![png](output_124_12.png)
    



    
![png](output_124_13.png)
    



    
![png](output_124_14.png)
    



    
![png](output_124_15.png)
    


### 1.4.4 Collinearity <a class="anchor" id="1.4.4"></a>

1. Majority of the numeric variables reported moderate to high correlation which were statistically significant.
2. Among pairwise combinations of numeric variables, high Pearson.Correlation.Coefficient values were noted for:
    * <span style="color: #FF0000">GDPPER</span> and <span style="color: #FF0000">GDPCAP</span>: Pearson.Correlation.Coefficient = +0.921
    * <span style="color: #FF0000">GHGEMI</span> and <span style="color: #FF0000">METEMI</span>: Pearson.Correlation.Coefficient = +0.905
3. Among the highly correlated pairs, variables with the lowest correlation against the target variable were removed.
    * <span style="color: #FF0000">GDPPER</span>: Pearson.Correlation.Coefficient = +0.690
    * <span style="color: #FF0000">METEMI</span>: Pearson.Correlation.Coefficient = +0.062
4. The cleaned dataset is comprised of:
    * **163 rows** (observations)
    * **16 columns** (variables)
        * **1/16 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/16 target** (numeric)
             * <span style="color: #FF0000">CANRAT</span>
        * **13/16 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">PM2EXP</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/16 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
def plot_correlation_matrix(corr, mask=None):
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, 
                ax=ax,
                mask=mask,
                annot=True, 
                vmin=-1, 
                vmax=1, 
                center=0,
                cmap='coolwarm', 
                linewidths=1, 
                linecolor='gray', 
                cbar_kws={'orientation': 'horizontal'})  
```


```python
##################################
# Computing the correlation coefficients
# and correlation p-values
# among pairs of numeric columns
##################################
cancer_rate_imputed_numeric_correlation_pairs = {}
cancer_rate_imputed_numeric_columns = cancer_rate_imputed_numeric.columns.tolist()
for numeric_column_a, numeric_column_b in itertools.combinations(cancer_rate_imputed_numeric_columns, 2):
    cancer_rate_imputed_numeric_correlation_pairs[numeric_column_a + '_' + numeric_column_b] = stats.pearsonr(
        cancer_rate_imputed_numeric.loc[:, numeric_column_a], 
        cancer_rate_imputed_numeric.loc[:, numeric_column_b])
```


```python
##################################
# Formulating the pairwise correlation summary
# for all numeric columns
##################################
cancer_rate_imputed_numeric_summary = cancer_rate_imputed_numeric.from_dict(cancer_rate_imputed_numeric_correlation_pairs, orient='index')
cancer_rate_imputed_numeric_summary.columns = ['Pearson.Correlation.Coefficient', 'Correlation.PValue']
display(cancer_rate_imputed_numeric_summary.sort_values(by=['Pearson.Correlation.Coefficient'], ascending=False).head(20))
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
      <th>Pearson.Correlation.Coefficient</th>
      <th>Correlation.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>GDPPER_GDPCAP</th>
      <td>0.9210</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>GHGEMI_METEMI</th>
      <td>0.9051</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>POPGRO_DTHCMD</th>
      <td>0.7595</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>GDPPER_LIFEXP</th>
      <td>0.7558</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_EPISCO</th>
      <td>0.7126</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_GDPCAP</th>
      <td>0.6970</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>GDPCAP_EPISCO</th>
      <td>0.6967</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_LIFEXP</th>
      <td>0.6923</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_GDPPER</th>
      <td>0.6868</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>LIFEXP_GDPCAP</th>
      <td>0.6838</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>GDPPER_EPISCO</th>
      <td>0.6808</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>GDPPER_URBPOP</th>
      <td>0.6664</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>GDPPER_CO2EMI</th>
      <td>0.6550</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>TUBINC_DTHCMD</th>
      <td>0.6436</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>URBPOP_LIFEXP</th>
      <td>0.6240</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>LIFEXP_EPISCO</th>
      <td>0.6203</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>URBPOP_GDPCAP</th>
      <td>0.5592</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CO2EMI_GDPCAP</th>
      <td>0.5502</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>URBPOP_CO2EMI</th>
      <td>0.5500</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>LIFEXP_CO2EMI</th>
      <td>0.5313</td>
      <td>0.0000</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
##################################
cancer_rate_imputed_numeric_correlation = cancer_rate_imputed_numeric.corr()
mask = np.triu(cancer_rate_imputed_numeric_correlation)
plot_correlation_matrix(cancer_rate_imputed_numeric_correlation,mask)
plt.show()
```


    
![png](output_129_0.png)
    



```python
##################################
# Formulating a function 
# to plot the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
def correlation_significance(df=None):
    p_matrix = np.zeros(shape=(df.shape[1],df.shape[1]))
    for col in df.columns:
        for col2 in df.drop(col,axis=1).columns:
            _ , p = stats.pearsonr(df[col],df[col2])
            p_matrix[df.columns.to_list().index(col),df.columns.to_list().index(col2)] = p
    return p_matrix
```


```python
##################################
# Plotting the correlation matrix
# for all pairwise combinations
# of numeric columns
# with significant p-values only
##################################
cancer_rate_imputed_numeric_correlation_p_values = correlation_significance(cancer_rate_imputed_numeric)                     
mask = np.invert(np.tril(cancer_rate_imputed_numeric_correlation_p_values<0.05)) 
plot_correlation_matrix(cancer_rate_imputed_numeric_correlation,mask)  
```


    
![png](output_131_0.png)
    



```python
##################################
# Filtering out one among the 
# highly correlated variable pairs with
# lesser Pearson.Correlation.Coefficient
# when compared to the target variable
##################################
cancer_rate_imputed_numeric.drop(['GDPPER','METEMI'], inplace=True, axis=1)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_imputed_numeric.shape)
```

    Dataset Dimensions: 
    


    (163, 14)


### 1.4.5 Shape Transformation <a class="anchor" id="1.4.5"></a>

1. A Yeo-Johnson transformation was applied to all numeric variables to improve distributional shape.
2. Most variables achieved symmetrical distributions with minimal outliers after transformation.
3. One variable which remained skewed even after applying shape transformation was removed.
    * <span style="color: #FF0000">PM2EXP</span> 
4. The transformed dataset is comprised of:
    * **163 rows** (observations)
    * **15 columns** (variables)
        * **1/15 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/15 target** (numeric)
             * <span style="color: #FF0000">CANRAT</span>
        * **12/15 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/15 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Conducting a Yeo-Johnson Transformation
# to address the distributional
# shape of the variables
##################################
yeo_johnson_transformer = PowerTransformer(method='yeo-johnson',
                                          standardize=False)
cancer_rate_imputed_numeric_array = yeo_johnson_transformer.fit_transform(cancer_rate_imputed_numeric)
```


```python
##################################
# Formulating a new dataset object
# for the transformed data
##################################
cancer_rate_transformed_numeric = pd.DataFrame(cancer_rate_imputed_numeric_array,
                                               columns=cancer_rate_imputed_numeric.columns)
```


```python
##################################
# Formulating the individual boxplots
# for all transformed numeric columns
##################################
for column in cancer_rate_transformed_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cancer_rate_transformed_numeric, x=column)
```


    
![png](output_137_0.png)
    



    
![png](output_137_1.png)
    



    
![png](output_137_2.png)
    



    
![png](output_137_3.png)
    



    
![png](output_137_4.png)
    



    
![png](output_137_5.png)
    



    
![png](output_137_6.png)
    



    
![png](output_137_7.png)
    



    
![png](output_137_8.png)
    



    
![png](output_137_9.png)
    



    
![png](output_137_10.png)
    



    
![png](output_137_11.png)
    



    
![png](output_137_12.png)
    



    
![png](output_137_13.png)
    



```python
##################################
# Filtering out the column
# which remained skewed even
# after applying shape transformation
##################################
cancer_rate_transformed_numeric.drop(['PM2EXP'], inplace=True, axis=1)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_transformed_numeric.shape)
```

    Dataset Dimensions: 
    


    (163, 13)


### 1.4.6 Centering and Scaling <a class="anchor" id="1.4.6"></a>

1. All numeric variables were transformed using the standardization method to achieve a comparable scale between values.
4. The scaled dataset is comprised of:
    * **163 rows** (observations)
    * **15 columns** (variables)
        * **1/15 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/15 target** (numeric)
             * <span style="color: #FF0000">CANRAT</span>
        * **12/15 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **1/15 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT</span>


```python
##################################
# Conducting standardization
# to transform the values of the 
# variables into comparable scale
##################################
standardization_scaler = StandardScaler()
cancer_rate_transformed_numeric_array = standardization_scaler.fit_transform(cancer_rate_transformed_numeric)
```


```python
##################################
# Formulating a new dataset object
# for the scaled data
##################################
cancer_rate_scaled_numeric = pd.DataFrame(cancer_rate_transformed_numeric_array,
                                          columns=cancer_rate_transformed_numeric.columns)
```


```python
##################################
# Formulating the individual boxplots
# for all transformed numeric columns
##################################
for column in cancer_rate_scaled_numeric:
        plt.figure(figsize=(17,1))
        sns.boxplot(data=cancer_rate_scaled_numeric, x=column)
```


    
![png](output_143_0.png)
    



    
![png](output_143_1.png)
    



    
![png](output_143_2.png)
    



    
![png](output_143_3.png)
    



    
![png](output_143_4.png)
    



    
![png](output_143_5.png)
    



    
![png](output_143_6.png)
    



    
![png](output_143_7.png)
    



    
![png](output_143_8.png)
    



    
![png](output_143_9.png)
    



    
![png](output_143_10.png)
    



    
![png](output_143_11.png)
    



    
![png](output_143_12.png)
    


### 1.4.7 Data Encoding <a class="anchor" id="1.4.7"></a>

1. One-hot encoding was applied to the <span style="color: #FF0000">HDICAP_VH</span> variable resulting to 4 additional columns in the dataset:
    * <span style="color: #FF0000">HDICAP_L</span>
    * <span style="color: #FF0000">HDICAP_M</span>
    * <span style="color: #FF0000">HDICAP_H</span>
    * <span style="color: #FF0000">HDICAP_VH</span>


```python
##################################
# Formulating the categorical column
# for encoding transformation
##################################
cancer_rate_categorical_encoded = pd.DataFrame(cancer_rate_cleaned_categorical.loc[:, 'HDICAT'].to_list(),
                                               columns=['HDICAT'])
```


```python
##################################
# Applying a one-hot encoding transformation
# for the categorical column
##################################
cancer_rate_categorical_encoded = pd.get_dummies(cancer_rate_categorical_encoded, columns=['HDICAT'])
```

### 1.4.8 Preprocessed Data Description <a class="anchor" id="1.4.8"></a>

1. The preprocessed dataset is comprised of:
    * **163 rows** (observations)
    * **18 columns** (variables)
        * **1/18 metadata** (object)
            * <span style="color: #FF0000">COUNTRY</span>
        * **1/18 target** (numeric)
             * <span style="color: #FF0000">CANRAT</span>
        * **12/18 predictor** (numeric)
             * <span style="color: #FF0000">URBPOP</span>
             * <span style="color: #FF0000">POPGRO</span>
             * <span style="color: #FF0000">LIFEXP</span>
             * <span style="color: #FF0000">TUBINC</span>
             * <span style="color: #FF0000">DTHCMD</span>
             * <span style="color: #FF0000">AGRLND</span>
             * <span style="color: #FF0000">GHGEMI</span>
             * <span style="color: #FF0000">FORARE</span>
             * <span style="color: #FF0000">CO2EMI</span>
             * <span style="color: #FF0000">POPDEN</span>
             * <span style="color: #FF0000">GDPCAP</span>
             * <span style="color: #FF0000">EPISCO</span>
        * **4/18 predictor** (categorical)
             * <span style="color: #FF0000">HDICAT_L</span>
             * <span style="color: #FF0000">HDICAT_M</span>
             * <span style="color: #FF0000">HDICAT_H</span>
             * <span style="color: #FF0000">HDICAT_VH</span>


```python
##################################
# Consolidating both numeric columns
# and encoded categorical columns
##################################
cancer_rate_preprocessed = pd.concat([cancer_rate_scaled_numeric,cancer_rate_categorical_encoded], axis=1, join='inner')  
```


```python
##################################
# Performing a general exploration of the consolidated dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_preprocessed.shape)
```

    Dataset Dimensions: 
    


    (163, 17)


## 1.5. Data Exploration <a class="anchor" id="1.5"></a>

### 1.5.1 Exploratory Data Analysis <a class="anchor" id="1.5.1"></a>

1. Bivariate analysis identified individual predictors with generally linear relationship to the target variable based on visual inspection.
2. Increasing values for the following predictors correspond to higher <span style="color: #FF0000">CANRAT</span> measurements: 
    * <span style="color: #FF0000">URBPOP</span>
    * <span style="color: #FF0000">LIFEXP</span>    
    * <span style="color: #FF0000">CO2EMI</span>    
    * <span style="color: #FF0000">GDPCAP</span>    
    * <span style="color: #FF0000">EPISCO</span>    
    * <span style="color: #FF0000">HDICAP_VH</span>
3. Decreasing values for the following predictors correspond to higher <span style="color: #FF0000">CANRAT</span> measurements: 
    * <span style="color: #FF0000">POPGRO</span>
    * <span style="color: #FF0000">TUBINC</span>    
    * <span style="color: #FF0000">DTHCMD</span> 
    * <span style="color: #FF0000">HDICAP_L</span>
    * <span style="color: #FF0000">HDICAP_M</span>
4. Values for the following predictors did not affect <span style="color: #FF0000">CANRAT</span> measurements: 
    * <span style="color: #FF0000">AGRLND</span>
    * <span style="color: #FF0000">GHGEMI</span>    
    * <span style="color: #FF0000">FORARE</span> 
    * <span style="color: #FF0000">POPDEN</span> 
    * <span style="color: #FF0000">HDICAP_H</span>


```python
##################################
# Segregating the target
# and predictor variable lists
##################################
cancer_rate_preprocessed_target = ['CANRAT']
cancer_rate_preprocessed_predictors = cancer_rate_preprocessed.drop('CANRAT', axis=1).columns
```


```python
##################################
# Segregating the target
# and predictor variable names
##################################
y_variable = 'CANRAT'
x_variables = cancer_rate_preprocessed_predictors
```


```python
##################################
# Defining the number of 
# rows and columns for the subplots
##################################
num_rows = 8
num_cols = 2
```


```python
##################################
# Formulating the subplot structure
##################################
fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 40))

##################################
# Flattening the multi-row and
# multi-column axes
##################################
axes = axes.ravel()

##################################
# Formulating the individual scatterplots
# for all scaled numeric columns
##################################
for i, x_variable in enumerate(x_variables):
    ax = axes[i]
    ax.scatter(cancer_rate_preprocessed[x_variable],cancer_rate_preprocessed[y_variable])
    ax.set_title(f'{y_variable} Versus {x_variable}')
    ax.set_xlabel(x_variable)
    ax.set_ylabel(y_variable)

##################################
# Adjusting the subplot layout
##################################
plt.tight_layout()

##################################
# Presenting the subplots
##################################
plt.show()
```


    
![png](output_155_0.png)
    


### 1.5.2 Hypothesis Testing <a class="anchor" id="1.5.2"></a>

1. The relationship between the numeric predictors to the <span style="color: #FF0000">CANRAT</span> target variable was statistically evaluated using the following hypotheses:
    * **Null**: Pearson correlation coefficient is equal to zero 
    * **Alternative**: Pearson correlation coefficient is not equal to zero    
2. There is sufficient evidence to conclude of a statistically significant linear relationship between the <span style="color: #FF0000">CANRAT</span> target variable and 10 of the 12 numeric predictors given their high Pearson correlation coefficient values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">GDPCAP</span>: Pearson.Correlation.Coefficient=+0.735, Correlation.PValue=0.000
    * <span style="color: #FF0000">LIFEXP</span>: Pearson.Correlation.Coefficient=+0.702, Correlation.PValue=0.000   
    * <span style="color: #FF0000">DTHCMD</span>: Pearson.Correlation.Coefficient=-0.687, Correlation.PValue=0.000 
    * <span style="color: #FF0000">EPISCO</span>: Pearson.Correlation.Coefficient=+0.648, Correlation.PValue=0.000 
    * <span style="color: #FF0000">TUBINC</span>: Pearson.Correlation.Coefficient=+0.628, Correlation.PValue=0.000 
    * <span style="color: #FF0000">CO2EMI</span>: Pearson.Correlation.Coefficient=+0.585, Correlation.PValue=0.000  
    * <span style="color: #FF0000">POPGRO</span>: Pearson.Correlation.Coefficient=-0.498, Correlation.PValue=0.000
    * <span style="color: #FF0000">URBPOP</span>: Pearson.Correlation.Coefficient=+0.479, Correlation.PValue=0.000   
    * <span style="color: #FF0000">GHGEMI</span>: Pearson.Correlation.Coefficient=+0.232, Correlation.PValue=0.002
    * <span style="color: #FF0000">FORARE</span>: Pearson.Correlation.Coefficient=+0.165, Correlation.PValue=0.035
3. The relationship between the categorical predictors to the <span style="color: #FF0000">CANRAT</span> target variable was statistically evaluated using the following hypotheses:
    * **Null**: Difference in the means between groups 0 and 1 is equal to zero 
    * **Alternative**: Difference in the means between groups 0 and 1 is not equal to zero    
2. There is sufficient evidence to conclude of a statistically significant difference between the means of <span style="color: #FF0000">CANRAT</span> measuremens obtained from groups 0 and 1 in 3 of the 4 categorical predictors given their high t-test statistic values with reported low p-values less than the significance level of 0.05.
    * <span style="color: #FF0000">HDICAT_VH</span>: T.Test.Statistic=-10.605, T.Test.PValue=0.000
    * <span style="color: #FF0000">HDICAT_L</span>: T.Test.Statistic=+6.559, T.Test.PValue=0.000   
    * <span style="color: #FF0000">HDICAT_M</span>: T.Test.Statistic=+5.104, T.Test.PValue=0.000 


```python
##################################
# Computing the correlation coefficients
# and correlation p-values
# between the target variable
# and numeric predictor columns
##################################
cancer_rate_preprocessed_numeric_correlation_target = {}
cancer_rate_preprocessed_numeric = cancer_rate_preprocessed.drop(['HDICAT_L','HDICAT_M','HDICAT_H','HDICAT_VH'], axis=1)
cancer_rate_preprocessed_numeric_columns = cancer_rate_preprocessed_numeric.columns.tolist()
for numeric_column in cancer_rate_preprocessed_numeric_columns:
    cancer_rate_preprocessed_numeric_correlation_target['CANRAT_' + numeric_column] = stats.pearsonr(
        cancer_rate_preprocessed_numeric.loc[:, 'CANRAT'], 
        cancer_rate_preprocessed_numeric.loc[:, numeric_column])
```


```python
##################################
# Formulating the pairwise correlation summary
# between the target variable
# and numeric predictor columns
##################################
cancer_rate_preprocessed_numeric_summary = cancer_rate_preprocessed_numeric.from_dict(cancer_rate_preprocessed_numeric_correlation_target, orient='index')
cancer_rate_preprocessed_numeric_summary.columns = ['Pearson.Correlation.Coefficient', 'Correlation.PValue']
display(cancer_rate_preprocessed_numeric_summary.sort_values(by=['Correlation.PValue'], ascending=True).head(13))
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
      <th>Pearson.Correlation.Coefficient</th>
      <th>Correlation.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CANRAT_CANRAT</th>
      <td>1.0000</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_GDPCAP</th>
      <td>0.7351</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_LIFEXP</th>
      <td>0.7024</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_DTHCMD</th>
      <td>-0.6871</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_EPISCO</th>
      <td>0.6484</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_TUBINC</th>
      <td>-0.6289</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_CO2EMI</th>
      <td>0.5855</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_POPGRO</th>
      <td>-0.4985</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_URBPOP</th>
      <td>0.4794</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_GHGEMI</th>
      <td>0.2325</td>
      <td>0.0028</td>
    </tr>
    <tr>
      <th>CANRAT_FORARE</th>
      <td>0.1653</td>
      <td>0.0350</td>
    </tr>
    <tr>
      <th>CANRAT_AGRLND</th>
      <td>-0.0245</td>
      <td>0.7560</td>
    </tr>
    <tr>
      <th>CANRAT_POPDEN</th>
      <td>0.0019</td>
      <td>0.9808</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Computing the t-test 
# statistic and p-values
# between the target variable
# and categorical predictor columns
##################################
cancer_rate_preprocessed_categorical_ttest_target = {}
cancer_rate_preprocessed_categorical = cancer_rate_preprocessed[['CANRAT','HDICAT_L','HDICAT_M','HDICAT_H','HDICAT_VH']]
cancer_rate_preprocessed_categorical_columns = ['HDICAT_L','HDICAT_M','HDICAT_H','HDICAT_VH']
for categorical_column in cancer_rate_preprocessed_categorical_columns:
    group_0 = cancer_rate_preprocessed_categorical[cancer_rate_preprocessed_categorical.loc[:,categorical_column]==0]
    group_1 = cancer_rate_preprocessed_categorical[cancer_rate_preprocessed_categorical.loc[:,categorical_column]==1]
    cancer_rate_preprocessed_categorical_ttest_target['CANRAT_' + categorical_column] = stats.ttest_ind(
        group_0['CANRAT'], 
        group_1['CANRAT'], 
        equal_var=True)
```


```python
##################################
# Formulating the pairwise ttest summary
# between the target variable
# and categorical predictor columns
##################################
cancer_rate_preprocessed_categorical_summary = cancer_rate_preprocessed_categorical.from_dict(cancer_rate_preprocessed_categorical_ttest_target, orient='index')
cancer_rate_preprocessed_categorical_summary.columns = ['T.Test.Statistic', 'T.Test.PValue']
display(cancer_rate_preprocessed_categorical_summary.sort_values(by=['T.Test.PValue'], ascending=True).head(4))
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
      <th>T.Test.Statistic</th>
      <th>T.Test.PValue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>CANRAT_HDICAT_VH</th>
      <td>-10.6057</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_HDICAT_L</th>
      <td>6.5598</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_HDICAT_M</th>
      <td>5.1050</td>
      <td>0.0000</td>
    </tr>
    <tr>
      <th>CANRAT_HDICAT_H</th>
      <td>-0.6360</td>
      <td>0.5257</td>
    </tr>
  </tbody>
</table>
</div>


## 1.6. Model Development <a class="anchor" id="1.6"></a>

### 1.6.1 Premodelling Data Description <a class="anchor" id="1.6.1"></a>

1. Among the 10 numeric variables determined to have a statistically significant linear relationship between the <span style="color: #FF0000">CANRAT</span> target variable, only 6 were retained with absolute Pearson correlation coefficient values greater than 0.50. 
    * <span style="color: #FF0000">GDPCAP</span>: Pearson.Correlation.Coefficient=+0.735, Correlation.PValue=0.000
    * <span style="color: #FF0000">LIFEXP</span>: Pearson.Correlation.Coefficient=+0.702, Correlation.PValue=0.000   
    * <span style="color: #FF0000">DTHCMD</span>: Pearson.Correlation.Coefficient=-0.687, Correlation.PValue=0.000 
    * <span style="color: #FF0000">EPISCO</span>: Pearson.Correlation.Coefficient=+0.648, Correlation.PValue=0.000 
    * <span style="color: #FF0000">TUBINC</span>: Pearson.Correlation.Coefficient=+0.628, Correlation.PValue=0.000 
    * <span style="color: #FF0000">CO2EMI</span>: Pearson.Correlation.Coefficient=+0.585, Correlation.PValue=0.000  
2. Among the 4 categorical predictors determined to have a a statistically significant difference between the means of <span style="color: #FF0000">CANRAT</span> measurements obtained from groups 0 and 1, only 1 was retained with absolute T-Test statistics greater than 10.
    * <span style="color: #FF0000">HDICAT_VH</span>: T.Test.Statistic=-10.605, T.Test.PValue=0.000


```python
##################################
# Consolidating relevant numeric columns
# and encoded categorical columns
# after hypothesis testing
##################################
cancer_rate_premodelling = cancer_rate_preprocessed.drop(['AGRLND','POPDEN','GHGEMI','FORARE','POPGRO','URBPOP','HDICAT_H','HDICAT_M','HDICAT_L'], axis=1)
```


```python
##################################
# Performing a general exploration of the filtered dataset
##################################
print('Dataset Dimensions: ')
display(cancer_rate_premodelling.shape)
```

    Dataset Dimensions: 
    


    (163, 8)



```python
##################################
# Listing the column names and data types
##################################
print('Column Names and Data Types:')
display(cancer_rate_premodelling.dtypes)
```

    Column Names and Data Types:
    


    CANRAT       float64
    LIFEXP       float64
    TUBINC       float64
    DTHCMD       float64
    CO2EMI       float64
    GDPCAP       float64
    EPISCO       float64
    HDICAT_VH       bool
    dtype: object



```python
##################################
# Taking a snapshot of the dataset
##################################
cancer_rate_premodelling.head()
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
      <th>CANRAT</th>
      <th>LIFEXP</th>
      <th>TUBINC</th>
      <th>DTHCMD</th>
      <th>CO2EMI</th>
      <th>GDPCAP</th>
      <th>EPISCO</th>
      <th>HDICAT_VH</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.0765</td>
      <td>1.6432</td>
      <td>-1.1023</td>
      <td>-0.9715</td>
      <td>1.7368</td>
      <td>1.5498</td>
      <td>1.3067</td>
      <td>True</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.9630</td>
      <td>1.4880</td>
      <td>-1.1023</td>
      <td>-1.0914</td>
      <td>0.9435</td>
      <td>1.4078</td>
      <td>1.1029</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.7428</td>
      <td>1.5370</td>
      <td>-1.2753</td>
      <td>-0.8363</td>
      <td>1.0317</td>
      <td>1.8794</td>
      <td>1.1458</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.6909</td>
      <td>0.6642</td>
      <td>-1.6963</td>
      <td>-0.9037</td>
      <td>1.6277</td>
      <td>1.6854</td>
      <td>0.7398</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.6342</td>
      <td>1.3819</td>
      <td>-1.4134</td>
      <td>-0.6571</td>
      <td>0.6863</td>
      <td>1.6578</td>
      <td>2.2183</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>




```python
##################################
# Gathering the pairplot for all variables
##################################
sns.pairplot(cancer_rate_premodelling, kind='reg')
plt.show()
```


    
![png](output_167_0.png)
    



```python
##################################
# Separating the target 
# and predictor columns
##################################
X = cancer_rate_premodelling.drop('CANRAT', axis = 1)
y = cancer_rate_premodelling.CANRAT
```


```python
##################################
# Formulating the train and test data
# using a 70-30 ratio
##################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state= 88888888)
```


```python
##################################
# Performing a general exploration of the train dataset
##################################
print('Dataset Dimensions: ')
display(X_train.shape)
```

    Dataset Dimensions: 
    


    (114, 7)



```python
##################################
# Performing a general exploration of the train dataset
##################################
print('Dataset Dimensions: ')
display(X_test.shape)
```

    Dataset Dimensions: 
    


    (49, 7)



```python
##################################
# Defining a function to compute
# model performance
##################################
def model_performance_evaluation(y_true, y_pred):
    metric_name = ['R2','MSE','MAE']
    metric_value = [r2_score(y_true, y_pred),
                   mean_squared_error(y_true, y_pred),
                   mean_absolute_error(y_true, y_pred)]    
    metric_summary = pd.DataFrame(zip(metric_name, metric_value),
                                  columns=['metric_name','metric_value']) 
    return(metric_summary)
```


```python
##################################
# Defining a function to investigate
# model performance with respect to
# the regularization parameter
##################################
def rmse_alpha_plot(model_type):
    MSE=[]
    coefs = []
    for alpha in alphas:
        model = model_type(alpha=alpha)
        model.fit(X_train, y_train)
        coefs.append(abs(model.coef_))
        y_pred = model.predict(X_test)
        MSE.append(mean_squared_error(y_test, y_pred))

    ax = plt.gca()
    ax.plot(alphas, MSE)
    ax.set_xscale("log")
    plt.xlabel("Alpha")
    plt.ylabel("Mean Squared Error")
    plt.title("Mean Squared Error versus Alpha Regularization")
    plt.show()
    
    
def rmse_l1_ratio_plot(model_type):
    MSE=[]
    coefs = []
    for l1_ratio in l1_ratios:
        model = model_type(l1_ratio=l1_ratio)
        model.fit(X_train, y_train)
        coefs.append(abs(model.coef_))
        y_pred = model.predict(X_test)
        MSE.append(mean_squared_error(y_test, y_pred))

    ax = plt.gca()
    ax.plot(l1_ratios, MSE)
    plt.xlabel("L1_Ratio")
    plt.ylabel("Mean Squared Error")
    plt.title("Mean Squared Error versus L1_Ratio Regularization")
    plt.show()
```

### 1.6.2 Linear Regression <a class="anchor" id="1.6.2"></a>

[Linear Regression](https://link.springer.com/book/10.1007/978-1-4757-3462-1) explores the linear relationship between a scalar response and one or more covariates by having the conditional mean of the dependent variable be an affine function of the independent variables. The relationship is modeled through a disturbance term which represents an unobserved random variable that adds noise. The algorithm is typically formulated from the data using the least squares method which seeks to estimate the coefficients by minimizing the squared residual function. The linear equation assigns one scale factor represented by a coefficient to each covariate and an additional coefficient called the intercept or the bias coefficient which gives the line an additional degree of freedom allowing to move up and down a two-dimensional plot.

1. The [linear regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) from the <mark style="background-color: #CCECFF"><b>sklearn.linear_model</b></mark> Python library API was implemented. 
2. The model contains 1 hyperparameter:
    * <span style="color: #FF0000">degree</span> = degree held constant at a value of 1
3. No hyperparameter tuning was conducted. 
4. The apparent model performance of the optimal model is summarized as follows:
    * **R-Squared** = 0.6332
    * **Mean Squared Error** = 0.3550
    * **Mean Absolute Error** = 0.4609
5. The independent test model performance of the final model is summarized as follows:
    * **R-Squared** = 0.6446
    * **Mean Squared Error** = 0.3716
    * **Mean Absolute Error** = 0.4773
6. Apparent and independent test model performance are relatively comparable, indicative of the absence of model overfitting.


```python
##################################
# Defining a pipeline for the 
# linear regression model
##################################
linear_regression_pipeline = Pipeline([('polynomial_features', PolynomialFeatures(include_bias=False, degree=1)), 
                                       ('linear_regression', LinearRegression())])

##################################
# Fitting a linear regression model
##################################
linear_regression_pipeline.fit(X_train, y_train)
```




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;polynomial_features&#x27;,
                 PolynomialFeatures(degree=1, include_bias=False)),
                (&#x27;linear_regression&#x27;, LinearRegression())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" ><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;polynomial_features&#x27;,
                 PolynomialFeatures(degree=1, include_bias=False)),
                (&#x27;linear_regression&#x27;, LinearRegression())])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" ><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PolynomialFeatures<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PolynomialFeatures.html">?<span>Documentation for PolynomialFeatures</span></a></label><div class="sk-toggleable__content fitted"><pre>PolynomialFeatures(degree=1, include_bias=False)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;LinearRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div></div></div>




```python
##################################
# Evaluating the linear regression model
# on the train set
##################################
linear_y_hat_train = linear_regression_pipeline.predict(X_train)

##################################
# Gathering the model evaluation metrics
##################################
linear_performance_train = model_performance_evaluation(y_train, linear_y_hat_train)
linear_performance_train['model'] = ['linear_regression'] * 3
linear_performance_train['set'] = ['train'] * 3
print('Linear Regression Model Performance on Train Data: ')
display(linear_performance_train)
```

    Linear Regression Model Performance on Train Data: 
    


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
      <th>metric_name</th>
      <th>metric_value</th>
      <th>model</th>
      <th>set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R2</td>
      <td>0.6332</td>
      <td>linear_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSE</td>
      <td>0.3550</td>
      <td>linear_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.4609</td>
      <td>linear_regression</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Evaluating the linear regression model
# on the test set
##################################
linear_y_hat_test = linear_regression_pipeline.predict(X_test)

##################################
# Gathering the model evaluation metrics
##################################
linear_performance_test = model_performance_evaluation(y_test, linear_y_hat_test)
linear_performance_test['model'] = ['linear_regression'] * 3
linear_performance_test['set'] = ['test'] * 3
print('Linear Regression Model Performance on Test Data: ')
display(linear_performance_test)
```

    Linear Regression Model Performance on Test Data: 
    


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
      <th>metric_name</th>
      <th>metric_value</th>
      <th>model</th>
      <th>set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R2</td>
      <td>0.6446</td>
      <td>linear_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSE</td>
      <td>0.3716</td>
      <td>linear_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.4773</td>
      <td>linear_regression</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the actual and predicted
# target variables
##################################
figure = plt.figure(figsize=(10,6))
axes = plt.axes()
plt.grid(True)
axes.plot(y_test, 
          linear_y_hat_test, 
          marker='o', 
          ls='', 
          ms=3.0)
lim = (-2, 2)
axes.set(xlabel='Actual Cancer Rate', 
         ylabel='Predicted Cancer Rate', 
         xlim=lim,
         ylim=lim,
         title='Linear Regression Model Prediction Performance');
```


    
![png](output_178_0.png)
    


### 1.6.3 Polynomial Regression <a class="anchor" id="1.6.3"></a>

[Polynomial Regression](https://link.springer.com/book/10.1007/978-1-4757-3462-1) explores the relationship between one or more covariates and a scalar response which is modelled as an nth degree polynomial of the covariates. The algorithm fits a nonlinear relationship between the value of the independent variables and the corresponding conditional mean of the dependent variable. Although polynomial regression fits a nonlinear model to the data, as a statistical estimation problem it is linear, in the sense that the regression function is linear in the unknown parameters that are estimated from the data. For this reason, polynomial regression is considered to be a special case of multiple linear regression.

1. The [polynomial regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) from the <mark style="background-color: #CCECFF"><b>sklearn.linear_model</b></mark> Python library API was implemented. 
2. The model contains 1 hyperparameter:
    * <span style="color: #FF0000">degree</span> = degree held constant at a value of 2
3. No hyperparameter tuning was conducted. 
4. The apparent model performance of the optimal model is summarized as follows:
    * **R-Squared** = 0.7908
    * **Mean Squared Error** = 0.2024
    * **Mean Absolute Error** = 0.3503
5. The independent test model performance of the final model is summarized as follows:
    * **R-Squared** = 0.6324
    * **Mean Squared Error** = 0.3844
    * **Mean Absolute Error** = 0.4867
6. Apparent and independent test model performance are relatively different, indicative of the presence of model overfitting.


```python
##################################
# Defining a pipeline for the 
# polynomial regression model
##################################
polynomial_regression_pipeline = Pipeline([('polynomial_features', PolynomialFeatures(include_bias=False, degree=2)), 
                                           ('polynomial_regression', LinearRegression())])

##################################
# Fitting a polynomial regression model
##################################
polynomial_regression_pipeline.fit(X_train, y_train)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;polynomial_features&#x27;, PolynomialFeatures(include_bias=False)),
                (&#x27;polynomial_regression&#x27;, LinearRegression())])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;polynomial_features&#x27;, PolynomialFeatures(include_bias=False)),
                (&#x27;polynomial_regression&#x27;, LinearRegression())])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PolynomialFeatures<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PolynomialFeatures.html">?<span>Documentation for PolynomialFeatures</span></a></label><div class="sk-toggleable__content fitted"><pre>PolynomialFeatures(include_bias=False)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;LinearRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div></div></div>




```python
##################################
# Evaluating the polynomial regression model
# on the train set
##################################
polynomial_y_hat_train = polynomial_regression_pipeline.predict(X_train)

##################################
# Gathering the model evaluation metrics
##################################
polynomial_performance_train = model_performance_evaluation(y_train, polynomial_y_hat_train)
polynomial_performance_train['model'] = ['polynomial_regression'] * 3
polynomial_performance_train['set'] = ['train'] * 3
print('Polynomial Regression Model Performance on Train Data: ')
display(polynomial_performance_train)
```

    Polynomial Regression Model Performance on Train Data: 
    


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
      <th>metric_name</th>
      <th>metric_value</th>
      <th>model</th>
      <th>set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R2</td>
      <td>0.7908</td>
      <td>polynomial_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSE</td>
      <td>0.2024</td>
      <td>polynomial_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.3503</td>
      <td>polynomial_regression</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Evaluating the polynomial regression model
# on the test set
##################################
polynomial_y_hat_test = polynomial_regression_pipeline.predict(X_test)

##################################
# Gathering the model evaluation metrics
##################################
polynomial_performance_test = model_performance_evaluation(y_test, polynomial_y_hat_test)
polynomial_performance_test['model'] = ['polynomial_regression'] * 3
polynomial_performance_test['set'] = ['test'] * 3
print('Polynomial Regression Model Performance on Test Data: ')
display(polynomial_performance_test)
```

    Polynomial Regression Model Performance on Test Data: 
    


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
      <th>metric_name</th>
      <th>metric_value</th>
      <th>model</th>
      <th>set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R2</td>
      <td>0.6324</td>
      <td>polynomial_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSE</td>
      <td>0.3844</td>
      <td>polynomial_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.4867</td>
      <td>polynomial_regression</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the actual and predicted
# target variables
##################################
figure = plt.figure(figsize=(10,6))
axes = plt.axes()
plt.grid(True)
axes.plot(y_test, 
          polynomial_y_hat_test, 
          marker='o', 
          ls='', 
          ms=3.0)
lim = (-2, 2)
axes.set(xlabel='Actual Cancer Rate', 
         ylabel='Predicted Cancer Rate', 
         xlim=lim,
         ylim=lim,
         title='Polynomial Regression Model Prediction Performance');
```


    
![png](output_183_0.png)
    


### 1.6.4 Ridge Regression <a class="anchor" id="1.6.4"></a>

[Ridge Regression](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634) is a regression technique aimed at preventing overfitting in linear regression models when the model is too complex and fits the training data very closely, but performs poorly on new and unseen data. The algorithm adds a penalty term (referred to as the L2 regularization) to the linear regression cost function that is proportional to the square of the magnitude of the coefficients. This approach helps to reduce the magnitude of the coefficients in the model, which in turn can prevent overfitting and is especially helpful where there are many correlated predictor variables in the model. A hyperparameter alpha serving as a constant that multiplies the L2 term thereby controlling regularization strength needs to be optimized through cross-validation.

1. The [ridge regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) from the <mark style="background-color: #CCECFF"><b>sklearn.linear_model</b></mark> Python library API was implemented. 
2. The model contains 1 hyperparameter:
    * <span style="color: #FF0000">alpha</span> = alpha made to vary across a range of values equal to 0.0001 to 1000
3. Hyperparameter tuning was conducted using the leave-one-out-cross-validation method with optimal model performance determined for: 
    * <span style="color: #FF0000">alpha</span> = 10
4. The apparent model performance of the optimal model is summarized as follows:
    * **R-Squared** = 0.6220
    * **Mean Squared Error** = 0.3659
    * **Mean Absolute Error** = 0.4680
5. The independent test model performance of the final model is summarized as follows:
    * **R-Squared** = 0.6352
    * **Mean Squared Error** = 0.3815
    * **Mean Absolute Error** = 0.4838
6. Apparent and independent test model performance are relatively comparable, indicative of the absence of model overfitting.


```python
##################################
# Defining the hyperparameters
# for the ridge regression model
##################################
alphas = [0.0001,0.001,0.01,0.1,1,10,100,1000]

##################################
# Formulating a string equivalent 
# of the alpha hyperparameter list
##################################
alphas_string = map(str, alphas)
alphas_string = list(alphas_string)

##################################
# Defining a pipeline for the 
# ridge regression model
##################################
ridge_regression_pipeline = Pipeline([('polynomial_features', PolynomialFeatures(include_bias=False, degree=1)), 
                                      ('ridge_regression', RidgeCV(alphas=alphas, cv=None, store_cv_values=True))])

##################################
# Fitting a ridge regression model
##################################
ridge_regression_pipeline.fit(X_train, y_train)
```

    D:\Github_Codes\ProjectPortfolio\Portfolio_Project_40\mlexplore_venv\Lib\site-packages\sklearn\linear_model\_ridge.py:2341: FutureWarning: 'store_cv_values' is deprecated in version 1.5 and will be removed in 1.7. Use 'store_cv_results' instead.
      warnings.warn(
    




<style>#sk-container-id-3 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-3 {
  color: var(--sklearn-color-text);
}

#sk-container-id-3 pre {
  padding: 0;
}

#sk-container-id-3 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-3 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-3 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-3 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-3 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-3 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-3 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-3 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-3 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-3 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-3 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-3 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-3 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-3 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-3 div.sk-label label.sk-toggleable__label,
#sk-container-id-3 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-3 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-3 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-3 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-3 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-3 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-3 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-3 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-3 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-3 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-3 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-3 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;polynomial_features&#x27;,
                 PolynomialFeatures(degree=1, include_bias=False)),
                (&#x27;ridge_regression&#x27;,
                 RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                         store_cv_values=True))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;polynomial_features&#x27;,
                 PolynomialFeatures(degree=1, include_bias=False)),
                (&#x27;ridge_regression&#x27;,
                 RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                         store_cv_values=True))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PolynomialFeatures<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PolynomialFeatures.html">?<span>Documentation for PolynomialFeatures</span></a></label><div class="sk-toggleable__content fitted"><pre>PolynomialFeatures(degree=1, include_bias=False)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-9" type="checkbox" ><label for="sk-estimator-id-9" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;RidgeCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.RidgeCV.html">?<span>Documentation for RidgeCV</span></a></label><div class="sk-toggleable__content fitted"><pre>RidgeCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
        store_cv_values=True)</pre></div> </div></div></div></div></div></div>




```python
##################################
# Determining the optimal alpha
##################################
ridge_regression_pipeline['ridge_regression'].alpha_
```




    np.float64(10.0)




```python
##################################
# Consolidating the LOOCV results
##################################
ridge_regression_pipeline['ridge_regression'].cv_values_
ridge_regression_LOOCV = pd.DataFrame(ridge_regression_pipeline['ridge_regression'].cv_values_,columns=map(str, alphas) )
ridge_regression_LOOCV.index.name = 'case_index'
ridge_regression_LOOCV.reset_index(inplace=True)
ridge_regression_LOOCV = pd.melt(ridge_regression_LOOCV, 
                                 id_vars = ['case_index'], 
                                 value_vars = alphas_string, 
                                 ignore_index = False)
ridge_regression_LOOCV.rename(columns = {'variable':'alpha', 'value':'MSE'}, inplace = True)
display(ridge_regression_LOOCV)
```

    D:\Github_Codes\ProjectPortfolio\Portfolio_Project_40\mlexplore_venv\Lib\site-packages\sklearn\utils\deprecation.py:102: FutureWarning: Attribute `cv_values_` is deprecated in version 1.5 and will be removed in 1.7. Use `cv_results_` instead.
      warnings.warn(msg, category=FutureWarning)
    


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
      <th>case_index</th>
      <th>alpha</th>
      <th>MSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0001</td>
      <td>0.0012</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.0001</td>
      <td>0.0058</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.0001</td>
      <td>0.0872</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.0001</td>
      <td>0.3473</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.0001</td>
      <td>0.1755</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>109</th>
      <td>109</td>
      <td>1000</td>
      <td>0.1383</td>
    </tr>
    <tr>
      <th>110</th>
      <td>110</td>
      <td>1000</td>
      <td>0.5213</td>
    </tr>
    <tr>
      <th>111</th>
      <td>111</td>
      <td>1000</td>
      <td>1.7320</td>
    </tr>
    <tr>
      <th>112</th>
      <td>112</td>
      <td>1000</td>
      <td>0.0289</td>
    </tr>
    <tr>
      <th>113</th>
      <td>113</td>
      <td>1000</td>
      <td>0.0201</td>
    </tr>
  </tbody>
</table>
<p>912 rows × 3 columns</p>
</div>



```python
##################################
# Plotting the ridge regression
# hyperparameter tuning results
# using LOOCV
##################################
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.grid(True)
sns.boxplot(x='alpha', 
            y='MSE', 
            data=ridge_regression_LOOCV, 
            color="#0070C0",
            showmeans=True,
            meanprops={'marker':'o',
                       'markerfacecolor':'#ADD8E6', 
                       'markeredgecolor':'#FF0000',
                       'markersize':'8'})
plt.xlabel('Alpha Hyperparameter')
plt.ylabel('Mean Squared Error')
plt.title('Ridge Regression Hyperparameter Tuning Using LOOCV')
plt.show()
```


    
![png](output_188_0.png)
    



```python
##################################
# Evaluating the ridge regression model
# on the train set
##################################
ridge_y_hat_train = ridge_regression_pipeline.predict(X_train)

##################################
# Gathering the model evaluation metrics
##################################
ridge_performance_train = model_performance_evaluation(y_train, ridge_y_hat_train)
ridge_performance_train['model'] = ['ridge_regression'] * 3
ridge_performance_train['set'] = ['train'] * 3
print('Ridge Regression Model Performance on Train Data: ')
display(ridge_performance_train)
```

    Ridge Regression Model Performance on Train Data: 
    


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
      <th>metric_name</th>
      <th>metric_value</th>
      <th>model</th>
      <th>set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R2</td>
      <td>0.6220</td>
      <td>ridge_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSE</td>
      <td>0.3659</td>
      <td>ridge_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.4680</td>
      <td>ridge_regression</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Evaluating the ridge regression model
# on the test set
##################################
ridge_y_hat_test = ridge_regression_pipeline.predict(X_test)

##################################
# Gathering the model evaluation metrics
##################################
ridge_performance_test = model_performance_evaluation(y_test, ridge_y_hat_test)
ridge_performance_test['model'] = ['ridge_regression'] * 3
ridge_performance_test['set'] = ['test'] * 3
print('Ridge Regression Model Performance on Test Data: ')
display(ridge_performance_test)
```

    Ridge Regression Model Performance on Test Data: 
    


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
      <th>metric_name</th>
      <th>metric_value</th>
      <th>model</th>
      <th>set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R2</td>
      <td>0.6352</td>
      <td>ridge_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSE</td>
      <td>0.3815</td>
      <td>ridge_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.4838</td>
      <td>ridge_regression</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the actual and predicted
# target variables
##################################
figure = plt.figure(figsize=(10,6))
axes = plt.axes()
plt.grid(True)
axes.plot(y_test, 
          ridge_y_hat_test, 
          marker='o', 
          ls='', 
          ms=3.0)
lim = (-2, 2)
axes.set(xlabel='Actual Cancer Rate', 
         ylabel='Predicted Cancer Rate', 
         xlim=lim,
         ylim=lim,
         title='Ridge Regression Model Prediction Performance');
```


    
![png](output_191_0.png)
    


### 1.6.5 Least Absolute Shrinkage and Selection Operator Regression <a class="anchor" id="1.6.5"></a>

[Least Absolute Shrinkage and Selection Operator Regression](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1996.tb02080.x) is a regression technique aimed at preventing overfitting in linear regression models when the model is too complex and fits the training data very closely, but performs poorly on new and unseen data. The algorithm adds a penalty term (referred to as the L1 regularization) to the linear regression cost function that is proportional to the absolute value of the coefficients. This approach can be useful for feature selection, as it tends to shrink the coefficients of less important predictor variables to zero, which can help simplify the model and improve its interpretability. A hyperparameter alpha serving as a constant that multiplies the L1 term thereby controlling regularization strength needs to be optimized through cross-validation.

1. The [lasso regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) from the <mark style="background-color: #CCECFF"><b>sklearn.linear_model</b></mark> Python library API was implemented. 
2. The model contains 1 hyperparameter:
    * <span style="color: #FF0000">alpha</span> = alpha made to vary across a range of values equal to 0.0001 to 1000
3. Hyperparameter tuning was conducted using the leave-one-out-cross-validation method with optimal model performance determined for: 
    * <span style="color: #FF0000">alpha</span> = 0.01
4. The apparent model performance of the optimal model is summarized as follows:
    * **R-Squared** = 0.6295
    * **Mean Squared Error** = 0.3586
    * **Mean Absolute Error** = 0.4608
5. The independent test model performance of the final model is summarized as follows:
    * **R-Squared** = 0.6405
    * **Mean Squared Error** = 0.3760
    * **Mean Absolute Error** = 0.4805
6. Apparent and independent test model performance are relatively comparable, indicative of the absence of model overfitting.


```python
##################################
# Defining the hyperparameters
# for the lasso regression model
##################################
alphas = [0.0001,0.001,0.01,0.1,1,10,100,1000]

##################################
# Formulating a string equivalent 
# of the alpha hyperparameter list
##################################
alphas_string = map(str, alphas)
alphas_string = list(alphas_string)

##################################
# Defining a pipeline for the 
# lasso regression model
##################################
lasso_regression_pipeline = Pipeline([('polynomial_features', PolynomialFeatures(include_bias=False, degree=1)), 
                                      ('lasso_regression', LassoCV(alphas=alphas, cv=LeaveOneOut()))])

##################################
# Fitting a lasso regression model
##################################
lasso_regression_pipeline.fit(X_train, y_train)
```




<style>#sk-container-id-4 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-4 {
  color: var(--sklearn-color-text);
}

#sk-container-id-4 pre {
  padding: 0;
}

#sk-container-id-4 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-4 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-4 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-4 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-4 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-4 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-4 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-4 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-4 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-4 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-4 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-4 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-4 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-4 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-4 div.sk-label label.sk-toggleable__label,
#sk-container-id-4 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-4 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-4 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-4 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-4 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-4 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-4 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-4 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-4 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-4 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-4 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-4 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;polynomial_features&#x27;,
                 PolynomialFeatures(degree=1, include_bias=False)),
                (&#x27;lasso_regression&#x27;,
                 LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                         cv=LeaveOneOut()))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-10" type="checkbox" ><label for="sk-estimator-id-10" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;polynomial_features&#x27;,
                 PolynomialFeatures(degree=1, include_bias=False)),
                (&#x27;lasso_regression&#x27;,
                 LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
                         cv=LeaveOneOut()))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-11" type="checkbox" ><label for="sk-estimator-id-11" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PolynomialFeatures<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PolynomialFeatures.html">?<span>Documentation for PolynomialFeatures</span></a></label><div class="sk-toggleable__content fitted"><pre>PolynomialFeatures(degree=1, include_bias=False)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-12" type="checkbox" ><label for="sk-estimator-id-12" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;LassoCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LassoCV.html">?<span>Documentation for LassoCV</span></a></label><div class="sk-toggleable__content fitted"><pre>LassoCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000], cv=LeaveOneOut())</pre></div> </div></div></div></div></div></div>




```python
##################################
# Determining the optimal alpha
##################################
lasso_regression_pipeline['lasso_regression'].alpha_
```




    np.float64(0.01)




```python
##################################
# Consolidating the LOOCV results
##################################
lasso_regression_pipeline['lasso_regression'].mse_path_
lasso_regression_LOOCV = pd.DataFrame(lasso_regression_pipeline['lasso_regression'].mse_path_.transpose())
lasso_regression_LOOCV = lasso_regression_LOOCV[[7,6,5,4,3,2,1,0]]
lasso_regression_LOOCV = lasso_regression_LOOCV.set_axis(map(str, alphas), axis=1)
lasso_regression_LOOCV.index.name = 'case_index'
lasso_regression_LOOCV.reset_index(inplace=True)
display(lasso_regression_LOOCV)
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
      <th>case_index</th>
      <th>0.0001</th>
      <th>0.001</th>
      <th>0.01</th>
      <th>0.1</th>
      <th>1</th>
      <th>10</th>
      <th>100</th>
      <th>1000</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0011</td>
      <td>0.0007</td>
      <td>0.0009</td>
      <td>0.0000</td>
      <td>0.0020</td>
      <td>0.0020</td>
      <td>0.0020</td>
      <td>0.0020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.0058</td>
      <td>0.0057</td>
      <td>0.0053</td>
      <td>0.0074</td>
      <td>1.5211</td>
      <td>1.5211</td>
      <td>1.5211</td>
      <td>1.5211</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.0872</td>
      <td>0.0872</td>
      <td>0.0894</td>
      <td>0.2171</td>
      <td>1.5328</td>
      <td>1.5328</td>
      <td>1.5328</td>
      <td>1.5328</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.3470</td>
      <td>0.3452</td>
      <td>0.3919</td>
      <td>0.3913</td>
      <td>0.1931</td>
      <td>0.1931</td>
      <td>0.1931</td>
      <td>0.1931</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.1760</td>
      <td>0.1800</td>
      <td>0.2250</td>
      <td>0.5213</td>
      <td>2.3015</td>
      <td>2.3015</td>
      <td>2.3015</td>
      <td>2.3015</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>109</th>
      <td>109</td>
      <td>0.1380</td>
      <td>0.1374</td>
      <td>0.1589</td>
      <td>0.2133</td>
      <td>0.1710</td>
      <td>0.1710</td>
      <td>0.1710</td>
      <td>0.1710</td>
    </tr>
    <tr>
      <th>110</th>
      <td>110</td>
      <td>0.0088</td>
      <td>0.0075</td>
      <td>0.0007</td>
      <td>0.0089</td>
      <td>1.1168</td>
      <td>1.1168</td>
      <td>1.1168</td>
      <td>1.1168</td>
    </tr>
    <tr>
      <th>111</th>
      <td>111</td>
      <td>2.1149</td>
      <td>2.1371</td>
      <td>2.3557</td>
      <td>2.5285</td>
      <td>1.1061</td>
      <td>1.1061</td>
      <td>1.1061</td>
      <td>1.1061</td>
    </tr>
    <tr>
      <th>112</th>
      <td>112</td>
      <td>0.0013</td>
      <td>0.0020</td>
      <td>0.0164</td>
      <td>0.0582</td>
      <td>0.0015</td>
      <td>0.0015</td>
      <td>0.0015</td>
      <td>0.0015</td>
    </tr>
    <tr>
      <th>113</th>
      <td>113</td>
      <td>0.2060</td>
      <td>0.2151</td>
      <td>0.2999</td>
      <td>0.2414</td>
      <td>0.2579</td>
      <td>0.2579</td>
      <td>0.2579</td>
      <td>0.2579</td>
    </tr>
  </tbody>
</table>
<p>114 rows × 9 columns</p>
</div>



```python
##################################
# Transforming the dataframe
# detailing the LOOCV results
##################################
lasso_regression_LOOCV = pd.melt(lasso_regression_LOOCV, 
                                 id_vars = ['case_index'], 
                                 value_vars = alphas_string, 
                                 ignore_index = False)
lasso_regression_LOOCV.rename(columns = {'variable':'alpha', 'value':'MSE'}, inplace = True)
display(lasso_regression_LOOCV)
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
      <th>case_index</th>
      <th>alpha</th>
      <th>MSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.0001</td>
      <td>0.0011</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.0001</td>
      <td>0.0058</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.0001</td>
      <td>0.0872</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0.0001</td>
      <td>0.3470</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>0.0001</td>
      <td>0.1760</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>109</th>
      <td>109</td>
      <td>1000</td>
      <td>0.1710</td>
    </tr>
    <tr>
      <th>110</th>
      <td>110</td>
      <td>1000</td>
      <td>1.1168</td>
    </tr>
    <tr>
      <th>111</th>
      <td>111</td>
      <td>1000</td>
      <td>1.1061</td>
    </tr>
    <tr>
      <th>112</th>
      <td>112</td>
      <td>1000</td>
      <td>0.0015</td>
    </tr>
    <tr>
      <th>113</th>
      <td>113</td>
      <td>1000</td>
      <td>0.2579</td>
    </tr>
  </tbody>
</table>
<p>912 rows × 3 columns</p>
</div>



```python
##################################
# Plotting the lasso regression
# hyperparameter tuning results
# using LOOCV
##################################
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.grid(True)
sns.boxplot(x='alpha', 
            y='MSE', 
            data=lasso_regression_LOOCV, 
            color='#0070C0',
            showmeans=True,
            meanprops={'marker':'o',
                       'markerfacecolor':'#ADD8E6', 
                       'markeredgecolor':'#FF0000',
                       'markersize':'8'})
plt.xlabel('Alpha Hyperparameter')
plt.ylabel('Mean Squared Error')
plt.title('Lasso Regression Hyperparameter Tuning Using LOOCV')
plt.show()
```


    
![png](output_197_0.png)
    



```python
##################################
# Evaluating the lasso regression model
# on the train set
##################################
lasso_y_hat_train = lasso_regression_pipeline.predict(X_train)

##################################
# Gathering the model evaluation metrics
##################################
lasso_performance_train = model_performance_evaluation(y_train, lasso_y_hat_train)
lasso_performance_train['model'] = ['lasso_regression'] * 3
lasso_performance_train['set'] = ['train'] * 3
print('Lasso Regression Model Performance on Train Data: ')
display(lasso_performance_train)
```

    Lasso Regression Model Performance on Train Data: 
    


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
      <th>metric_name</th>
      <th>metric_value</th>
      <th>model</th>
      <th>set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R2</td>
      <td>0.6295</td>
      <td>lasso_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSE</td>
      <td>0.3586</td>
      <td>lasso_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.4608</td>
      <td>lasso_regression</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Evaluating the lasso regression model
# on the test set
##################################
lasso_y_hat_test = lasso_regression_pipeline.predict(X_test)

##################################
# Gathering the model evaluation metrics
##################################
lasso_performance_test = model_performance_evaluation(y_test, lasso_y_hat_test)
lasso_performance_test['model'] = ['lasso_regression'] * 3
lasso_performance_test['set'] = ['test'] * 3
print('Lasso Regression Model Performance on Test Data: ')
display(lasso_performance_test)
```

    Lasso Regression Model Performance on Test Data: 
    


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
      <th>metric_name</th>
      <th>metric_value</th>
      <th>model</th>
      <th>set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R2</td>
      <td>0.6405</td>
      <td>lasso_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSE</td>
      <td>0.3760</td>
      <td>lasso_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.4805</td>
      <td>lasso_regression</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the actual and predicted
# target variables
##################################
figure = plt.figure(figsize=(10,6))
axes = plt.axes()
plt.grid(True)
axes.plot(y_test, 
          lasso_y_hat_test, 
          marker='o', 
          ls='', 
          ms=3.0)
lim = (-2, 2)
axes.set(xlabel='Actual Cancer Rate', 
         ylabel='Predicted Cancer Rate', 
         xlim=lim,
         ylim=lim,
         title='Lasso Regression Model Prediction Performance');
```


    
![png](output_200_0.png)
    


### 1.6.6 Elastic Net Regression <a class="anchor" id="1.6.6"></a>

[Elastic Net Regression](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2005.00503.x) is a regression technique aimed at preventing overfitting in linear regression models when the model is too complex and fits the training data very closely, but performs poorly on new and unseen data. The algorithm is a combination of the ridge and LASSO regression methods, by adding both L1 and L2 regularization terms in the cost function. This approach can be useful when there are many predictor variables that are correlated with the response variable, but only a subset of them are truly important for predicting the response. The L1 regularization term can help to select the important variables, while the L2 regularization term can help to reduce the magnitude of the coefficients. Hyperparameters alpha which serves as the constant that multiplies the penalty terms, and l1_ratio that serves as the mixing parameter that penalizes as a combination of L1 and L2 regularization - need to be optimized through cross-validation.

1. The [elastic net regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) from the <mark style="background-color: #CCECFF"><b>sklearn.linear_model</b></mark> Python library API was implemented. 
2. The model contains 2 hyperparameters:
    * <span style="color: #FF0000">l1_ratio</span> = l1_ratio made to vary across a range of values equal to 0.1 to 0.9
    * <span style="color: #FF0000">alpha</span> = alpha made to vary across a range of values equal to 0.0001 to 1000
3. Hyperparameter tuning was conducted using the leave-one-out-cross-validation method with optimal model performance determined for: 
    * <span style="color: #FF0000">l1_ratio</span> = 0.9
    * <span style="color: #FF0000">alpha</span> = 0.01
4. The apparent model performance of the optimal model is summarized as follows:
    * **R-Squared** = 0.6298
    * **Mean Squared Error** = 0.3582
    * **Mean Absolute Error** = 0.4608
5. The independent test model performance of the final model is summarized as follows:
    * **R-Squared** = 0.6409
    * **Mean Squared Error** = 0.3755
    * **Mean Absolute Error** = 0.4800
6. Apparent and independent test model performance are relatively comparable, indicative of the absence of model overfitting.


```python
##################################
# Defining the hyperparameters
# for the elastic-net regression model
##################################
l1_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
alphas = [0.0001,0.001,0.01,0.1,1,10,100,1000]

##################################
# Formulating a string equivalent 
# of the l1_ratio hyperparameter list
##################################
l1_ratios_string = map(str, l1_ratios)
l1_ratios_string_reversed = list(l1_ratios_string)
l1_ratios_string_reversed.reverse()

##################################
# Formulating a string equivalent 
# of the alpha hyperparameter list
##################################
alphas_string = map(str, alphas)
alphas_string_reversed = list(alphas_string)
alphas_string_reversed.reverse()

##################################
# Defining a pipeline for the 
# elastic-net regression model
##################################
elasticnet_regression_pipeline = Pipeline([('polynomial_features', PolynomialFeatures(include_bias=False, degree=1)), 
                                           ('elasticnet_regression', ElasticNetCV(alphas=alphas, l1_ratio=l1_ratios, cv=LeaveOneOut()))])

##################################
# Fitting an elastic-net regression model
##################################
elasticnet_regression_pipeline.fit(X_train, y_train)
```




<style>#sk-container-id-5 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-5 {
  color: var(--sklearn-color-text);
}

#sk-container-id-5 pre {
  padding: 0;
}

#sk-container-id-5 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-5 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-5 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-5 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-5 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-5 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-5 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-5 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-5 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-5 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-5 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-5 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-5 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-5 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-5 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-5 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-5 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-5 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-5 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-5 div.sk-label label.sk-toggleable__label,
#sk-container-id-5 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-5 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-5 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-5 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-5 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-5 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-5 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-5 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-5 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-5 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-5 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-5 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-5 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-5" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Pipeline(steps=[(&#x27;polynomial_features&#x27;,
                 PolynomialFeatures(degree=1, include_bias=False)),
                (&#x27;elasticnet_regression&#x27;,
                 ElasticNetCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                                      1000],
                              cv=LeaveOneOut(),
                              l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                        0.9]))])</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-13" type="checkbox" ><label for="sk-estimator-id-13" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Pipeline<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.pipeline.Pipeline.html">?<span>Documentation for Pipeline</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Pipeline(steps=[(&#x27;polynomial_features&#x27;,
                 PolynomialFeatures(degree=1, include_bias=False)),
                (&#x27;elasticnet_regression&#x27;,
                 ElasticNetCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100,
                                      1000],
                              cv=LeaveOneOut(),
                              l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                        0.9]))])</pre></div> </div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-14" type="checkbox" ><label for="sk-estimator-id-14" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;PolynomialFeatures<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.PolynomialFeatures.html">?<span>Documentation for PolynomialFeatures</span></a></label><div class="sk-toggleable__content fitted"><pre>PolynomialFeatures(degree=1, include_bias=False)</pre></div> </div></div><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-15" type="checkbox" ><label for="sk-estimator-id-15" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;ElasticNetCV<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.ElasticNetCV.html">?<span>Documentation for ElasticNetCV</span></a></label><div class="sk-toggleable__content fitted"><pre>ElasticNetCV(alphas=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000],
             cv=LeaveOneOut(),
             l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])</pre></div> </div></div></div></div></div></div>




```python
##################################
# Determining the optimal alpha
##################################
elasticnet_regression_pipeline['elasticnet_regression'].alpha_
```




    np.float64(0.01)




```python
##################################
# Determining the optimal l1_ratio
##################################
elasticnet_regression_pipeline['elasticnet_regression'].l1_ratio_
```




    np.float64(0.9)




```python
##################################
# Consolidating the LOOCV results
##################################
elasticnet_regression_LOOCV_raw = elasticnet_regression_pipeline['elasticnet_regression'].mse_path_
elasticnet_regression_LOOCV = pd.DataFrame(elasticnet_regression_LOOCV_raw.reshape(-1, 114))
elasticnet_regression_LOOCV.index = np.repeat(np.arange(elasticnet_regression_LOOCV_raw.shape[0]), elasticnet_regression_LOOCV_raw.shape[1]) + 1
elasticnet_regression_LOOCV.index.name = 'l1_ratio'
elasticnet_regression_LOOCV.reset_index(inplace=True)
```


```python
##################################
# Creating a dataframe based on l1_ratio
# from the LOOCV results
##################################
elasticnet_regression_LOOCV_l1_ratio = elasticnet_regression_LOOCV
display(elasticnet_regression_LOOCV_l1_ratio)
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
      <th>l1_ratio</th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>...</th>
      <th>104</th>
      <th>105</th>
      <th>106</th>
      <th>107</th>
      <th>108</th>
      <th>109</th>
      <th>110</th>
      <th>111</th>
      <th>112</th>
      <th>113</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.0020</td>
      <td>1.5211</td>
      <td>1.5328</td>
      <td>0.1931</td>
      <td>2.3015</td>
      <td>0.6067</td>
      <td>0.2884</td>
      <td>1.2766</td>
      <td>0.4167</td>
      <td>...</td>
      <td>0.0133</td>
      <td>0.9307</td>
      <td>2.7951</td>
      <td>2.2956</td>
      <td>0.6066</td>
      <td>0.1710</td>
      <td>1.1168</td>
      <td>1.1061</td>
      <td>0.0015</td>
      <td>0.2579</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.0020</td>
      <td>1.5211</td>
      <td>1.5328</td>
      <td>0.1931</td>
      <td>2.3015</td>
      <td>0.6067</td>
      <td>0.2884</td>
      <td>1.2766</td>
      <td>0.4167</td>
      <td>...</td>
      <td>0.0133</td>
      <td>0.9307</td>
      <td>2.7951</td>
      <td>2.2956</td>
      <td>0.6066</td>
      <td>0.1710</td>
      <td>1.1168</td>
      <td>1.1061</td>
      <td>0.0015</td>
      <td>0.2579</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.0020</td>
      <td>1.5211</td>
      <td>1.5328</td>
      <td>0.1931</td>
      <td>2.3015</td>
      <td>0.6067</td>
      <td>0.2884</td>
      <td>1.2766</td>
      <td>0.4167</td>
      <td>...</td>
      <td>0.0133</td>
      <td>0.9307</td>
      <td>2.7951</td>
      <td>2.2956</td>
      <td>0.6066</td>
      <td>0.1710</td>
      <td>1.1168</td>
      <td>1.1061</td>
      <td>0.0015</td>
      <td>0.2579</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.0022</td>
      <td>0.1173</td>
      <td>0.3284</td>
      <td>0.2850</td>
      <td>0.8757</td>
      <td>0.8514</td>
      <td>0.1458</td>
      <td>0.2705</td>
      <td>0.0295</td>
      <td>...</td>
      <td>0.0014</td>
      <td>1.8480</td>
      <td>1.5636</td>
      <td>0.6217</td>
      <td>0.3866</td>
      <td>0.1205</td>
      <td>0.1191</td>
      <td>2.3643</td>
      <td>0.0781</td>
      <td>0.0792</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.0049</td>
      <td>0.0010</td>
      <td>0.0994</td>
      <td>0.3206</td>
      <td>0.3641</td>
      <td>0.8689</td>
      <td>0.5433</td>
      <td>0.0626</td>
      <td>0.0132</td>
      <td>...</td>
      <td>0.0081</td>
      <td>2.0241</td>
      <td>1.5584</td>
      <td>0.1701</td>
      <td>0.3002</td>
      <td>0.1235</td>
      <td>0.0001</td>
      <td>2.6688</td>
      <td>0.0987</td>
      <td>0.3930</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>67</th>
      <td>9</td>
      <td>0.0020</td>
      <td>1.5211</td>
      <td>1.5328</td>
      <td>0.1931</td>
      <td>2.3015</td>
      <td>0.6067</td>
      <td>0.2884</td>
      <td>1.2766</td>
      <td>0.4167</td>
      <td>...</td>
      <td>0.0133</td>
      <td>0.9307</td>
      <td>2.7951</td>
      <td>2.2956</td>
      <td>0.6066</td>
      <td>0.1710</td>
      <td>1.1168</td>
      <td>1.1061</td>
      <td>0.0015</td>
      <td>0.2579</td>
    </tr>
    <tr>
      <th>68</th>
      <td>9</td>
      <td>0.0001</td>
      <td>0.0051</td>
      <td>0.2029</td>
      <td>0.3840</td>
      <td>0.5113</td>
      <td>0.8459</td>
      <td>0.4257</td>
      <td>0.1601</td>
      <td>0.0067</td>
      <td>...</td>
      <td>0.0027</td>
      <td>1.9887</td>
      <td>1.6780</td>
      <td>0.2703</td>
      <td>0.2514</td>
      <td>0.2032</td>
      <td>0.0066</td>
      <td>2.5404</td>
      <td>0.0650</td>
      <td>0.2614</td>
    </tr>
    <tr>
      <th>69</th>
      <td>9</td>
      <td>0.0007</td>
      <td>0.0052</td>
      <td>0.0875</td>
      <td>0.3870</td>
      <td>0.2200</td>
      <td>0.8004</td>
      <td>0.7328</td>
      <td>0.0480</td>
      <td>0.0338</td>
      <td>...</td>
      <td>0.0085</td>
      <td>1.7579</td>
      <td>1.8444</td>
      <td>0.1508</td>
      <td>0.3006</td>
      <td>0.1555</td>
      <td>0.0007</td>
      <td>2.3405</td>
      <td>0.0158</td>
      <td>0.2965</td>
    </tr>
    <tr>
      <th>70</th>
      <td>9</td>
      <td>0.0007</td>
      <td>0.0057</td>
      <td>0.0872</td>
      <td>0.3448</td>
      <td>0.1798</td>
      <td>0.7830</td>
      <td>0.7569</td>
      <td>0.0328</td>
      <td>0.0315</td>
      <td>...</td>
      <td>0.0233</td>
      <td>1.5690</td>
      <td>2.0451</td>
      <td>0.1154</td>
      <td>0.2950</td>
      <td>0.1373</td>
      <td>0.0076</td>
      <td>2.1355</td>
      <td>0.0020</td>
      <td>0.2146</td>
    </tr>
    <tr>
      <th>71</th>
      <td>9</td>
      <td>0.0011</td>
      <td>0.0058</td>
      <td>0.0872</td>
      <td>0.3470</td>
      <td>0.1760</td>
      <td>0.7811</td>
      <td>0.7633</td>
      <td>0.0314</td>
      <td>0.0313</td>
      <td>...</td>
      <td>0.0253</td>
      <td>1.5500</td>
      <td>2.0788</td>
      <td>0.1121</td>
      <td>0.2942</td>
      <td>0.1381</td>
      <td>0.0088</td>
      <td>2.1147</td>
      <td>0.0013</td>
      <td>0.2060</td>
    </tr>
  </tbody>
</table>
<p>72 rows × 115 columns</p>
</div>



```python
##################################
# Creating a dataframe based on alpha
# from the LOOCV results
##################################
elasticnet_regression_LOOCV_alpha = elasticnet_regression_LOOCV.drop(['l1_ratio'], axis=1)
elasticnet_regression_LOOCV_alpha['alpha'] = list(range(1,9))*9
display(elasticnet_regression_LOOCV_alpha)
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>105</th>
      <th>106</th>
      <th>107</th>
      <th>108</th>
      <th>109</th>
      <th>110</th>
      <th>111</th>
      <th>112</th>
      <th>113</th>
      <th>alpha</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0020</td>
      <td>1.5211</td>
      <td>1.5328</td>
      <td>0.1931</td>
      <td>2.3015</td>
      <td>0.6067</td>
      <td>0.2884</td>
      <td>1.2766</td>
      <td>0.4167</td>
      <td>0.2884</td>
      <td>...</td>
      <td>0.9307</td>
      <td>2.7951</td>
      <td>2.2956</td>
      <td>0.6066</td>
      <td>0.1710</td>
      <td>1.1168</td>
      <td>1.1061</td>
      <td>0.0015</td>
      <td>0.2579</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.0020</td>
      <td>1.5211</td>
      <td>1.5328</td>
      <td>0.1931</td>
      <td>2.3015</td>
      <td>0.6067</td>
      <td>0.2884</td>
      <td>1.2766</td>
      <td>0.4167</td>
      <td>0.2884</td>
      <td>...</td>
      <td>0.9307</td>
      <td>2.7951</td>
      <td>2.2956</td>
      <td>0.6066</td>
      <td>0.1710</td>
      <td>1.1168</td>
      <td>1.1061</td>
      <td>0.0015</td>
      <td>0.2579</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0020</td>
      <td>1.5211</td>
      <td>1.5328</td>
      <td>0.1931</td>
      <td>2.3015</td>
      <td>0.6067</td>
      <td>0.2884</td>
      <td>1.2766</td>
      <td>0.4167</td>
      <td>0.2884</td>
      <td>...</td>
      <td>0.9307</td>
      <td>2.7951</td>
      <td>2.2956</td>
      <td>0.6066</td>
      <td>0.1710</td>
      <td>1.1168</td>
      <td>1.1061</td>
      <td>0.0015</td>
      <td>0.2579</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0022</td>
      <td>0.1173</td>
      <td>0.3284</td>
      <td>0.2850</td>
      <td>0.8757</td>
      <td>0.8514</td>
      <td>0.1458</td>
      <td>0.2705</td>
      <td>0.0295</td>
      <td>0.0002</td>
      <td>...</td>
      <td>1.8480</td>
      <td>1.5636</td>
      <td>0.6217</td>
      <td>0.3866</td>
      <td>0.1205</td>
      <td>0.1191</td>
      <td>2.3643</td>
      <td>0.0781</td>
      <td>0.0792</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.0049</td>
      <td>0.0010</td>
      <td>0.0994</td>
      <td>0.3206</td>
      <td>0.3641</td>
      <td>0.8689</td>
      <td>0.5433</td>
      <td>0.0626</td>
      <td>0.0132</td>
      <td>0.0433</td>
      <td>...</td>
      <td>2.0241</td>
      <td>1.5584</td>
      <td>0.1701</td>
      <td>0.3002</td>
      <td>0.1235</td>
      <td>0.0001</td>
      <td>2.6688</td>
      <td>0.0987</td>
      <td>0.3930</td>
      <td>5</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>67</th>
      <td>0.0020</td>
      <td>1.5211</td>
      <td>1.5328</td>
      <td>0.1931</td>
      <td>2.3015</td>
      <td>0.6067</td>
      <td>0.2884</td>
      <td>1.2766</td>
      <td>0.4167</td>
      <td>0.2884</td>
      <td>...</td>
      <td>0.9307</td>
      <td>2.7951</td>
      <td>2.2956</td>
      <td>0.6066</td>
      <td>0.1710</td>
      <td>1.1168</td>
      <td>1.1061</td>
      <td>0.0015</td>
      <td>0.2579</td>
      <td>4</td>
    </tr>
    <tr>
      <th>68</th>
      <td>0.0001</td>
      <td>0.0051</td>
      <td>0.2029</td>
      <td>0.3840</td>
      <td>0.5113</td>
      <td>0.8459</td>
      <td>0.4257</td>
      <td>0.1601</td>
      <td>0.0067</td>
      <td>0.0189</td>
      <td>...</td>
      <td>1.9887</td>
      <td>1.6780</td>
      <td>0.2703</td>
      <td>0.2514</td>
      <td>0.2032</td>
      <td>0.0066</td>
      <td>2.5404</td>
      <td>0.0650</td>
      <td>0.2614</td>
      <td>5</td>
    </tr>
    <tr>
      <th>69</th>
      <td>0.0007</td>
      <td>0.0052</td>
      <td>0.0875</td>
      <td>0.3870</td>
      <td>0.2200</td>
      <td>0.8004</td>
      <td>0.7328</td>
      <td>0.0480</td>
      <td>0.0338</td>
      <td>0.0796</td>
      <td>...</td>
      <td>1.7579</td>
      <td>1.8444</td>
      <td>0.1508</td>
      <td>0.3006</td>
      <td>0.1555</td>
      <td>0.0007</td>
      <td>2.3405</td>
      <td>0.0158</td>
      <td>0.2965</td>
      <td>6</td>
    </tr>
    <tr>
      <th>70</th>
      <td>0.0007</td>
      <td>0.0057</td>
      <td>0.0872</td>
      <td>0.3448</td>
      <td>0.1798</td>
      <td>0.7830</td>
      <td>0.7569</td>
      <td>0.0328</td>
      <td>0.0315</td>
      <td>0.0752</td>
      <td>...</td>
      <td>1.5690</td>
      <td>2.0451</td>
      <td>0.1154</td>
      <td>0.2950</td>
      <td>0.1373</td>
      <td>0.0076</td>
      <td>2.1355</td>
      <td>0.0020</td>
      <td>0.2146</td>
      <td>7</td>
    </tr>
    <tr>
      <th>71</th>
      <td>0.0011</td>
      <td>0.0058</td>
      <td>0.0872</td>
      <td>0.3470</td>
      <td>0.1760</td>
      <td>0.7811</td>
      <td>0.7633</td>
      <td>0.0314</td>
      <td>0.0313</td>
      <td>0.0750</td>
      <td>...</td>
      <td>1.5500</td>
      <td>2.0788</td>
      <td>0.1121</td>
      <td>0.2942</td>
      <td>0.1381</td>
      <td>0.0088</td>
      <td>2.1147</td>
      <td>0.0013</td>
      <td>0.2060</td>
      <td>8</td>
    </tr>
  </tbody>
</table>
<p>72 rows × 115 columns</p>
</div>



```python
##################################
# Transforming the l1_ratio dataframe
# detailing the LOOCV results
##################################
elasticnet_regression_LOOCV_l1_ratio = pd.melt(elasticnet_regression_LOOCV_l1_ratio, 
                                               id_vars=['l1_ratio'], 
                                               value_vars=list(range(0,114)), 
                                               ignore_index=False)
elasticnet_regression_LOOCV_l1_ratio.rename(columns = {'variable':'case_index', 'value':'MSE'}, inplace = True)
display(elasticnet_regression_LOOCV_l1_ratio)
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
      <th>l1_ratio</th>
      <th>case_index</th>
      <th>MSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0.0020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0.0020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0.0020</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0.0022</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0.0049</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>67</th>
      <td>9</td>
      <td>113</td>
      <td>0.2579</td>
    </tr>
    <tr>
      <th>68</th>
      <td>9</td>
      <td>113</td>
      <td>0.2614</td>
    </tr>
    <tr>
      <th>69</th>
      <td>9</td>
      <td>113</td>
      <td>0.2965</td>
    </tr>
    <tr>
      <th>70</th>
      <td>9</td>
      <td>113</td>
      <td>0.2146</td>
    </tr>
    <tr>
      <th>71</th>
      <td>9</td>
      <td>113</td>
      <td>0.2060</td>
    </tr>
  </tbody>
</table>
<p>8208 rows × 3 columns</p>
</div>



```python
##################################
# Renaming the l1_ratio levels
# based on the proper category labels
##################################
elasticnet_regression_LOOCV_l1_ratio_conditions = [
    (elasticnet_regression_LOOCV_l1_ratio['l1_ratio'] == 1),
    (elasticnet_regression_LOOCV_l1_ratio['l1_ratio'] == 2),
    (elasticnet_regression_LOOCV_l1_ratio['l1_ratio'] == 3),
    (elasticnet_regression_LOOCV_l1_ratio['l1_ratio'] == 4),
    (elasticnet_regression_LOOCV_l1_ratio['l1_ratio'] == 5),
    (elasticnet_regression_LOOCV_l1_ratio['l1_ratio'] == 6),
    (elasticnet_regression_LOOCV_l1_ratio['l1_ratio'] == 7),
    (elasticnet_regression_LOOCV_l1_ratio['l1_ratio'] == 8),
    (elasticnet_regression_LOOCV_l1_ratio['l1_ratio'] == 9)]

elasticnet_regression_LOOCV_l1_ratio_values = l1_ratios_string_reversed
elasticnet_regression_LOOCV_l1_ratio['l1_ratio'] = np.select(elasticnet_regression_LOOCV_l1_ratio_conditions,
                                                             elasticnet_regression_LOOCV_l1_ratio_values,
                                                             default=str(np.nan))
elasticnet_regression_LOOCV_l1_ratio['l1_ratio'] = elasticnet_regression_LOOCV_l1_ratio['l1_ratio'].astype('category')
```


```python
##################################
# Plotting the elasticnet regression
# hyperparameter tuning results
# using LOOCV
##################################
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.grid(True)
sns.boxplot(x='l1_ratio', 
            y='MSE', 
            data=elasticnet_regression_LOOCV_l1_ratio, 
            color='#0070C0', 
            showmeans=True,
            meanprops={'marker':'o',
                       'markerfacecolor':'#ADD8E6', 
                       'markeredgecolor':'#FF0000',
                       'markersize':'8'})
plt.xlabel('L1 Ratio Hyperparameter')
plt.ylabel('Mean Squared Error')
plt.title('Elastic Net Regression Hyperparameter Tuning Using LOOCV')
plt.show()
```


    
![png](output_210_0.png)
    



```python
##################################
# Transforming the alpha dataframe
# detailing the LOOCV results
##################################
elasticnet_regression_LOOCV_alpha = pd.melt(elasticnet_regression_LOOCV_alpha, 
                                               id_vars=['alpha'], 
                                               value_vars=list(range(0,114)), 
                                               ignore_index=False)
elasticnet_regression_LOOCV_alpha.rename(columns = {'variable':'case_index', 'value':'MSE'}, inplace = True)
display(elasticnet_regression_LOOCV_alpha)
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
      <th>alpha</th>
      <th>case_index</th>
      <th>MSE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0.0020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>0</td>
      <td>0.0020</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>0</td>
      <td>0.0020</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>0</td>
      <td>0.0022</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>0.0049</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>67</th>
      <td>4</td>
      <td>113</td>
      <td>0.2579</td>
    </tr>
    <tr>
      <th>68</th>
      <td>5</td>
      <td>113</td>
      <td>0.2614</td>
    </tr>
    <tr>
      <th>69</th>
      <td>6</td>
      <td>113</td>
      <td>0.2965</td>
    </tr>
    <tr>
      <th>70</th>
      <td>7</td>
      <td>113</td>
      <td>0.2146</td>
    </tr>
    <tr>
      <th>71</th>
      <td>8</td>
      <td>113</td>
      <td>0.2060</td>
    </tr>
  </tbody>
</table>
<p>8208 rows × 3 columns</p>
</div>



```python
##################################
# Renaming the alpha levels
# based on the proper category labels
##################################
elasticnet_regression_LOOCV_alpha_conditions = [
    (elasticnet_regression_LOOCV_alpha['alpha'] == 1),
    (elasticnet_regression_LOOCV_alpha['alpha'] == 2),
    (elasticnet_regression_LOOCV_alpha['alpha'] == 3),
    (elasticnet_regression_LOOCV_alpha['alpha'] == 4),
    (elasticnet_regression_LOOCV_alpha['alpha'] == 5),
    (elasticnet_regression_LOOCV_alpha['alpha'] == 6),
    (elasticnet_regression_LOOCV_alpha['alpha'] == 7),
    (elasticnet_regression_LOOCV_alpha['alpha'] == 8)]

elasticnet_regression_LOOCV_alpha_values = alphas_string_reversed
elasticnet_regression_LOOCV_alpha['alpha'] = np.select(elasticnet_regression_LOOCV_alpha_conditions,
                                                       elasticnet_regression_LOOCV_alpha_values,
                                                       default=str(np.nan))
elasticnet_regression_LOOCV_alpha['alpha'] = elasticnet_regression_LOOCV_alpha['alpha'].astype('category')
```


```python
##################################
# Plotting the elasticnet regression
# hyperparameter tuning results
# using LOOCV
##################################
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.grid(True)
sns.boxplot(x='alpha', 
            y='MSE', 
            data=elasticnet_regression_LOOCV_alpha, 
            color='#0070C0', 
            showmeans=True,
            meanprops={'marker':'o',
                       'markerfacecolor':'#ADD8E6', 
                       'markeredgecolor':'#FF0000',
                       'markersize':'8'})
plt.xlabel('Alpha Hyperparameter')
plt.ylabel('Mean Squared Error')
plt.title('Elastic Net Regression Hyperparameter Tuning Using LOOCV')
plt.show()
```


    
![png](output_213_0.png)
    



```python
##################################
# Evaluating the elastic-net regression model
# on the train set
##################################
elasticnet_y_hat_train = elasticnet_regression_pipeline.predict(X_train)

##################################
# Gathering the model evaluation metrics
##################################
elasticnet_performance_train = model_performance_evaluation(y_train, elasticnet_y_hat_train)
elasticnet_performance_train['model'] = ['elasticnet_regression'] * 3
elasticnet_performance_train['set'] = ['train'] * 3
print('Elastic Net Regression Model Performance on Train Data: ')
display(elasticnet_performance_train)
```

    Elastic Net Regression Model Performance on Train Data: 
    


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
      <th>metric_name</th>
      <th>metric_value</th>
      <th>model</th>
      <th>set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R2</td>
      <td>0.6298</td>
      <td>elasticnet_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSE</td>
      <td>0.3582</td>
      <td>elasticnet_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.4608</td>
      <td>elasticnet_regression</td>
      <td>train</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Evaluating the elastic-net regression model
# on the test set
##################################
elasticnet_y_hat_test = elasticnet_regression_pipeline.predict(X_test)

##################################
# Gathering the model evaluation metrics
##################################
elasticnet_performance_test = model_performance_evaluation(y_test, elasticnet_y_hat_test)
elasticnet_performance_test['model'] = ['elasticnet_regression'] * 3
elasticnet_performance_test['set'] = ['test'] * 3
print('Elastic Net Regression Model Performance on Test Data: ')
display(elasticnet_performance_test)
```

    Elastic Net Regression Model Performance on Test Data: 
    


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
      <th>metric_name</th>
      <th>metric_value</th>
      <th>model</th>
      <th>set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R2</td>
      <td>0.6409</td>
      <td>elasticnet_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSE</td>
      <td>0.3755</td>
      <td>elasticnet_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.4800</td>
      <td>elasticnet_regression</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Plotting the actual and predicted
# target variables
##################################
figure = plt.figure(figsize=(10,6))
axes = plt.axes()
plt.grid(True)
axes.plot(y_test, 
          elasticnet_y_hat_test, 
          marker='o', 
          ls='', 
          ms=3.0)
lim = (-2, 2)
axes.set(xlabel='Actual Cancer Rate', 
         ylabel='Predicted Cancer Rate', 
         xlim=lim,
         ylim=lim,
         title='Elastic Net Regression Model Prediction Performance');
```


    
![png](output_216_0.png)
    


## 1.7. Consolidated Findings <a class="anchor" id="1.7"></a>

1. The [linear regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) demonstrated the best independent test model performance among the candidate models. This model which did not apply any form of regularization proved to be more superior than the penalized models potentially due to the presence of a smaller subset of equally informative predictors - leading to a minimal over-confidence in the parameter estimates and in effect, minimal impact of the regularization constraints.
    * **R-Squared** = 0.6446
    * **Mean Squared Error** = 0.3716
    * **Mean Absolute Error** = 0.4773
3. Among the penalized models, the optimal [elastic net regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html) demonstrated the best independent test model performance with the tuned hyperparameter leaning towards a higher L1 regularization effect (optimal L1 ratio equals 0.9 which is reminiscent of lasso where L1 ratio equals 1.0).
    * **R-Squared** = 0.6409
    * **Mean Squared Error** = 0.3755
    * **Mean Absolute Error** = 0.4800
2. The optimal [lasso regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html) using an L1 regularization term demonstrated an equally high independent test model performance, confirming the earlier findings.
    * **R-Squared** = 0.6405
    * **Mean Squared Error** = 0.3760
    * **Mean Absolute Error** = 0.4805
3. The optimal [ridge regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) using an L2 regularization term equally demonstrated good independent test model performance.
    * **R-Squared** = 0.6352
    * **Mean Squared Error** = 0.3815
    * **Mean Absolute Error** = 0.4838
4. The [polynomial regression model](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) demonstrated the worst independent test model performance. In addition, metrics obtained from the apparent and independent test model performance are relatively different, indicative of the presence of model overfitting.
    * **R-Squared** = 0.6324
    * **Mean Squared Error** = 0.3844
    * **Mean Absolute Error** = 0.4867
5. The computed r-squared metrics for the formulated models were relatively low - only ranging from 0.63 to 0.65, which could be further improved by:
    * Considering more informative predictors
    * Considering more complex models other than linear regression and its variants


```python
##################################
# Consolidating all the
# model performance measures
##################################
performance_comparison = pd.concat([linear_performance_train, 
                                    linear_performance_test,
                                    polynomial_performance_train, 
                                    polynomial_performance_test,
                                    ridge_performance_train, 
                                    ridge_performance_test,
                                    lasso_performance_train, 
                                    lasso_performance_test,
                                    elasticnet_performance_train, 
                                    elasticnet_performance_test], 
                                   ignore_index=True)
print('Consolidated Model Performance on Train and Test Data: ')
display(performance_comparison)
```

    Consolidated Model Performance on Train and Test Data: 
    


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
      <th>metric_name</th>
      <th>metric_value</th>
      <th>model</th>
      <th>set</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>R2</td>
      <td>0.6332</td>
      <td>linear_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MSE</td>
      <td>0.3550</td>
      <td>linear_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MAE</td>
      <td>0.4609</td>
      <td>linear_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>3</th>
      <td>R2</td>
      <td>0.6446</td>
      <td>linear_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MSE</td>
      <td>0.3716</td>
      <td>linear_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>5</th>
      <td>MAE</td>
      <td>0.4773</td>
      <td>linear_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>6</th>
      <td>R2</td>
      <td>0.7908</td>
      <td>polynomial_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>7</th>
      <td>MSE</td>
      <td>0.2024</td>
      <td>polynomial_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>8</th>
      <td>MAE</td>
      <td>0.3503</td>
      <td>polynomial_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>9</th>
      <td>R2</td>
      <td>0.6324</td>
      <td>polynomial_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>10</th>
      <td>MSE</td>
      <td>0.3844</td>
      <td>polynomial_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>11</th>
      <td>MAE</td>
      <td>0.4867</td>
      <td>polynomial_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>12</th>
      <td>R2</td>
      <td>0.6220</td>
      <td>ridge_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>13</th>
      <td>MSE</td>
      <td>0.3659</td>
      <td>ridge_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>14</th>
      <td>MAE</td>
      <td>0.4680</td>
      <td>ridge_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>15</th>
      <td>R2</td>
      <td>0.6352</td>
      <td>ridge_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>16</th>
      <td>MSE</td>
      <td>0.3815</td>
      <td>ridge_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>17</th>
      <td>MAE</td>
      <td>0.4838</td>
      <td>ridge_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>18</th>
      <td>R2</td>
      <td>0.6295</td>
      <td>lasso_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>19</th>
      <td>MSE</td>
      <td>0.3586</td>
      <td>lasso_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>20</th>
      <td>MAE</td>
      <td>0.4608</td>
      <td>lasso_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>21</th>
      <td>R2</td>
      <td>0.6405</td>
      <td>lasso_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>22</th>
      <td>MSE</td>
      <td>0.3760</td>
      <td>lasso_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>23</th>
      <td>MAE</td>
      <td>0.4805</td>
      <td>lasso_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>24</th>
      <td>R2</td>
      <td>0.6298</td>
      <td>elasticnet_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>25</th>
      <td>MSE</td>
      <td>0.3582</td>
      <td>elasticnet_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>26</th>
      <td>MAE</td>
      <td>0.4608</td>
      <td>elasticnet_regression</td>
      <td>train</td>
    </tr>
    <tr>
      <th>27</th>
      <td>R2</td>
      <td>0.6409</td>
      <td>elasticnet_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>28</th>
      <td>MSE</td>
      <td>0.3755</td>
      <td>elasticnet_regression</td>
      <td>test</td>
    </tr>
    <tr>
      <th>29</th>
      <td>MAE</td>
      <td>0.4800</td>
      <td>elasticnet_regression</td>
      <td>test</td>
    </tr>
  </tbody>
</table>
</div>



```python
##################################
# Consolidating all the R2
# model performance measures
##################################
performance_comparison_R2 = performance_comparison[performance_comparison['metric_name']=='R2']
performance_comparison_R2_train = performance_comparison_R2[performance_comparison_R2['set']=='train'].loc[:,"metric_value"]
performance_comparison_R2_test = performance_comparison_R2[performance_comparison_R2['set']=='test'].loc[:,"metric_value"]
```


```python
##################################
# Plotting all the R2
# model performance measures
# between train and test sets
##################################
performance_comparison_R2_plot = pd.DataFrame({'train': performance_comparison_R2_train.values,
                                              'test': performance_comparison_R2_test.values},
                                              index=performance_comparison_R2['model'].unique())
performance_comparison_R2_plot = performance_comparison_R2_plot.plot.barh(figsize=(10, 6))
performance_comparison_R2_plot.set_xlim(0.00,1.00)
performance_comparison_R2_plot.set_title("Model Comparison by R-Squared Performance on Test Data")
performance_comparison_R2_plot.set_xlabel("R-Squared Performance")
performance_comparison_R2_plot.set_ylabel("Regression Model")
performance_comparison_R2_plot.grid(False)
performance_comparison_R2_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in performance_comparison_R2_plot.containers:
    performance_comparison_R2_plot.bar_label(container, fmt='%.5f', padding=-50)
```


    
![png](output_220_0.png)
    



```python
##################################
# Consolidating all the MSE
# model performance measures
##################################
performance_comparison_MSE = performance_comparison[performance_comparison['metric_name']=='MSE']
performance_comparison_MSE_train = performance_comparison_MSE[performance_comparison_MSE['set']=='train'].loc[:,"metric_value"]
performance_comparison_MSE_test = performance_comparison_MSE[performance_comparison_MSE['set']=='test'].loc[:,"metric_value"]
```


```python
##################################
# Plotting all the MSE
# model performance measures
# between train and test sets
##################################
performance_comparison_MSE_plot = pd.DataFrame({'train': performance_comparison_MSE_train.values,
                                                'test': performance_comparison_MSE_test.values},
                                               index=performance_comparison_MSE['model'].unique())
performance_comparison_MSE_plot = performance_comparison_MSE_plot.plot.barh(figsize=(10, 6))
performance_comparison_MSE_plot.set_xlim(0.00,1.00)
performance_comparison_MSE_plot.set_title("Model Comparison by Mean Squared Error Performance on Test Data")
performance_comparison_MSE_plot.set_xlabel("Mean Squared Error Performance")
performance_comparison_MSE_plot.set_ylabel("Regression Model")
performance_comparison_MSE_plot.grid(False)
performance_comparison_MSE_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in performance_comparison_MSE_plot.containers:
    performance_comparison_MSE_plot.bar_label(container, fmt='%.5f', padding=-50)
```


    
![png](output_222_0.png)
    



```python
##################################
# Consolidating all the MAE
# model performance measures
##################################
performance_comparison_MAE = performance_comparison[performance_comparison['metric_name']=='MAE']
performance_comparison_MAE_train = performance_comparison_MAE[performance_comparison_MAE['set']=='train'].loc[:,"metric_value"]
performance_comparison_MAE_test = performance_comparison_MAE[performance_comparison_MAE['set']=='test'].loc[:,"metric_value"]
```


```python
##################################
# Plotting all the MAE
# model performance measures
# between train and test sets
##################################
performance_comparison_MAE_plot = pd.DataFrame({'train': performance_comparison_MAE_train.values,
                                                'test': performance_comparison_MAE_test.values},
                                               index=performance_comparison_MAE['model'].unique())
performance_comparison_MAE_plot = performance_comparison_MAE_plot.plot.barh(figsize=(10, 6))
performance_comparison_MAE_plot.set_xlim(0.00,1.00)
performance_comparison_MAE_plot.set_title("Model Comparison by Mean Absolute Error Performance on Test Data")
performance_comparison_MAE_plot.set_xlabel("Mean Absolute Error Performance")
performance_comparison_MAE_plot.set_ylabel("Regression Model")
performance_comparison_MAE_plot.grid(False)
performance_comparison_MAE_plot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
for container in performance_comparison_MAE_plot.containers:
    performance_comparison_MAE_plot.bar_label(container, fmt='%.5f', padding=-50)
```


    
![png](output_224_0.png)
    


# 2. Summary <a class="anchor" id="Summary"></a>

![Project40_Summary.png](54982e31-4bac-4107-a6ab-1b976f719d5c.png)

# 3. References <a class="anchor" id="References"></a>
* **[Book]** [Data Preparation for Machine Learning: Data Cleaning, Feature Selection, and Data Transforms in Python](https://machinelearningmastery.com/data-preparation-for-machine-learning/) by Jason Brownlee
* **[Book]** [Feature Engineering and Selection: A Practical Approach for Predictive Models](http://www.feat.engineering/) by Max Kuhn and Kjell Johnson
* **[Book]** [Feature Engineering for Machine Learning](https://www.oreilly.com/library/view/feature-engineering-for/9781491953235/) by Alice Zheng and Amanda Casari
* **[Book]** [Applied Predictive Modeling](https://link.springer.com/book/10.1007/978-1-4614-6849-3?page=1) by Max Kuhn and Kjell Johnson
* **[Book]** [Data Mining: Practical Machine Learning Tools and Techniques](https://www.sciencedirect.com/book/9780123748560/data-mining-practical-machine-learning-tools-and-techniques?via=ihub=) by Ian Witten, Eibe Frank, Mark Hall and Christopher Pal 
* **[Book]** [Data Cleaning](https://dl.acm.org/doi/book/10.1145/3310205) by Ihab Ilyas and Xu Chu
* **[Book]** [Data Wrangling with Python](https://www.oreilly.com/library/view/data-wrangling-with/9781491948804/) by Jacqueline Kazil and Katharine Jarmul
* **[Python Library API]** [NumPy](https://numpy.org/doc/) by NumPy Team
* **[Python Library API]** [pandas](https://pandas.pydata.org/docs/) by Pandas Team
* **[Python Library API]** [seaborn](https://seaborn.pydata.org/) by Seaborn Team
* **[Python Library API]** [matplotlib.pyplot](https://matplotlib.org/3.5.3/api/_as_gen/matplotlib.pyplot.html) by MatPlotLib Team
* **[Python Library API]** [itertools](https://docs.python.org/3/library/itertools.html) by Python Team
* **[Python Library API]** [operator](https://docs.python.org/3/library/operator.html) by Python Team
* **[Python Library API]** [sklearn.experimental](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.experimental) by Scikit-Learn Team
* **[Python Library API]** [sklearn.impute](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.impute) by Scikit-Learn Team
* **[Python Library API]** [sklearn.linear_model](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model) by Scikit-Learn Team
* **[Python Library API]** [sklearn.preprocessing](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.preprocessing) by Scikit-Learn Team
* **[Python Library API]** [sklearn.metrics](https://scikit-learn.org/stable/modules/model_evaluation.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.model_selection](https://scikit-learn.org/stable/model_selection.html) by Scikit-Learn Team
* **[Python Library API]** [sklearn.pipeline](https://scikit-learn.org/stable/modules/compose.html) by Scikit-Learn Team
* **[Python Library API]** [scipy](https://docs.scipy.org/doc/scipy/) by SciPy Team
* **[Article]** [Step-by-Step Exploratory Data Analysis (EDA) using Python](https://www.analyticsvidhya.com/blog/2022/07/step-by-step-exploratory-data-analysis-eda-using-python/#:~:text=Exploratory%20Data%20Analysis%20(EDA)%20with,distributions%20using%20Python%20programming%20language.) by Malamahadevan Mahadevan (Analytics Vidhya)
* **[Article]** [Exploratory Data Analysis in Python — A Step-by-Step Process](https://towardsdatascience.com/exploratory-data-analysis-in-python-a-step-by-step-process-d0dfa6bf94ee) by Andrea D'Agostino (Towards Data Science)
* **[Article]** [Exploratory Data Analysis with Python](https://medium.com/@douglas.rochedo/exploratory-data-analysis-with-python-78b6c1d479cc) by Douglas Rocha (Medium)
* **[Article]** [4 Ways to Automate Exploratory Data Analysis (EDA) in Python](https://builtin.com/data-science/EDA-python) by Abdishakur Hassan (BuiltIn)
* **[Article]** [10 Things To Do When Conducting Your Exploratory Data Analysis (EDA)](https://www.analyticsvidhya.com) by Alifia Harmadi (Medium)
* **[Article]** [How to Handle Missing Data with Python](https://machinelearningmastery.com/handle-missing-data-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Statistical Imputation for Missing Values in Machine Learning](https://machinelearningmastery.com/statistical-imputation-for-missing-values-in-machine-learning/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Imputing Missing Data with Simple and Advanced Techniques](https://towardsdatascience.com/imputing-missing-data-with-simple-and-advanced-techniques-f5c7b157fb87) by Idil Ismiguzel (Towards Data Science)
* **[Article]** [Missing Data Imputation Approaches | How to handle missing values in Python](https://www.machinelearningplus.com/machine-learning/missing-data-imputation-how-to-handle-missing-values-in-python/) by Selva Prabhakaran (Machine Learning +)
* **[Article]** [Master The Skills Of Missing Data Imputation Techniques In Python(2022) And Be Successful](https://medium.com/analytics-vidhya/a-quick-guide-on-missing-data-imputation-techniques-in-python-2020-5410f3df1c1e) by Mrinal Walia (Analytics Vidhya)
* **[Article]** [How to Preprocess Data in Python](https://builtin.com/machine-learning/how-to-preprocess-data-python) by Afroz Chakure (BuiltIn)
* **[Article]** [Easy Guide To Data Preprocessing In Python](https://www.kdnuggets.com/2020/07/easy-guide-data-preprocessing-python.html) by Ahmad Anis (KDNuggets)
* **[Article]** [Data Preprocessing in Python](https://towardsdatascience.com/data-preprocessing-in-python-b52b652e37d5) by Tarun Gupta (Towards Data Science)
* **[Article]** [Data Preprocessing using Python](https://medium.com/@suneet.bhopal/data-preprocessing-using-python-1bfee9268fb3) by Suneet Jain (Medium)
* **[Article]** [Data Preprocessing in Python](https://medium.com/@abonia/data-preprocessing-in-python-1f90d95d44f4) by Abonia Sojasingarayar (Medium)
* **[Article]** [Data Preprocessing in Python](https://medium.datadriveninvestor.com/data-preprocessing-3cd01eefd438) by Afroz Chakure (Medium)
* **[Article]** [Detecting and Treating Outliers | Treating the Odd One Out!](https://www.analyticsvidhya.com/blog/2021/05/detecting-and-treating-outliers-treating-the-odd-one-out/) by Harika Bonthu (Analytics Vidhya)
* **[Article]** [Outlier Treatment with Python](https://medium.com/analytics-vidhya/outlier-treatment-9bbe87384d02) by Sangita Yemulwar (Analytics Vidhya)
* **[Article]** [A Guide to Outlier Detection in Python](https://builtin.com/data-science/outlier-detection-python) by Sadrach Pierre (BuiltIn)
* **[Article]** [How To Find Outliers in Data Using Python (and How To Handle Them)](https://careerfoundry.com/en/blog/data-analytics/how-to-find-outliers/) by Eric Kleppen (Career Foundry)
* **[Article]** [Statistics in Python — Collinearity and Multicollinearity](https://towardsdatascience.com/statistics-in-python-collinearity-and-multicollinearity-4cc4dcd82b3f) by Wei-Meng Lee (Towards Data Science)
* **[Article]** [Understanding Multicollinearity and How to Detect it in Python](https://towardsdatascience.com/everything-you-need-to-know-about-multicollinearity-2f21f082d6dc) by Terence Shin (Towards Data Science)
* **[Article]** [A Python Library to Remove Collinearity](https://www.yourdatateacher.com/2021/06/28/a-python-library-to-remove-collinearity/) by Gianluca Malato (Your Data Teacher)
* **[Article]** [8 Best Data Transformation in Pandas](https://ai.plainenglish.io/data-transformation-in-pandas-29b2b3c61b34) by Tirendaz AI (Medium)
* **[Article]** [Data Transformation Techniques with Python: Elevate Your Data Game!](https://medium.com/@siddharthverma.er.cse/data-transformation-techniques-with-python-elevate-your-data-game-21fcc7442cc2) by Siddharth Verma (Medium)
* **[Article]** [Data Scaling with Python](https://www.kdnuggets.com/2023/07/data-scaling-python.html) by Benjamin Obi Tayo (KDNuggets)
* **[Article]** [How to Use StandardScaler and MinMaxScaler Transforms in Python](https://machinelearningmastery.com/standardscaler-and-minmaxscaler-transforms-in-python/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Feature Engineering: Scaling, Normalization, and Standardization](https://www.analyticsvidhya.com/blog/2020/04/feature-scaling-machine-learning-normalization-standardization/) by Aniruddha Bhandari  (Analytics Vidhya)
* **[Article]** [How to Normalize Data Using scikit-learn in Python](https://www.digitalocean.com/community/tutorials/normalize-data-in-python) by Jayant Verma (Digital Ocean)
* **[Article]** [What are Categorical Data Encoding Methods | Binary Encoding](https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/) by Shipra Saxena  (Analytics Vidhya)
* **[Article]** [Guide to Encoding Categorical Values in Python](https://pbpython.com/categorical-encoding.html) by Chris Moffitt (Practical Business Python)
* **[Article]** [Categorical Data Encoding Techniques in Python: A Complete Guide](https://soumenatta.medium.com/categorical-data-encoding-techniques-in-python-a-complete-guide-a913aae19a22) by Soumen Atta (Medium)
* **[Article]** [Categorical Feature Encoding Techniques](https://towardsdatascience.com/categorical-encoding-techniques-93ebd18e1f24) by Tara Boyle (Medium)
* **[Article]** [Ordinal and One-Hot Encodings for Categorical Data](https://machinelearningmastery.com/one-hot-encoding-for-categorical-data/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [Hypothesis Testing with Python: Step by Step Hands-On Tutorial with Practical Examples](https://towardsdatascience.com/hypothesis-testing-with-python-step-by-step-hands-on-tutorial-with-practical-examples-e805975ea96e) by Ece Işık Polat (Towards Data Science)
* **[Article]** [17 Statistical Hypothesis Tests in Python (Cheat Sheet)](https://machinelearningmastery.com/statistical-hypothesis-tests-in-python-cheat-sheet/) by Jason Brownlee (Machine Learning Mastery)
* **[Article]** [A Step-by-Step Guide to Hypothesis Testing in Python using Scipy](https://medium.com/@gabriel_renno/a-step-by-step-guide-to-hypothesis-testing-in-python-using-scipy-8eb5b696ab07) by Gabriel Rennó (Medium)
* **[Publication]** [Ridge Regression: Biased Estimation for Nonorthogonal Problems](https://www.tandfonline.com/doi/abs/10.1080/00401706.1970.10488634) by Arthur Hoerl and Robert Kennard (Technometrics)
* **[Publication]** [Regression Shrinkage and Selection Via the Lasso](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1996.tb02080.x) by Rob Tibshirani (Journal of the Royal Statistical Society)
* **[Publication]** [ Regularization and Variable Selection via the Elastic Net](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.1467-9868.2005.00503.x) by Hui Zou and Trevor Hastie (Journal of the Royal Statistical Society)

***


```python
from IPython.display import display, HTML
display(HTML("<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>"))
```


<style>.rendered_html { font-size: 15px; font-family: 'Trebuchet MS'; }</style>

