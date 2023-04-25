# Ex-06-Feature-Transformation

# AIM

To read the given data and perform Feature Transformation process and save the data to a file.

# EXPLANATION

Feature Transformation is a technique by which we can boost our model performance. Feature transformation is a mathematical transformation in which we apply a mathematical formula to a particular column(feature) and transform the values which are useful for our further analysis.

# ALGORITHM

## STEP 1

Read the given Data

## STEP 2

Clean the Data Set using Data Cleaning Process

## STEP 3

Apply Feature Transformation techniques to all the features of the data set

## STEP 4

Save the data to the file

# CODE

```py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

df = pd.read_csv("/content/Data_to_Transform.csv")
df

df.isnull().sum()

df.describe()

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Highly Negative Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

sm.qqplot(df['Moderate Negative Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = np.log(df['Highly Positive Skew'])

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew'] = np.log(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = 1/df['Highly Positive Skew']

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Highly Positive Skew'] = df['Highly Positive Skew']**(1/1.2)

sm.qqplot(df['Highly Positive Skew'],fit=True,line='45')
plt.show()

df['Moderate Positive Skew_1'], parameters=stats.yeojohnson(df['Moderate Positive Skew'])

sm.qqplot(df['Moderate Positive Skew_1'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import PowerTransformer
transformer=PowerTransformer("yeo-johnson")
df['ModerateNegativeSkew_2']=pd.DataFrame(transformer.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution = 'normal')
df['ModerateNegativeSkew_2'] = pd.DataFrame(qt.fit_transform(df[['Moderate Negative Skew']]))
sm.qqplot(df['ModerateNegativeSkew_2'],fit=True,line='45')
plt.show()

```

# OUTPUT

## DATASET

![output](/1.png)

## ISNULL

![output](/2.png)

## INFO

![output](/3.png)

## DESCRIBE

![output](/4.png)

## HIGHLY POSITIVE SKEW

![output](/5.png)

## HIGHLY NEGATIVE SKEW

![output](/6.png)

## MODERATE POSITIVE SKEW

![output](/7.png)

## MODERATE NEGATIVE SKEW

![output](/8.png)

## LOG OF MODERATE POSITIVE SKEW:

![output](/9.png)

## LOG OF HIGHLY POSITIVE SKEW

![output](/10.png)

## RECIPROCAL OF HIGHLY POSITIVE SKEW

![output](/11.png)

## SQUARE ROOT TRANSFORMATION

![output](/12.png)

## POWER TRANSFORMATION OF MODERATE NEGATIVE SKEW

![output](/13.png)

## QUANTILE TRANSFORMATION

![output](/14.png)

# RESULT:
Thus, Feature transformation is performed and executed successfully for the given dataset


