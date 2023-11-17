#!/usr/bin/env python
# coding: utf-8

# # About ConnectTel
# 
# ConnectTel is a leading telecommunications company at the forefront of innovation and connectivity solutions. With a strong presence in the global market, ConnectTel has established itself as a trusted provider of reliable voice, data, and Internet services. Offering a comprehensive range of telecommunications solutions, including mobile networks, broadband connections, and enterprise solutions, ConnectTel caters to both individual and corporate customers, they are committed to providing exceptional customer service and cutting-edge technology. ConnectTel ensures seamless communication experiences for millions of users worldwide. Through strategic partnerships and a customer-centric approach, ConnectTel continues to revolutionize the telecom industry, empowering individuals and businesses to stay connected and thrive in the digital age.

# # Problem Overview
# ConnectTel Telecom Company faces the pressing need to address
# customer churn, which poses a significant threat to its business
# sustainability and growth.
# The company's current customer retention strategies lack precision and
# effectiveness, resulting in the loss of valuable customers to competitors.
# To overcome this challenge, ConnectTel aims to develop a
# robust customer churn prediction system for which I was contacted
# to handle as a Data Scientist. By leveraging advanced analytics and machine
# learning techniques on available customer data, the company seeks to
# accurately forecast customer churn and implement targeted retention
# initiatives.
# This proactive approach will enable ConnectTel to reduce customer
# attrition, enhance customer loyalty, and maintain a competitive edge in the
# highly dynamic and competitive telecommunications industry.

# In[1]:


# Data Analysis
import pandas as pd
import numpy as np

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Data Pre-Processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Classifier Libraries
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# !pip install xgboost
get_ipython().system('pip install xgboost')
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# Evaluation metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix


# In[2]:


df = pd.read_csv(r"C:\Users\Mumsie\Downloads\Customer-Churn.csv")
df.head()


# ## Features in the dataset and meaning:
# 
# 1. CustomerID: A unique identifier assigned to each telecom customer, enabling tracking and identification of individual customers.
# 2. Gender: The gender of the customer, which can be categorized as male, or female. This information helps in analyzing gender-based trends in customer churn.
# 3. SeniorCitizen: A binary indicator that identifies whether the customer is a senior citizen or not. This attribute helps in understanding if there are any specific churn patterns among senior customers.
# 4. Partner: Indicates whether the customer has a partner or not. This attribute helps in evaluating the impact of having a partner on churn behavior.
# 5. Dependents: Indicates whether the customer has dependents or not. This attribute helps in assessing the influence of having dependents on customer churn.
# 6. Tenure: The duration for which the customer has been subscribed to the telecom service. It represents the loyalty or longevity of the customer’s relationship with the company and is a significant predictor of churn.
# 7. PhoneService: Indicates whether the customer has a phone service or not. This attribute helps in understanding the impact of phone service on churn.
# 8. MultipleLines: Indicates whether the customer has multiple lines or not. This attribute helps in analyzing the effect of having multiple lines on customer churn.
# 9. InternetService: Indicates the type of internet service subscribed by the customer, such as DSL, fiber optic, or no internet service. It helps in evaluating the relationship between internet service and churn.
# 10. OnlineSecurity: Indicates whether the customer has online security services or not. This attribute helps in analyzing the impact of online security on customer churn.
# 11. OnlineBackup: Indicates whether the customer has online backup services or not. This attribute helps in evaluating the impact of online backup on churn behavior.
# 12. DeviceProtection: Indicates whether the customer has device protection services or not. This attribute helps in understanding the influence of device protection on churn.
# 13. TechSupport: Indicates whether the customer has technical support services or not. This attribute helps in assessing the impact of tech support on churn behavior.
# 14. StreamingTV: Indicates whether the customer has streaming TV services or not. This attribute helps in evaluating the impact of streaming TV on customer churn.
# 15. StreamingMovies: Indicates whether the customer has streaming movie services or not. This attribute helps in understanding the influence  of streaming movies on churn behavior.
# 16. Contract: Indicates the type of contract the customer has, such as a month-to-month, one-year, or two-year contract. It is a crucial factor in predicting churn as different contract lengths may have varying impacts on customer loyalty.
# 17. PaperlessBilling: Indicates whether the customer has opted for paperless billing or not. This attribute helps in analyzing the effect of  paperless billing on customer churn.
# 18. PaymentMethod: Indicates the method of payment used by the customer, such as electronic checks, mailed checks, bank transfers, or credit cards. This attribute helps in evaluating the impact of payment methods on churn.
# 19. MonthlyCharges: The amount charged to the customer on a monthly basis. It helps in understanding the relationship between monthly charges and churn behavior.
# 20. TotalCharges: The total amount charged to the customer over the entire tenure. It represents the cumulative revenue generated from the customer and may have an impact on churn.
# 21. Churn: The target variable indicates whether the customer has churned (canceled the service) or not. It is the main variable to predict in telecom customer churn analysis.

# In[3]:


# I will transpose the dataframe in order to view the complete features (i.e. rows become columns)

df.head().T


# In[4]:


# Cleaning the column names (changing column names to lower case and triming white spaces)

df.columns = df.columns.str.lower().str.replace(' ', '_')
df.head(3)


# In[5]:


categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


# In[6]:


df.head().T


# Next, I will inspect/carry out data verification to ascertain Data type, number of features and rows, missing data, etc

# In[7]:


df.dtypes


# In[8]:


df.info()


# Next, I will convert the feature 'totalcharges' to a number data type

# In[9]:


pd.to_numeric(df.totalcharges, errors = 'coerce')


# In[10]:


tc = pd.to_numeric(df.totalcharges, errors = 'coerce')
tc.isnull().sum()


# 11 values are missing so I will investigate what is going on within the dataframe with reference tio the missing values. I will investigate the features 'customerid' and 'totalcharges' for more detail

# In[11]:


df[tc.isnull()][['customerid', 'totalcharges']]


# There are missing values in 'totalcharges', so I will first convert the feature to a numeric data type, and replace these values with 0

# In[12]:


df.totalcharges = pd.to_numeric(df.totalcharges, errors = 'coerce')


# In[13]:


df.totalcharges = df.totalcharges.fillna(0)


# In[14]:


#Statistical Analysis of the data

df.describe()


# In[15]:


#Next, I will check for missing values, and visualize the missing data

print(df.isna().sum())

plt.figure(figsize = (10,3))
sns.heatmap(df.isna(), cbar=True, cmap='Blues_r');


# In[16]:


df.columns


# I will check for outliers in certain columns

# In[17]:


sns.boxplot(x=df['tenure']);


# In[18]:


sns.boxplot(x=df['monthlycharges']);


# In[19]:


sns.boxplot(x=df['totalcharges']);


# > No outliers have been found in the investigated features of the dataset

# Next, I will investigate certain features beginning with the senior citizen feature

# In[20]:


df.columns


# In[21]:


def age_group(category):
    if category == 1:
        return 'senior citizen'
    else:
        return 'adult'
    
df['age_group'] = df['seniorcitizen'].apply(age_group)
df.head(2)


# In[22]:


plt.figure(figsize = (5,5))
sns.countplot(x = df['age_group'])
plt.xlabel('Age Group')
plt.ylabel('Frequency')
plt.title('Total Number of Customers by Age Group');


# It can be seen that there are fewer senior citizens in the dataset

# Next, I will investigate and visualize the tenure column

# In[23]:


df['tenure'].min()


# In[24]:


df['tenure'].max()


# In[25]:


def tenure_category(months):
    if months <= 12:
        return '12 months'
    elif months <= 36:
        return '13-36 months'
    else:
        return '>36 months'
    
df['tenure_category'] = df['tenure'].apply(tenure_category)
df.head(2)


# In[26]:


plt.figure(figsize = (8,5))
sns.countplot(x = df['tenure_category'])
plt.xlabel('Tenure Category in Months')
plt.ylabel('Frequency')
plt.title('Total Number of Customers by Tenure');


# Majority of the customers have their tenures longer than 3 years, followed by customers whose tenure is less than 1 year.

# Next, I will investigate the monthly charges

# In[27]:


df['monthlycharges'].min()


# In[28]:


df['monthlycharges'].max()


# In[29]:


def monthlycharge_group(charge):
    if charge <= 51.75:
        return 'Low'
    elif charge <= 85.25:
        return 'Moderate'
    else:
        return 'High'
    
df['monthlycharge_group'] = df['monthlycharges'].apply(monthlycharge_group)
df.head(2)


# In[30]:


print(df['monthlycharge_group'].value_counts())

plt.figure(figsize = (8,5))
sns.countplot(x = df['monthlycharge_group'])
plt.xlabel('Monthly Charge Category')
plt.ylabel('Frequency')
plt.title('Total Number of Customers by Monthly Charges');


# **Low:** There are `2,451` customers in the 'Low' monthly charge category. \
# **Moderate:** There are `2,439` customers in the 'Moderate' monthly charge category.\
# **High:** There are `2,153` customers in the 'High' monthly charge category.
# 
# **Low:** The 'Low' monthly charge category, comprising 2,451 customers, suggests a significant portion of the customer base with lower monthly charges. Customers in this category may have opted for basic plans or services, resulting in lower monthly expenses. While this group contributes to the customer base, their lower monthly charges might indicate a potential risk of lower individual revenue.
# 
# **Moderate:** The 'Moderate' monthly charge category, with 2,439 customers, represents a middle-ground in terms of monthly charges. Customers in this category likely have a moderate level of service usage or have chosen plans with intermediate pricing. This group contributes moderately to the overall monthly revenue and might represent a balance between service usage and affordability.
# 
# **High:** The 'High' monthly charge category, consisting of 2,153 customers, signifies those with higher monthly charges. Customers in this category may have subscribed to premium plans, additional services, or have consistently higher service usage on a monthly basis. While this group is smaller in size, they are crucial for generating higher individual monthly revenue.

# Next, I will investigate the total charges

# In[31]:


df['totalcharges'].min()


# In[32]:


df['totalcharges'].max()


# In[33]:


def totalcharge_group(charge):
    if charge <= 2894:
        return 'Low'
    elif charge <= 5788:
        return 'Moderate'
    else:
        return 'High'
    
df['totalcharge_group'] = df['totalcharges'].apply(totalcharge_group)
df.head(2)


# In[34]:


print(df['totalcharge_group'].value_counts())

plt.figure(figsize = (8,5))
sns.countplot(x = df['totalcharge_group'])
plt.xlabel('Total Charge Category')
plt.ylabel('Frequency')
plt.title('Total Number of Customers by Total Charges');


# **Low:** The 'Low' total charge category, with 4,781 customers, suggests a substantial portion of the customer base with lower cumulative charges throughout their tenure. These customers may have opted for basic plans or have lower overall service usage, contributing to their lower total charges. While this group represents a significant customer base, their lower total charges might also indicate a potential risk of lower revenue generation for the company.
# 
# **Moderate:** The 'Moderate' total charge category, encompassing 1,475 customers, signifies a middle-ground in terms of cumulative charges. Customers in this category likely have a moderate level of service usage or have chosen plans with intermediate pricing. This group contributes moderately to the overall revenue and might represent a balance between service usage and affordability.
# 
# **High:** The 'High' total charge category, comprising 787 customers, represents those with higher cumulative charges. These customers may have subscribed to premium plans, additional services, or have consistently higher service usage over their tenure. While this group is smaller in size, they are crucial for revenue generation, as their higher total charges contribute significantly to the overall financial health of the company.

# In[35]:


df.head()


# I will investigate the age group of customers by the churn feature

# In[36]:


print(df[['age_group', 'churn']].value_counts())

plt.figure(figsize = (8,5))
sns.countplot(x = df['age_group'], hue = df['churn'])
plt.xlabel('Age Group')
plt.ylabel('Frequency')
plt.title('Customer Churn by Age Group');


# Adults:\
# **No Churn (no):** 4508 customers \
# **Churn (yes):** 1393 customers \
# The majority of adults did not churn, as the count for 'no churn' is higher.
# 
# Senior Citizens:\
# **No Churn (no):** 666 customers \
# **Churn (yes):** 476 customers \
# The majority of senior citizens also did not churn, as the count for 'no churn' is higher.
# 
# In both age groups, the 'no churn' category has a higher count, indicating that, for the given dataset, more customers are not churning in both the adult and senior citizen age groups.

# Next, I will investigate the gender of customers by the churn feature

# In[37]:


print(df[['gender', 'churn']].value_counts())

plt.figure(figsize = (8,5))
sns.countplot(x = df['gender'], hue = df['churn'])
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Customer Churn by Gender');


# Females: \
# **No Churn (no):** 2549 customers \
# **Churn (yes):** 939 customers 
# 
# Males: \
# **No Churn (no):** 2625 customers \
# **Churn (yes):** 930 customers
# 
# The majority of both males and females did not churn, as the count for 'no churn' is higher in both gender categories. However, it's interesting to note that the count of churned females ('yes') is higher than the count of churned males ('yes'). This suggests that, for this dataset, there are more instances of churn among female customers.

# Next, I will investigate the presence/absence of dependents by the churn feature

# In[38]:


print(df[['dependents', 'churn']].value_counts())

plt.figure(figsize = (8,5))
sns.countplot(x = df['dependents'], hue = df['churn'])
plt.xlabel('Dependents')
plt.ylabel('Frequency')
plt.title('Customer Churn by Dependents');


# Customers without Dependents: \
# **No Churn (no):** 3390 customers \
# **Churn (yes):** 1543 customers
# 
# Customers with Dependents: \
# **No Churn (no):** 1784 customers \
# **Churn (yes):** 326 customers
# 
# The majority of customers both with and without dependents did not churn, as the count for 'no churn' is higher in both categories. However, it appears that customers without dependents have a higher count of churn ('yes') compared to those with dependents. This suggests that, for the given dataset, customers without dependents are more likely to churn.

# Next, I will investigate customer contract by the churn feature

# In[39]:


print(df[['contract', 'churn']].value_counts())

plt.figure(figsize = (8,5))
sns.countplot(x = df['contract'], hue = df['churn'])
plt.xlabel('Customer contract')
plt.ylabel('Frequency')
plt.title('Customer Churn by contract type');


# Month-to-Month Contract: \
# **No Churn (no):** 2220 customers \
# **Churn (yes):** 1655 customers
# 
# One-Year Contract: \
# **No Churn (no):** 1307 customers \
# **Churn (yes):** 166 customers
# 
# Two-Year Contract: \
# **No Churn (no):** 1647 customers \
# **Churn (yes):** 48 customers
# 
# The majority of customers with a two-year contract did not churn ('no'), followed by those with a month-to-month contract. However, customers with a one-year contract have a higher churn count compared to those with a two-year contract.
# 
# This information provides insights into how the contract length may influence customer churn, with customers on shorter-term contracts (month-to-month) having a higher likelihood of churning compared to those on longer-term contracts.

# In[40]:


df.columns


# In[41]:


df = df.drop(['age_group', 'tenure_category', 'monthlycharge_group', 'totalcharge_group'], axis=1)
df.head(2)


# Next, I will investigate the churn feature

# In[42]:


df.churn


# In ML, we care more about number representation (i.e. 0 or 1) where: 1 = churn (positive example), 0 = not churn (negative example).
# 
# I want to get the values that are 'yes' (I will pick just the first five)

# In[43]:


(df.churn =='yes').head()


# If a value is yes, then it is 'true'. Conversely, if a value is no, it gets 'false'. Now I will replace these boolean values wit numbers (0 or 1)

# In[44]:


(df.churn == 'yes').astype(int).head()


# yes = 1 (customers churn) \
# no = 0 (customers did not churn)

# In[45]:


df.churn = (df.churn == 'yes').astype(int)


# ### Next, I will set up the validation framework
# I will perform the train/validation/test split with Scikit-Learn \
# (Note: for this model, I will divide my dataset into 3: training set, validation set, and test set)

# In[46]:


from sklearn.model_selection import train_test_split


# In[47]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)


# Note that test size = 20%, and random state is to ensure the results are reproducible (i.e. same results can be gotten at any point in time)
# 
# Next, I will check the length/size of each

# In[48]:


len(df_full_train) , len(df_test)


# Next, I will split the 'full_train' (80% of the original dataset) again into 'train' and 'validation sets'. \
# Note that, to get the 'validation' set to be as big as the 20% that df_test is, I will need to get 25% of 'full_train' (which is equivalent to 20% of the original dataset)

# In[49]:


df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


# In[50]:


len(df_train), len(df_val), len(df_test)


# In[51]:


df_train.head()


# The indices are seen to be shuffled and I will like to reorder them (personal preference)

# In[52]:


df_train.reset_index().head(3)


# Next, I will drop the previous index

# In[53]:


df_train.reset_index(drop=True).head(3)


# Next, I will repeat the process for the validation and test sets (and reassign all sets back to their respective variables)

# In[54]:


df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# Next, I will get my 'y' variables... I write the churn variables in to the sets, and then delete the (initial) churn from my database

# In[55]:


y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values


# In[56]:


del df_train['churn']
del df_val['churn']
del df_test['churn']


# Notice that I did not delete the churn variable from df_full_train, and that's simply because I want to explore the target variable (churn) next.

# ### Exploratory Data Analysis (cont'd)
# Next, I will:
# - Check missing values
# - Look at the target variable (churn)
# - Look at the numerical and categorical variables

# I will investigate these using the full_train dataset. \
# First, i will reassign a new ordered index, and drop the reshuffled index (again, personal preference)..

# In[57]:


df_full_train = df_full_train.reset_index(drop=True)
df_full_train.head()


# Next, I will look at missing values..

# In[58]:


df_full_train.isnull().sum()


# No missing values. So, next, I will look at the target variable (churn)

# In[59]:


df_full_train.churn


# Next, I will look at the distribution within the churn feature (i.e. how many users churn versus how many do not churn)

# In[60]:


df_full_train.churn.value_counts()


# The number of churned users is approximately 3 times less, but I will confirm the percentage using the normalize=True keyword.   
# It divides the number by the total counts of elements in the series.

# In[61]:


df_full_train.churn.value_counts(normalize=True)


# The churn rate (rate at which users churn) is approx 27%
# 
# We can also compute this (particular (Read: binary dataset)) churn rate using the mean.   
# N/B: The mean can also give the churn rate because it is the sum of all 1s (values are 0 or 1) divided by n (the total count), i.e. the fraction of 1s in a binary dataset.

# In[62]:


global_churn_rate = df_full_train.churn.mean()
round(global_churn_rate, 2)


# This can be interpreted as 27% of users are churning
# 
# Next, I will investigate the categorical variables, and the numerical variables

# In[63]:


df_full_train.dtypes


# I can see that some data types are not correctly assigned, for example, seniorcitizen is an integer whereas it is supposed to be a categorical variable.
# However, my variables of interest are tenure, monthlycharges, and totalcharges.
# 
# I will create separate variable names to contain the numerical variables and the categorical variables

# In[64]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']


# In[65]:


categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
       'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod',]


# Investigating the number of unique variables in each categorical feature

# In[66]:


df_full_train[categorical].nunique()


# About six of the variables are binary, about eight have 3 values and one has 4 values

# ### Feature importance: Churn Rate and Risk Ratio
# Feature importance analysis (which is still a part of EDA) involves identifying which features affect the target variable.   
# I will investigate:
# 
# - Churn Rate (within each group)
# - Risk Ratio
# - Mutual Information
# 

# #### Churn Rate (within each group)

# In[67]:


df_full_train.head(3)


# Investigating gender
# 
# - I will look at the subset of females & then males
# - get their mean churn (churn rate)

# In[68]:


churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
churn_female


# In[69]:


churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()
churn_male


# In[70]:


global_churn = df_full_train.churn.mean()
global_churn


# The churn rate for males and females are not too different from the global churn.
# 
# 

# Investigating churn within the 'partner' feature..

# In[71]:


df_full_train.partner.value_counts()


# Some customers live with their partners (yes=2702)    
# and some live without partners (no=2932)

# In[72]:


churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
churn_partner


# In[73]:


global_churn - churn_partner


# The churn rate is approx 21% which looks like it is significantly less (6%) than the global churn rate

# In[74]:


churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
churn_no_partner


# In[75]:


global_churn - churn_no_partner


# The churn rate of those who live without partners is approx 6% more than global churn rate
# 
# This gives some idea that perhaps the partner variable is more important (relative to the gender variable) for predicting churn.   
# So, I will measure the feature importance next..

# #### Risk Ratio
# Note that if the churn rate is positive (difference greater than zero), it means that the global churn is higher than the group churn and that indicates customers who are less likely to churn. Conversely, if the churn rate is negative (difference less than zero), it means the global churn is lower than the group churn, indicating customers who are more likely to churn (this gives information in absolute terms).
# 
# Note that, we can also divide one by another, instead of just difference (divide the group churn rate by the global churn rate) (this gives information in relative terms).

# In[76]:


churn_no_partner/global_churn


# In[77]:


churn_partner/global_churn


# Customers without a partner have a churn rate of 22% higher than global churn, while for customers with a partner, the churn rate is 24% lower than global churn.
# 
# 

# Next, I will investigate the difference and risk ratio of churn rates per group relative to the global churn to determine importance

# In[78]:


df_group = df_full_train.groupby('gender').churn.agg(['mean', 'count'])
df_group


# Next, I will create columns for the differences, and risk ratios in churn rates between global churn and group churn

# In[79]:


df_group['diff'] = df_group['mean'] - global_churn 
df_group['risk'] = df_group['mean'] / global_churn 
df_group


# Next, I will repeat the two previous steps for all the categorical variables (which I created and assigned to the variable name "categorical") by creating a function
# 
# 

# Note, the last statement (df_group) to display my code is within the loop of my function so we won't see it. In order to display my function/code, I will import a special function from a library in IPython for displaying things (PS: Jupyter used to be ipython, hence the name)

# In[80]:


from IPython.display import display


# In[81]:


for c in categorical:
    print(c)
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn 
    df_group['risk'] = df_group['mean'] / global_churn 
    display(df_group)
    print()
    print()


# These statistics provide valuable insights into churn rates among different customer groups:
# 
# **Gender:**   
# Females have a slightly higher churn rate (27.68%) than the global churn (26%).
# Males have a slightly lower churn rate (26.32%) than the global churn.
# 
# **Senior Citizen:**   
# Senior citizens have a significantly higher churn rate (41.34%) compared to the global churn.
# Non-senior citizens have a lower churn rate (24.23%) compared to the global churn.
# 
# **Partner:**   
# Customers without a partner have a higher churn rate (32.98%) compared to the global churn.
# Customers with a partner have a lower churn rate (20.50%) compared to the global churn.
# 
# **Dependents:**   
# Customers without dependents have a higher churn rate (31.38%) compared to the global churn.
# Customers with dependents have a lower churn rate (16.57%) compared to the global churn.
# 
# **Phone Service:**   
# Customers with phone service have a slightly higher churn rate (27.30%) compared to the global churn.
# Customers without phone service have a lower churn rate (24.13%) compared to the global churn.
# 
# **Multiple Lines:**   
# Customers with multiple lines have a slightly higher churn rate (29.07%) compared to the global churn.
# Customers with no phone service have a lower churn rate (24.13%) compared to the global churn.
# 
# **Internet Service:**   
# Fiber optic internet service users have a significantly higher churn rate (42.52%) compared to the global churn.
# DSL internet service users have a lower churn rate (19.23%) compared to the global churn.
# 
# **Online Security:**   
# Customers without online security have a significantly higher churn rate (42.09%) compared to the global churn.
# Customers with online security have a lower churn rate (15.32%) compared to the global churn.
# 
# **Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies:**   
# Similar trends can be observed in these categories, where having the service leads to lower churn rates compared to not having it.
# 
# **Contract:**   
# Month-to-month contract customers have a significantly higher churn rate (43.17%) compared to the global churn.
# One-year and two-year contract customers have lower churn rates (12.06% and 2.83%, respectively) compared to the global churn.
# 
# **Paperless Billing:**   
# Customers with paperless billing have a slightly higher churn rate (33.82%) compared to the global churn.
# Customers without paperless billing have a lower churn rate (17.21%) compared to the global churn.
# 
# **Payment Method:**   
# Customers using electronic checks have a significantly higher churn rate (45.59%) compared to the global churn.
# Customers using other payment methods have lower churn rates.
# 
# > These insights provide valuable information for targeted customer retention strategies. For instance, senior citizens, customers with partners, those with long-term contracts, and customers with certain additional services (like online security) are less likely to churn, while customers with month-to-month contracts and certain internet services (fiber optic) are more likely to churn.

# In terms of importance, I will apply a metric which generates a number that describes the importance of the variable overall (for example, if contract is less or more important than gender, etc.) next.

# ### Feature Importance: Mutual Information
# [Mutual information](https://en.wikipedia.org/wiki/Mutual_information) is a concept from information theory which tells us how much we can learn about one variable if we know the value of another

# In[82]:


from sklearn.metrics import mutual_info_score


# In[83]:


mutual_info_score(df_full_train.churn, df_full_train.contract)


# This, for example, tells how much we learn about churn by observing the value of a contract variable, and likewise how much we know about the contract variable by observing churn
# 
# 

# Next, I will apply this metric to all the categorical variables in this dataset to see which of them has the highest (or least) mutual information score.   
# 
# **Note that** the mutual_info_score takes in two arguments (and I want a metric that takes in one argument) so I will wrap it in a function with one argument...

# In[84]:


def mutual_info_churn_score(series):
    return mutual_info_score(series, df_full_train.churn)


# Now I will apply the function to my DataFrame...

# In[85]:


df_full_train[categorical].apply(mutual_info_churn_score)


# Next, I will sort it to have the values in descending order (highest/most important values first)

# In[86]:


mi = df_full_train[categorical].apply(mutual_info_churn_score)
mi.sort_values(ascending = False)


# It can be seen that contract is the most important variable and gender is the least important variable. Also, variables like onlinesecurity, techsupport, internetservice are quite/relatively important, and in relative terms, gender, phoneservice, multiplelines, seniorcitizen, etc are not as important.
# 
# This mutual information score helps to show variables that are more important or less important for this model

# ### Feature Importance: Correlation
# [Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) is a way to measure the importance for numerical features...

# In[87]:


df_full_train[numerical]


# Next, I will correlate the numerical variables with churn

# In[88]:


df_full_train[numerical].corrwith(df_full_train.churn)


# In[89]:


plt.figure(figsize=(5, 5))

# Calculate correlation between numerical features and 'churn'
correlation_matrix = df_full_train[numerical].corrwith(df_full_train['churn'])

# Reshape the correlation series into a DataFrame
correlation_df = pd.DataFrame(correlation_matrix, columns=['correlation'])

# Create a heatmap using the reshaped DataFrame
hm = sns.heatmap(correlation_df.transpose(), cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10})

plt.show()


# **Tenure:** There is a moderately negative correlation of approximately -0.352 with churn. This indicates that as the tenure (the duration a customer has been with the company) increases, the likelihood of churn decreases. In other words, customers who have been with the company for a longer time are less likely to churn.
# 
# **Monthly Charges:** There is a positive correlation of approximately 0.197 with churn. This suggests that as the monthly charges increase, the likelihood of churn also increases. Customers with higher monthly charges are more likely to churn.
# 
# **Total Charges:** There is a negative correlation of approximately -0.196 with churn. This implies that as the total charges incurred by a customer increase, the likelihood of churn decreases. Customers who have accumulated higher total charges are less likely to churn. This seems counterintuitive but can be explained by the fact that the longer customers stay with the company, the more charges they would incur, which explains the negative correlation..
# 
# > These correlations provide insights into the relationships between these numerical features and customer churn, which can be valuable for understanding factors that influence churn and for developing strategies to reduce it.

# I will investigate this further using tenure for example

# In[90]:


df_full_train[df_full_train.tenure <=2].churn.mean()


# This indicates that among customers who have a tenure of 2 months or less, the average churn rate is approximately 59.53%.
# 
# In other words, nearly 59.53% of customers who have been with the company for a very short period (2 months or less) tend to churn. This is a significant churn rate within this specific group of new customers, suggesting that the early stages of the customer relationship are critical, and there might be factors during this period that lead to a higher likelihood of churn. Understanding and addressing these factors could be essential for retaining new customers.

# In[91]:


df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean()


# This indicates that among customers who have a tenure greater than 2 months but less than or equal to 12 months, the average churn rate is approximately 39.94%.
# 
# In other words, for customers who have been with the company for a relatively short period, specifically between 2 months and 12 months, the churn rate is approx 40%. This suggests that there is a lower likelihood of churn among customers who have been with the company for a bit longer compared to the very new customers (as was seen in the bivariate analysis of "churn by tenure" earlier). This insight might be valuable for tailoring retention strategies for customers in this tenure range.

# In[92]:


df_full_train[df_full_train.tenure > 12].churn.mean()


# This indicates that among customers who have a tenure greater than 12 months, the average churn rate is approximately 17.63%.
# 
# In other words, for customers who have been with the company for more than a year (12 months or longer), the churn rate is around 17.63%. This suggests that there is a significantly lower likelihood of churn among customers with longer tenures, indicating that customer loyalty tends to increase with the duration of their relationship with the company. This insight highlights the importance of customer retention strategies for newer customers, as they are more likely to churn compared to long-standing customers.

# I will visualize this for clearer insight...

# In[93]:


import matplotlib.pyplot as plt

# I will define tenure groups and corresponding churn rates
tenure_groups = ['0-2 months', '3-12 months', '12+ months']
churn_rates = [59.5, 39.9, 17.6] 

# Next, I will create a bar chart
plt.figure(figsize=(8, 6))
plt.bar(tenure_groups, churn_rates, color=['#AED9E0', '#B8E986', '#F4A7B9'])
plt.xlabel('Tenure Groups')
plt.ylabel('Churn Rate (%)')
plt.title('Churn Rate by Tenure Group')
plt.ylim(0, 100)  # y-axis limits set to represent percentages (0-100%)
plt.show()


# > Increase in tenure leades to a decrease in churn rate (negative correlation)

# I will create another example using monthlycharges...

# In[94]:


df_full_train[df_full_train.monthlycharges <=20].churn.mean()


# This group includes customers with monthly charges less than or equal to $20. The churn rate in this group is relatively low, indicating that customers in this range are less likely to churn, with only 8.8% of them churning.

# In[95]:


df_full_train[(df_full_train.monthlycharges >20) & (df_full_train.monthlycharges <50)].churn.mean()


# This group includes customers with monthly charges greater than $20 but less than $50. The churn rate in this group is higher than the previous group, suggesting that customers in this range have a moderate likelihood of churning, with 18.4% of them churning.

# In[96]:


df_full_train[df_full_train.monthlycharges >=50].churn.mean()


# This group includes customers with monthly charges greater than or equal to $50. The churn rate in this group is the highest among the three ranges, indicating that customers with higher monthly charges are more likely to churn, with 32.5% of them churning.
# 
# 

# I will visualize this...

# In[97]:


charge_ranges = ["0-20", "21-50", "50+"]
churn_rates_percent = [8.8, 18.4, 32.5] 

plt.figure(figsize=(8, 6))
plt.bar(charge_ranges, churn_rates_percent, color=['skyblue', 'lightcoral', 'lightgreen'])
plt.xlabel("Monthly Charges")
plt.ylabel("Churn Rate (%)")
plt.title("Churn Rate by Monthly Charges")
plt.ylim(0, 40) 
plt.show()


# > The churn rate tends to increase as the monthly charges increase, with the highest churn rate observed in the group of customers with the highest monthly charges (Positive correlation)
# 

# Now, if I care about the importance but not the direction of the corelation values, I will proceed to absolute the values..
# 
# 

# In[98]:


df_full_train[numerical].corrwith(df_full_train.churn).abs()


# > Tenure is seen to be the most important variable, followed by monthlycharges, and totalcharges is the least important value.

# ### One-hot Encoding
# Next, I will encode the categorical features using Scikit-Learn

# In[99]:


from sklearn.feature_extraction import DictVectorizer


# In[100]:


df_train[['gender', 'contract']].iloc[:100] #looking at the first 100


# In[101]:


train_dicts = df_train[categorical + numerical].to_dict(orient='records') #converting it to a dictionary


# In[102]:


train_dicts[0]


# In[103]:


dv = DictVectorizer(sparse=False) #teach the vectorizer what kind of values are present


# In[104]:


X_train = dv.fit_transform(train_dicts) #training & transforming our dictvectorizer into a feature matrix


# In[105]:


val_dicts = df_val[categorical + numerical].to_dict(orient='records') #repeat for validation dataset


# In[106]:


X_val = dv.fit_transform(val_dicts) # I only transform the validation dataset, not fit it


# ### Logistic Regression
# - Binary classification
# - Linear vs logistic regression

# In[107]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z)) #function used to convert a score into a probability


# In[108]:


z = np.linspace(-7, 7, 51)


# In[109]:


sigmoid(z)


# In[110]:


plt.plot(z, sigmoid(z));


# In[111]:


def linear_regression(xi):
    result = w0
    
    for j in range(len(w)):
        result = result + xi[j] * w[j]
        
    return result


# In[112]:


def logistic_regression(xi):
    score = w0
    
    for j in range(len(w)):
        result = result + xi[j] * w[j]
     
    result = sigmoid(score) # in log reg, I convert the number/score I get into a value between 0 and 1 (major diff between log reg & lin reg)
    return result


# ### Training Logistic Regression with Scikit-Learn
# - Train a model with Scikit-Learn
# - Apply it to the validation dataset
# - Calculate the accuracy

# In[113]:


from sklearn.linear_model import LogisticRegression


# In[114]:


model = LogisticRegression()
model.fit(X_train, y_train)


# In[115]:


model.intercept_[0]


# In[116]:


model.coef_[0].round(3)


# In[117]:


#first column = probability of belonging to the -ve class (not churning), while the 2nd column = probability of churning (column I am interested in)
model.predict_proba(X_train) 


# In[118]:


y_pred = model.predict_proba(X_train)[:, 1] #training dataset


# In[119]:


y_pred = model.predict_proba(X_val)[:, 1] #validation dataset


# In[120]:


churn_decision = (y_pred >= 0.5) #using a default threshhold of 0.5 for churn decision
churn_decision   #false = not churning, true = likely to churn


# In[121]:


df_val[churn_decision].customerid #output reveals those who will churn (and will get promotional emails, discounts, etc)


# testing the (accuracy of the) model to check for correctly made predictions
# 

# In[122]:


y_val


# In[123]:


churn_decision.astype(int)   #I am interested in how many of them match in total btw y_val & churn_decision


# In[124]:


y_val == churn_decision   #true if numbers match and false if they don't


# In[125]:


#next, I use mean to see the number that actually match (correctly returns 80% of our predictions)

(y_val == churn_decision).mean()


# In[126]:


df_pred = pd.DataFrame()
df_pred['probaility'] = y_pred
df_pred['prediction'] = churn_decision.astype(int)
df_pred['actual'] = y_val
df_pred


# In[127]:


df_pred['correct'] = df_pred.prediction == df_pred.actual
df_pred


# ### Model Interpretation
# - Look at the coefficients
# - Train a smaller model with fewer features

# In[128]:


dv.feature_names_ #or dv.get_feature_names_out()


# In[129]:


model.coef_[0].round(3)


# In[130]:


#joining both lines of code:

dict(zip(dv.feature_names_, model.coef_[0].round(3)))


# In[131]:


#training a smaller model

small = ['contract', 'tenure', 'monthlycharges']


# In[132]:


df_train[small].iloc[:10].to_dict(orient='records') #use this for vectorizer


# In[133]:


dicts_train_small = df_train[small].to_dict(orient='records')
dicts_val_small = df_val[small].to_dict(orient='records')


# In[134]:


dv_small = DictVectorizer(sparse=False)
dv_small.fit(dicts_train_small)


# In[135]:


dv_small.feature_names_


# In[136]:


X_train_small = dv_small.transform(dicts_train_small)


# In[137]:


model_small = LogisticRegression()
model_small.fit(X_train_small, y_train)


# In[138]:


w0 = model_small.intercept_[0]
w0


# In[139]:


w = model_small.coef_[0]
w.round(3)


# In[140]:


#joining them together

dict(zip(dv_small.feature_names_, w.round(3)))


# In[141]:


'''
For example, if there is a customer who is on a monthly contract, pays monthly charges of $50, 
and has been with the company for 5 months, it can be calculated as:
'''

sigmoid(-2.47 + 0.97 + 50 * 0.027 + 5 * (-0.036))     #note that -2.47 is the intercept (w0)


# the probability of this customer churning is 42%

# ### Using the Model

# In[142]:


dicts_full_train = df_full_train[categorical + numerical].to_dict(orient='records')


# In[143]:


#create dictvectorizer

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)


# In[144]:


y_full_train = df_full_train.churn.values


# In[145]:


model = LogisticRegression()
model.fit(X_full_train, y_full_train)   #I have my model


# In[146]:


#repeat the same process for the test dataset

dicts_test = df_test[categorical + numerical].to_dict(orient='records')


# In[147]:


#apply dictvectorizer

X_test = dv.transform(dicts_test)  #here, I only transform, not fit


# In[148]:


#now, apply model

y_pred = model.predict_proba(X_test)[:, 1]


# In[149]:


#make decisions (e.g. clients higher than 0.5 are more likely to churn)

churn_decision = (y_pred >= 0.5)


# In[150]:


#test if churn decision is correct (compute Accuracy)

(churn_decision == y_test).mean()


# Accuracy is 81.5% accurate (slightly more accurate. We want to avoid a situation where the difference in accuracy is large)

# In[151]:


'''
Using the model: For example, I would like to find out if there is a customer who is more likely to leave or not, 
and if they are more likely to leave, I'd like to send them a promotional email asking them not to leave. 
I will choose a random customer from my data set
'''

customer = dicts_test[10]
customer


# In[152]:


'''
I will proceed to compute a score for a male, who is a senior, who lives with a partner, and has dependents, 
no tech support, have streaming tv, on monthly contract, have tenure of 32 months, with monthly charges of 93.95, etc
'''

X_small = dv.transform([customer])


# In[153]:


X_small.shape


# > 1 customer with 45 features

# In[154]:


#put into the model

model.predict_proba(X_small)[0, 1]


# > The model tells us that this senior male has only 40% of churning, so I will not send him a promotional email

# In[155]:


#check if he was actually going to churn

y_test[10]


# He was not going to churn so in this case, my decision not to send him a promotional email was correct

# In[156]:


# Example 2

customer2 = dicts_test[-1]
customer2


# In[157]:


'''
I will proceed to compute a score for a female, who is not a senior (younger), who lives with a partner, 
and has dependents, no tech support, have streaming tv, on monthly contract, has tenure of 17 months, 
with monthly charges of 104.2, etc
'''

X_small = dv.transform([customer2])


# In[158]:


#put into the model

model.predict_proba(X_small)[0, 1]


# > The model tells us that this customer2 has almost 60% of churning, so I will send her a promotional email, offer some discount to change her mind, etc.

# In[159]:


#check if she was actually going to churn

y_test[-1]


# She was going to churn so in this case, my decision to send her a promotional email was correct

# ### Evaluation Metrics
# 
# - Determine if the model trained for predicting churn (in previous project) is a good one.
# - Metric - function that compares the predictions with the actual values and outputs a single number that tells how good the predictions are

# In[160]:


dv = DictVectorizer(sparse=False)

train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

model = LogisticRegression()
model.fit(X_train, y_train)


# In[161]:


val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)

y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
(y_val == churn_decision).mean()


# ## Accuracy and dummy model
# - Evaluate the model on different thresholds
# - Check the accuracy of dummy baselines

# In[162]:


len(y_val)


# In[163]:


(y_val == churn_decision).mean()


# In[164]:


(y_val == churn_decision).sum()


# Number of correct decisions made = 1132
# 
# Next, I divide this figure by the total number of customers to get the accuracy

# In[165]:


1132/ 1409


# In[166]:


accuracy_score(y_val, y_pred >= 0.5)


# In[167]:


precision_score(y_val, y_pred >= 0.5)


# In[168]:


recall_score(y_val, y_pred >= 0.5) 


# In[169]:


f1_score(y_val, y_pred >= 0.5)


# In[170]:


roc_auc_score(y_val, y_pred >= 0.5)


# I will further calculate/examine the *precision_score, recall_score, f1_score, roc_auc_score* later on as I proceed.  
# 
# **Now, note that the decision to predict churn or not churn was 50%.    
# Next, I will confirm if that was actually a good decision (as opposed to 30% or 60% perhaps)**

# In[171]:


thresholds = np.linspace(0, 1, 21)

scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)


# It can be seen that 0.50 is indeed the best threshold (0.803). It's slighly more accurate than 0.55, and the threshold steadily declines at other decision scores. It can also be seen on the chart below (the line chart peaks at 0.5).

# In[172]:


plt.plot(thresholds, scores);


# In[173]:


from collections import Counter


# In[174]:


Counter(y_pred >= 1.0)


# This counts the occurrences of `True` and `False` values in the condition `y_pred >= 1.0`. In other words, it is counting how many predictions have a probability greater than or equal to 1.0. The result is a Counter object, and in this case, it shows that there are 1409 instances where the condition is False (customers not churning).

# In[175]:


1 - y_val.mean()


# This is calculating the complement of the mean value of `y_val`.    
# `y_val` contains the actual binary labels for whether customers churned or not (1 or 0).    
# The mean of `y_val` represents the proportion of customers who churned (subtracting this mean from 1 gives the proportion of customers who did not churn).

# > Accuracy score can be quite misleading (esecially in cases of class imbalance). It usually does not say how good the model is as well.
# >
# > There are other ways of rating binary classification models (other than accuracy) that are quite useful (especially when talking about problems with class imbalance like the churn prediction).

# ## Confusion table
# - Different types of errors and correct decisions
# - Arranging them in a table

# In[176]:


actual_positive = (y_val == 1) #correct prediction of customers who churn
actual_negative = (y_val == 0) #correct prediction of customers who did not churn


# In[177]:


t = 0.5   #threshold
predict_positive = (y_pred >= t)   #predict positive when above threshold
predict_negative = (y_pred < t)    #predict negative when below threshold


# >Next, combine predictions and actual outcomes. Focus is on cases when both predicted and actual positives are `True`. 
# >
# >(Remember that using the logical operator `and`, both cases have to be true to return a `True`. If one case is true ahile the other is false, it will return a false). 
# >
# >The `'&'` operator computes the element-wise logically.

# In[178]:


tp = (predict_positive & actual_positive).sum()
tn = (predict_negative & actual_negative).sum()

fp = (predict_positive & actual_negative).sum()
fn = (predict_negative & actual_positive).sum()


# *Note Values:*
# 
# - TP = 210
# - TN = 922
# - FP = 101
# - FN = 176

# In[179]:


confusion_matrix = np.array([
    [tn, fp],
    [fn, tp]
])
confusion_matrix


# Next, normalize the matrix (instead of absolute numbers, make it relative numbers)

# In[180]:


(confusion_matrix / confusion_matrix.sum()).round(2)


# *Relative Values:*
# 
# - TP = 15%
# - TN = 65%
# - FP = 8%
# - FN = 12%
# 
# > Accuracy = 80% (65% + 15%)

# Next, I will create a Confusion Matrix

# In[184]:


from sklearn.metrics import confusion_matrix

# Assuming y_pred contains probabilities
threshold = 0.5
y_pred_binary = (y_pred > threshold).astype(int)

# Now, compute the confusion matrix
lcm = confusion_matrix(y_val, y_pred_binary)

sns.heatmap(lcm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


# **True Positive (TP): 210**   
# These are cases where the model correctly predicted that customers would churn. It contributes to the accuracy of the model in identifying churn instances.
# 
# **True Negative (TN): 922**   
# These are cases where the model correctly predicted that customers would not churn. It contributes to the accuracy of the model in identifying non-churn instances.
# 
# **False Positive (FP): 101**   
# These are cases where the model incorrectly predicted that customers would churn (Type I error). In other words, the model made an incorrect positive prediction. In this context, this might lead to unnecessary interventions or actions for customers who are not actually likely to churn (like wrongly sending them promotional mails and/or offering discounts).
# 
# **False Negative (FN): 176**   
# These are cases where the model incorrectly predicted that customers would not churn (Type II error). In other words, the model made an incorrect negative prediction. In this context, this might result in the failure to identify customers who are at risk of churning, leading to missed opportunities for retention efforts.

# ## Precision and Recall

# In[ ]:


#Precision tell us how many positive predictions turned out to be correct (fraction of correct positive predictions)

p = tp / (tp + fp)
p


# In[ ]:


precision_score(y_val, y_pred >= 0.5)


# Precision is 67.5% (these are those who correctly receive the promotional email)
# 
# This means that out of all the positive predictions made by the model (instances where it predicted customers would churn), approximately 67.5% of those predictions turned out to be correct. In other words, when the model predicted churn, it was accurate about 67.5% of the time. 
# 
# Conversely, about 33% are mistakes (these are people who are not supposed to receive the promotional email but received it anyway).
# 
# High precision is desirable, especially in scenarios where the cost of false positives (incorrectly predicting churn) is high. However, it's essential to consider precision along with other metrics like recall, accuracy, and the specific business context to have a comprehensive understanding of the model's performance.

# In[ ]:


# Recall measures the model's ability to capture all the positive instances. 
# It answers the question: "Out of all the actual positive instances, how many did the model correctly predict?" 

'''High recall indicates that the model is effective at capturing most of the positive instances, 
but it may also include false positives.'''

r = tp / (tp + fn)
r


# In[ ]:


recall_score(y_val, y_pred >= 0.5) 


# The recall value of 0.544 means that the model is capturing approximately 54.4% of the customers who are actually churning. In other words, there are instances where the model fails to identify customers who are churning (46%), leading to false negatives.
# 
# A higher recall is generally desirable in scenarios where missing positive instances (churn cases in this context) is more critical than having a precise prediction. However, it is important to consider the trade-off between precision and recall, as improving one may negatively impact the other. The specific balance depends on the goals and priorities of the business problem being addressed with the churn prediction model.
# 
# > It is clear to see that ***`Accuracy`*** is not the best metric for identifying churning users. Initially, the model seemed to be doing good at an accuracy of 80% but after looking at the precision and recall, it is clear to see that the model fails to identify 46% of users and could actually cause the company to send promotional email to 33% of users who we thought would churn but will likely only take advantage of the promotional discount. So, we see that how accuracy can be misleading (especially in cases of class imbalance). It is always good to look at precision and recall as well.

# ***`Class imbalance`*** in model building refers to a situation where the distribution of classes in the target variable is not uniform, meaning that one class significantly outnumbers the other(s). In binary classification problems, this typically involves one class being more prevalent than the other. For example, in a customer churn prediction scenario, if the majority of customers do not churn (class "no churn"), and only a small percentage actually churn (class "churn"), there will be a class imbalance as in this instance.
# 
# Class imbalance can pose challenges for machine learning models because they might become biased toward the majority class. In scenarios with imbalanced classes, a model can achieve high accuracy by simply predicting the majority class most of the time, even if it fails to correctly identify instances of the minority class.

# # ROC Curves
# 
# ## TPR and FRP

# A Receiver Operating Characteristic (ROC) curve is a graphical representation that illustrates the performance of a binary classification model at various classification thresholds. It plots the true positive rate (sensitivity) against the false positive rate (1 - specificity) for different threshold values.
# 
# - True Positive Rate (Sensitivity): The proportion of actual positive instances correctly identified by the model.
# 
# - False Positive Rate (1 - Specificity): The proportion of actual negative instances incorrectly identified as positive by the model.
# 
# By varying the classification threshold, the ROC curve shows how sensitivity and specificity trade off against each other. The area under the ROC curve (AUC-ROC) is a commonly used metric to quantify the overall performance of a binary classification model. A higher AUC-ROC indicates better discrimination between positive and negative instances across different thresholds, with a value of 1.0 representing perfect performance.

# In[ ]:


tpr = tp / (tp + fn)
tpr


# The TPR value of approximately 0.54 (54%) suggests that the model correctly identified about 54% of the customers who actually churned.
# 
# TPR is particularly relevant when the cost of missing positive instances (false negatives) is high. In the context of customer churn, it indicates the effectiveness of the model in capturing customers who are likely to leave.

# In[ ]:


fpr = fp / (fp + tn)
fpr


# The FPR value of approximately 0.10 (10%) suggests that the model incorrectly identified about 10% of the customers as churned when they did not actually churn.
# 
# FPR is crucial when the cost of falsely identifying negative instances as positive (false positives) is high. In the context of customer churn, it indicates the rate at which the model makes incorrect predictions about customers who are not likely to leave.

# > The threshold of 0.5 (50%) returned the values of FPR & TPR above. ROC curve evaluates all the possible thresholds. I will compute them below (using the linspace):

# In[ ]:


scores = []

thresholds = np.linspace(0, 1, 101)

for t in thresholds:
    actual_positive = (y_val == 1)
    actual_negative = (y_val == 0)
    
    predict_positive = (y_pred >= t)
    predict_negative = (y_pred < t)

    tp = (predict_positive & actual_positive).sum()
    tn = (predict_negative & actual_negative).sum()

    fp = (predict_positive & actual_negative).sum()
    fn = (predict_negative & actual_positive).sum()
    
    scores.append((t, tp, fp, fn, tn))


# In[ ]:


columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
df_scores = pd.DataFrame(scores, columns=columns)

df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)


# In[ ]:


plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR')
plt.xlabel('Threshold')
plt.legend();


# The plot visualizes how TPR and FPR change with varying classification thresholds.
# 
# The x-axis represents different threshold values, which are probability thresholds for classifying customers as churned or not churned. These thresholds influence the model's sensitivity and specificity.
# 
# At a threshold of 0.0, both the TPR and FPR are 1.0 indicating that the model is classifying all instances as positive (meaning that this model predicts everyone as churning). Therefore, it captures all true positives (TPR = 1.0), but it also incorrectly includes all negatives as positives (as churning), resulting in a FPR of 1.0. 
# 
# The TPR and FPR descend at different rates. The FPR goes down faster (we want it to go down as fast as possible; we want to minimize it so we want it to be as low as possible), while for TPR, we want to keep it around 1.0

# ## Random model

# A random model in the context of binary classification makes predictions without any reliance on the input features. It essentially assigns class labels randomly, without considering the data's characteristics. For a balanced dataset, a random model would achieve an accuracy close to 50% by chance. It serves as a baseline comparison for evaluating the performance of more sophisticated models, like those trained on actual patterns in the data (in this case, the TPR & FPR).

# In[ ]:


np.random.seed(1)
y_rand = np.random.uniform(0, 1, size=len(y_val))


# In[ ]:


((y_rand >= 0.5) == y_val).mean()


# In[ ]:


def tpr_fpr_dataframe(y_val, y_pred):
    scores = []

    thresholds = np.linspace(0, 1, 101)

    for t in thresholds:
        actual_positive = (y_val == 1)
        actual_negative = (y_val == 0)

        predict_positive = (y_pred >= t)
        predict_negative = (y_pred < t)

        tp = (predict_positive & actual_positive).sum()
        tn = (predict_negative & actual_negative).sum()

        fp = (predict_positive & actual_negative).sum()
        fn = (predict_negative & actual_positive).sum()

        scores.append((t, tp, fp, fn, tn))

    columns = ['threshold', 'tp', 'fp', 'fn', 'tn']
    df_scores = pd.DataFrame(scores, columns=columns)

    df_scores['tpr'] = df_scores.tp / (df_scores.tp + df_scores.fn)
    df_scores['fpr'] = df_scores.fp / (df_scores.fp + df_scores.tn)
    
    return df_scores


# In[ ]:


df_rand = tpr_fpr_dataframe(y_val, y_rand)


# In[ ]:


plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR')
plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR')
plt.xlabel('Threshold')
plt.legend();


# The plot illustrates the Receiver Operating Characteristic (ROC) curve for a random model in the context of customer churn. The True Positive Rate (TPR) and False Positive Rate (FPR) are plotted against different threshold values. In a random model, as the threshold varies, both TPR and FPR change in a nearly linear fashion, descending from 1.0 to 0.0. The curve indicates that the model's ability to correctly identify true positives decreases along with the increase in false positives. 

# ## Ideal model

# An ideal model, in the context of customer churn prediction, would have a Receiver Operating Characteristic (ROC) curve that closely follows the upper-left corner of the plot. This ideal curve would have a True Positive Rate (TPR) of 1.0 (100%) and a False Positive Rate (FPR) of 0.0 (0%) across all possible threshold values. Essentially, the ideal model would be able to perfectly distinguish between customers who are likely to churn and those who are not, with no false positive predictions and a maximized true positive rate.

# In[ ]:


#counting negative and positive

num_neg = (y_val == 0).sum()
num_pos = (y_val == 1).sum()
num_neg, num_pos


# In[ ]:


y_ideal = np.repeat([0, 1], [num_neg, num_pos])
y_ideal #validation set (everything is ordered here)

y_ideal_pred = np.linspace(0, 1, len(y_val)) #create predictions


# In[ ]:


1 - y_val.mean()


# In[ ]:


accuracy_score(y_ideal, y_ideal_pred >= 0.726)


# In[ ]:


df_ideal = tpr_fpr_dataframe(y_ideal, y_ideal_pred)
df_ideal[::10]


# In[ ]:


plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR')
plt.xlabel('Threshold')
plt.legend();


# In[ ]:


plt.plot(df_scores.threshold, df_scores['tpr'], label='TPR', color='black')
plt.plot(df_scores.threshold, df_scores['fpr'], label='FPR', color='blue')

plt.plot(df_ideal.threshold, df_ideal['tpr'], label='TPR ideal')
plt.plot(df_ideal.threshold, df_ideal['fpr'], label='FPR ideal')

# plt.plot(df_rand.threshold, df_rand['tpr'], label='TPR random', color='grey')
# plt.plot(df_rand.threshold, df_rand['fpr'], label='FPR random', color='grey')

plt.xlabel('Threshold')
plt.legend();


# In[ ]:


plt.figure(figsize=(5, 5))

plt.plot(df_scores.fpr, df_scores.tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend();


# In[ ]:


from sklearn.metrics import roc_curve


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred)


# In[ ]:


plt.figure(figsize=(5, 5))

plt.plot(fpr, tpr, label='Model')
plt.plot([0, 1], [0, 1], label='Random', linestyle='--')

plt.xlabel('FPR')
plt.ylabel('TPR')

plt.legend();


# ## ROC AUC
# - Area under the ROC curve
# - Interpretation of AUC

# In[ ]:


from sklearn.metrics import auc


# In[ ]:


auc(fpr, tpr)


# In[ ]:


auc(df_scores.fpr, df_scores.tpr)


# In[ ]:


auc(df_ideal.fpr, df_ideal.tpr)


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred)
auc(fpr, tpr)


# In[ ]:


roc_auc_score(y_val, y_pred)


# In[ ]:


neg = y_pred[y_val == 0]
pos = y_pred[y_val == 1]


# In[ ]:


import random


# In[ ]:


n = 100000
success = 0 

for i in range(n):
    pos_ind = random.randint(0, len(pos) - 1)
    neg_ind = random.randint(0, len(neg) - 1)

    if pos[pos_ind] > neg[neg_ind]:
        success = success + 1

success / n


# In[ ]:


n = 50000

np.random.seed(1)
pos_ind = np.random.randint(0, len(pos), size=n)
neg_ind = np.random.randint(0, len(neg), size=n)

(pos[pos_ind] > neg[neg_ind]).mean()


# ## Cross-Validation
# - Evaluating the same model on different subsets of data
# - Getting the average prediction and the spread within predictions

# In[ ]:


def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)
    
    return dv, model


# In[ ]:


dv, model = train(df_train, y_train, C=0.001)


# In[ ]:


def predict(df, dv, model):
    dicts = df[categorical + numerical].to_dict(orient='records')

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# In[ ]:


y_pred = predict(df_val, dv, model)


# In[ ]:


from sklearn.model_selection import KFold


# In[ ]:


from tqdm.auto import tqdm


# In[ ]:


n_splits = 5

for C in tqdm([0.001, 0.01, 0.1, 0.5, 1, 5, 10]):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        df_train = df_full_train.iloc[train_idx]
        df_val = df_full_train.iloc[val_idx]

        y_train = df_train.churn.values
        y_val = df_val.churn.values

        dv, model = train(df_train, y_train, C=C)
        y_pred = predict(df_val, dv, model)

        auc = roc_auc_score(y_val, y_pred)
        scores.append(auc)

    print('C=%s %.3f +- %.3f' % (C, np.mean(scores), np.std(scores)))


# In[ ]:


scores


# In[ ]:


dv, model = train(df_full_train, df_full_train.churn.values, C=1.0)
y_pred = predict(df_test, dv, model)

auc = roc_auc_score(y_test, y_pred)
auc


# ## Model Building using Random Forest Classifier

# In[187]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Convert categorical and numerical features to a format suitable for the model
train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

# Create and train the RandomForestClassifier
rf_model = RandomForestClassifier(random_state=1)
rf_model.fit(X_train, y_train)

# Transform the validation data and make predictions
val_dict = df_val[categorical + numerical].to_dict(orient='records')
X_val = dv.transform(val_dict)
y_pred_rf = rf_model.predict_proba(X_val)[:, 1]

# Set a threshold for churn decision
churn_decision_rf = (y_pred_rf >= 0.5)

# Evaluate the performance of the RandomForestClassifier
accuracy_rf = accuracy_score(y_val, churn_decision_rf)
precision_rf = precision_score(y_val, churn_decision_rf)
recall_rf = recall_score(y_val, churn_decision_rf)
f1_rf = f1_score(y_val, churn_decision_rf)
roc_auc_rf = roc_auc_score(y_val, y_pred_rf)

# Print the metrics
print('RandomForestClassifier Metrics:')
print('Accuracy:', accuracy_rf)
print('Precision:', precision_rf)
print('Recall:', recall_rf)
print('F1-score:', f1_rf)
print('AUC-ROC:', roc_auc_rf)


# **Accuracy (0.7904):**
# Accuracy is the ratio of correctly predicted instances to the total instances.
# This means that the RandomForestClassifier correctly predicted whether a customer will churn or not approximately 79.04% of the time.
# 
# **Precision (0.6625):**
# Precision is the ratio of correctly predicted positive observations to the total predicted positives.   
# This means that when the RandomForestClassifier predicts a customer will churn, it is correct about 66.25% of the time.
# 
# **Recall (0.5064):**
# Recall (Sensitivity or True Positive Rate) is the ratio of correctly predicted positive observations to the all observations in the actual class.    
# This means that the RandomForestClassifier is able to capture about 50.64% of the customers who actually churned.
# 
# **F1-score (0.5740):**
# F1-score is the weighted average of Precision and Recall. It's a metric that considers both false positives and false negatives.   
# In this context, an F1-score of 0.5740 indicates a balance between precision and recall.
# 
# **AUC-ROC (0.8243):**
# AUC-ROC (Area Under the Receiver Operating Characteristic curve) is a metric that represents the area under the curve when plotting the true positive rate against the false positive rate at various thresholds.
# An AUC-ROC of 0.8243 suggests good overall performance of the model, especially in distinguishing between churn and non-churn instances.

# Next, I'll build the confusion matrix for the Random Forest Classifier Model

# In[190]:


from sklearn.metrics import confusion_matrix

# Assuming y_pred_rf contains probabilities
threshold_rf = 0.5
y_pred_binary_rf = (y_pred_rf > threshold_rf).astype(int)

# Now, compute the confusion matrix for RandomForestClassifier
lcm_rf = confusion_matrix(y_val, y_pred_binary_rf)

# Plot the confusion matrix
plt.figure(figsize=(5, 5))
sns.heatmap(lcm_rf, annot=True, cmap='Blues', fmt='g', cbar=False,
            annot_kws={'size': 14}, xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted', fontsize=10)
plt.ylabel('Actual', fontsize=10)
plt.title('RandomForestClassifier Confusion Matrix', fontsize=12)
plt.show()


# **True Positive (TP):** Customers who were predicted to churn and actually did.    
# The model correctly predicted 195 instances as positive (churn), and they were actually positive.
# 
# **True Negative (TN):** Customers who were predicted not to churn and actually did not.    
# The model correctly predicted 932 instances as negative (not churn), and they were actually negative.
# 
# **False Positive (FP):** Customers who were predicted to churn but did not.    
# The model incorrectly predicted 91 instances as positive (churn), but they were actually negative.
# 
# **False Negative (FN):** Customers who were predicted not to churn but did.    
# The model incorrectly predicted 191 instances as negative (not churn), but they were actually positive.

# > Both Logistic Regression and Random Forest Classifier models show relatively similar accuracy, precision, and AUC-ROC values.
# >
# >The Logistic Regression model has a slightly higher recall, indicating a better ability to identify actual positive cases.
# >
# >The Random Forest Classifier demonstrates a higher AUC-ROC, suggesting better overall discriminatory power.
# >
# >In the context of customer churn, identifying potential churners (recall) is crucial. Hence, the Logistic Regression model might be preferred in this scenario.
# >> The choice between the models depends on the specific business goals and priorities of ConnectTel. If precision is crucial, the Random Forest Classifier might be a suitable choice. If recall and identifying potential churners are a priority, the Logistic Regression model may be more appropriate.

# Next, I will apply Eight different Machine Learning Algorithms to the dataset...

# In[208]:


import warnings
warnings.filterwarnings("ignore")

# Using the df_train, df_val, y_train, y_val, numerical, categorical, and dv defined as before

# Convert categorical and numerical features to a format suitable for the model
train_dict = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dict)

# Define classifiers
classifiers = [
    [XGBClassifier(), 'XGB Classifier'],
    [RandomForestClassifier(random_state=1), 'Random Forest'],
    [KNeighborsClassifier(), 'K-Nearest Neighbours'],
    [SGDClassifier(), 'SGD Classifier'],
    [SVC(), 'SVC'],
    [GaussianNB(), 'Naive Bayes'],
    [DecisionTreeClassifier(random_state=42), 'Decision Tree'],
    [LogisticRegression(), 'Logistic Regression']
]


# In[209]:


acc_list = {}
precision_list = {}
recall_list = {}
roc_list = {}

# Evaluate each classifier
for classifier in classifiers:
    model = classifier[0]
    model_name = classifier[1]
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Transform the validation data and make predictions
    val_dict = df_val[categorical + numerical].to_dict(orient='records')
    X_val = dv.transform(val_dict)
    pred = model.predict(X_val)
    
    # Calculate metrics
    a_score = accuracy_score(y_val, pred)
    p_score = precision_score(y_val, pred)
    r_score = recall_score(y_val, pred)
    roc_score = roc_auc_score(y_val, pred)
    
    # Store metrics in dictionaries
    acc_list[model_name] = [round(a_score*100, 2)]
    precision_list[model_name] = [round(p_score*100, 2)]
    recall_list[model_name] = [round(r_score*100, 2)]
    roc_list[model_name] = [round(roc_score*100, 2)]

    # Print a new line for each model except the last one
    if model_name != classifiers[-1][1]:
        print('')


# In[204]:


print('Accuracy Score')
s1 = pd.DataFrame(acc_list)
s1.head()


# In[205]:


print('Precision')
s2 = pd.DataFrame(precision_list)
s2.head()


# In[206]:


print('Recall')
s3 = pd.DataFrame(recall_list)
s3.head()


# In[207]:


print('ROC Score')
s4 = pd.DataFrame(roc_list)
s4.head()


# ### Detailed Model Comparison:
# 
# 1. **Logistic Regression:**
# Accuracy: 80.34%   
# Precision: 67.52%   
# Recall: 54.4%   
# ROC-AUC: 72.27%   
# **Analysis:**    
# Good balance between precision and recall.   
# High accuracy, suitable for a baseline model. 
# 
# >
# 
# 2. **Random Forest Classifier:** 
# Accuracy: 79.84%    
# Precision: 67.11%    
# Recall: 51.81%    
# ROC-AUC: 83.11%    
# **Analysis:**   
# High ROC-AUC indicates good overall performance.   
# Balanced precision and recall.   
# 
# >
# 
# 3. **XGB Classifier:**
# Accuracy: 77.86%    
# Precision: 61.71%    
# Recall: 50.52%    
# ROC-AUC: 69.35%    
# **Analysis:**   
# Moderate performance, slightly lower recall.
# 
# >
# 
# 4. **K-Nearest Neighbours:**
# Accuracy: 77.43%    
# Precision: 61.72%    
# Recall: 46.37%    
# ROC-AUC: 67.76%    
# **Analysis:**
# Moderate accuracy, lower recall.   
# 
# >
# 
# 5. **Naive Bayes:**
# Accuracy: 68.49%    
# Precision: 45.99%    
# Recall: 86.27%    
# ROC-AUC: 74.02%    
# **Analysis:**
# High recall but lower precision.   
# 
# >
# 
# 6. **Decision Tree:**
# Accuracy: 71.47%    
# Precision: 48.03%    
# Recall: 50.52%    
# ROC-AUC: 64.95%    
# **Analysis:**
# Comparable performance, slightly lower ROC-AUC.   
# 
# >
# 
# 7. **SGD Classifier:**
# Accuracy: 74.88%    
# Precision: 65.09%    
# Recall: 17.88%    
# ROC-AUC: 57.13%    
# **Analysis:**
# High precision but significantly lower recall.   
# 
# >
# 
# 8. **Support Vector Classifier (SVC):**
# Accuracy: 72.6%    
# Precision: 0.0%    
# Recall: 0.0%    
# ROC-AUC: 50.0% (Random)    
# **Analysis:**
# Poor performance, not suitable for practical use.
# 
# 
# ### Summary:
# - ***`RandomForestClassifier`*** is the standout performer, providing a good balance between precision and recall with a high ROC-AUC.
# 
# 
# - ***`Logistic Regression`*** offers a balanced approach, suitable as a baseline model with good precision and recall.
# 
# 
# - Depending on business priorities, Naive Bayes may be considered for its high recall, even though precision is compromised (consider trade-offs).
# 
# - XGB Classifier and K-Nearest Neighbours are reasonable choices but need further optimization.
# 
# - Avoid SGD Classifier and SVC due to significant trade-offs between precision and recall.
# 
# ### Recommendation for ConnectTel Telecom:
# Given the urgency of ConnectTel Telecom to address customer churn, the RandomForestClassifier is recommended as the primary and most suitable model. It provides a good overall performance, balancing accuracy, precision, and recall. However, depending on the specific business goals and trade-offs between precision and recall, Logistic Regression might also be a viable choice. Further optimization and monitoring are advisable for ongoing effectiveness.

# In[ ]:




