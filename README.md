# CONNECTTEL_CUSTOMER_CHURN_PREDICTION

### (Predicting Customer Churn with Machine Learning)

![image](https://github.com/ChidimmaIdika/CONNECTTEL_CUSTOMER_CHURN_PREDICTION/assets/137975543/4d064123-c7a3-452c-aa38-c4e9c05c02e6)

---
# About ConnectTel
*ConnectTel is a leading telecommunications company at the forefront of innovation and connectivity solutions. With a strong presence in the global market, ConnectTel has established itself as a trusted provider of reliable voice, data, and Internet services. Offering a comprehensive range of telecommunications solutions, including mobile networks, broadband connections, and enterprise solutions, ConnectTel caters to both individual and corporate customers, they are committed to providing exceptional customer service and cutting-edge technology. ConnectTel ensures seamless communication experiences for millions of users worldwide. Through strategic partnerships and a customer-centric approach, ConnectTel continues to revolutionize the telecom industry, empowering individuals and businesses to stay connected and thrive in the digital age.*

---



## Table of Contents

- [Introduction](#introduction)
- [Data Exploration and Preprocessing](#data-exploration-and-preprocessing)
- [Data Analysis and Feature Importance](#data-analysis-and-feature-importance)
- [Model Building and Deployment](#model-building-and-deployment)
- [Conclusion](#conclusion)


## Introduction
In today's highly competitive business landscape, retaining customers is crucial for the long-term success of any company. Customer churn, or the rate at which customers stop doing business with a company, can have a significant impact on revenue. ConnectTel Telecom Company faces the pressing need to address customer churn, which poses a significant threat to its business sustainability and growth. The company's current customer retention strategies lack precision and effectiveness, resulting in the loss of valuable customers to competitors. To overcome this challenge, ConnectTel aims to develop a robust customer churn prediction system for which I was contacted to handle as a Data Scientist. By leveraging advanced analytics and machine learning techniques on available customer data, the company seeks to accurately forecast customer churn and implement targeted retention initiatives. This proactive approach will enable ConnectTel to reduce customer attrition, enhance customer loyalty, and maintain a competitive edge in the highly dynamic and competitive telecommunications industry.

---
***In this GitHub post, I explored the challenge of predicting customer churn in the telecommunications industry. Leveraging machine learning algorithms, I analyzed a dataset provided by ConnectTel Telecom, aiming to build models that can effectively predict customers likely to churn, as well as make data-driven decisions.***

---

## Data Exploration and Preprocessing
In this first part of the project, I began by exploring and preparing my data. The ConnectTel dataset contains information about various customers, including demographic details, the type of services they are subscribed to, and their payment history.

- **Exploratory Data Analysis:** I kickstarted the project with Exploratory Data Analysis (EDA) to gain insights into the dataset. I loaded and examined the dataset to understand its structure. I visualized key features and analyzed summary statistics to get a sense of the data. Through visualizations and statistical summaries, I uncovered patterns, distribution of features, and potential correlations. Understanding the data is crucial for informed decision-making in subsequent stages.

- **Data Preprocessing:** I address missing values, convert categorical variables into numerical format, and standardize numerical features. This step ensures that the data is ready for modeling.

## Data Analysis and Feature Importance
In this section, I went deeper into the data to identify which features have the most significant impact on customer churn. Understanding feature importance is crucial for model interpretation.

- **Correlation Analysis:** I calculated the correlation between various features and the target variable (churn). For instance, I discovered that there is a negative correlation between tenure (the length of time a customer has been with the company) and churn, which makes intuitive sense.

- **Churn Rates by Monthly Charges:** I grouped customers based on their monthly charges and analyzed churn rates within these groups. The findings revealed a positive correlation between monthly charges and churn, where customers with higher charges are more likely to churn.

- **Feature Importance with Logistic Regression:** I employed logistic regression to understand the importance of each feature in predicting customer churn. Tenure is identified as the most crucial variable, followed by monthly charges and total charges.

## Model Building and Deployment
When I had a clear understanding of the data and feature importance, I moved on to building a predictive model for customer churn.

- **Feature Engineering:**  I delved into Feature Engineering to prepare the dataset for machine learning models. This involved one-hot encoding categorical variables, handling missing values, and standardizing numerical features. The dataset was then split into training and validation sets to facilitate model training and evaluation.

- **One-Hot Encoding:** I encoded categorical features using Scikit-Learn's DictVectorizer, allowing me to transform the data into a format suitable for machine learning models.

- **Logistic Regression:** For the initial modeling, I employed Logistic Regression, a powerful tool for ***binary classification***, to train a predictive model. This fundamental binary classification algorithm allowed me to evaluate the baseline performance of predicting customer churn. I then applied the model to a validation dataset to calculate the accuracy. Other metrics such as precision, recall, and ROC-AUC were analyzed, providing insights into the model's strengths and limitations. 

- **Random Forest Classifier Model:** 
Building upon the Logistic Regression model, I implemented a Random Forest Classifier. This more sophisticated algorithm provided competitive performance, and I visualized its predictions using a confusion matrix. Comparative analysis with Logistic Regression highlighted trade-offs between precision and recall.

- **Model Comparison and Selection:**
In the final phase, I compared eight different machine learning algorithms, including XGB Classifier, K-Nearest Neighbours, Naive Bayes, Decision Tree, SGD Classifier, and Support Vector Classifier. Metrics such as accuracy, precision, recall, and ROC-AUC were scrutinized. The RandomForestClassifier emerged as the recommended model, offering a balanced performance between precision and recall.

- **Model Interpretation:** I examined the coefficients of the model to understand how different features impact the likelihood of customer churn. This information allowed me to make informed decisions and take action to retain customers.

## Conclusion
This project illustrates how data-driven analysis and predictive modeling can be applied to address real-world challenges, such as customer churn. By identifying key features and building various models, this project equips ConnectTel Telecom with actionable insights into predicting customer churn, and the ability to make informed decisions to retain customers and optimize business operations. The RandomForestClassifier is recommended as the primary model, striking a balance between precision and recall. Logistic Regression serves as a reliable baseline model. The choice between the models depends on business priorities and trade-offs.

The complete workflow, from data exploration to model selection, is available in this GitHub repository, providing a comprehensive resource for understanding and implementing customer churn prediction in the telecommunications domain.

---
 ***For more details, you can view my code and documentation process here... [ConnectTel Customer Churn Prediction](https://github.com/ChidimmaIdika/CONNECTTEL_CUSTOMER_CHURN_PREDICTION/blob/Chidimma/CONNECTTEL%20CUSTOMER%20CHURN%20PREDICTION.ipynb)***  

 *If it takes longer than 10 seconds to render, kindly [click this link](https://nbviewer.org/github/ChidimmaIdika/CONNECTTEL_CUSTOMER_CHURN_PREDICTION/blob/Chidimma/CONNECTTEL%20CUSTOMER%20CHURN%20PREDICTION.ipynb)*

---

Feel free to explore the details, provide feedback, and adapt the models for your specific business needs!
---
