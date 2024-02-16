# Loan-Default-Prediction-using-Machine-Learning-Techniques
Assessed the likelihood of loan default based on customer demographics and financial data by fitting a model  using various machine learning techniques and thereby predict defaulting applicants.

Interest on loans and associated fees are some of the biggest revenue sources for most banks and credit unions. Considering the magnitude of risk and financial loss involved, it is essential for banks to give loans to credible applicants who are highly likely to pay back the loan amount.
However, the rate of loan default is increasing exponentially. According to Forbes, nearly 40 percent of borrowers are expected to default on their student loans by 2023. Considering the statistics, instead of making money from loan interest, banks will suffer a huge capital loss. In order to prevent the loss, it is very important to have a system in place which will accurately predict the loan defaulters even before approving the loan.

## Objective
The Objective of our project was to assess the likelihood of loan default based on customer demographics and financial data by fitting a model using various machine learning techniques and thereby predict defaulting applicants. Furthermore, we were interested to find the most significant variables that contribute to determining loan default.

## Dataset
https://www.kaggle.com/faressayah/lending-club-loan-defaulters-prediction

## Data Description 
The data that used here is from Lending club. Lending Club is a peer-to-peer lending platform which offers loan trading on a secondary market. 
So what is a Peer-to-Peer platform? Peer-to-Peer lending enables individuals to obtain loans directly from other individuals, cutting out the financial institution as the middleman. These lending websites connect borrowers directly to investors. These sites have a wide range of interest rates based on the creditworthiness of the applicant.
Lending club facilitates personal loans, business loans, and financing of medical procedures. The data includes information about past loan applicants and whether they were able to repay the loan or not.The dataset included 27 variables and 3 lakhs 96 thousand 30 observations.
Data consists of customer demographics as well as financial details such as total amount funded, every month instalment (EMI) and rate of interest. Data also has housing and customer employment information such as housing ownership, years in job and annual income

## Data Pre-processing
Revolving line utilization rate and number of public record bankruptcies had missing data points, but they accounted for less than 0.5% of the total data. So We removed the missing values in those columns. We imputed the missing values for the variable mortgage accounts. We also removed the observation with annual income equal to 0 because it is highly unlikely for a bank to give loan to a person with no income. We also converted the categorical variables into factors.

## EDA 
The pie chart shows the distribution of defaulters and non-defaulters. Out of the total observations, 80% of the people had fully paid the loan whereas 20% were defaulters.
Here we can see that the data is highly imbalanced.

![image](https://user-images.githubusercontent.com/70087327/132171076-46381378-adc9-43e7-807f-2fb5a73dfdf8.png)

Next is the bar graph for the term of the loan. For the customers who took a loan for 60 months, 32% defaulted whereas 68% did not. On the other hand, for the term of 36 months, only 16% defaulted whereas 84% did not default.

![image](https://user-images.githubusercontent.com/70087327/132170819-544cad54-0b14-431e-869f-8314b6237345.png)

Next is the bar graph for home ownership. Here we found that Applicants who have their own house seem to default less as compared to those who live on rent or mortgage.

![image](https://user-images.githubusercontent.com/70087327/132170929-47dfabce-bb37-468d-b4a7-8e4e68539b37.png)

Later we plotted the graph for the purpose of the loan. Most of the customers have applied for the purpose of debt consolidation followed by credit card, house improvement and others.

![image](https://user-images.githubusercontent.com/70087327/132170975-91ede4ab-efc4-4bfb-a02c-d37f024cca95.png)


## Train Test Split
We splitted the data into training and testing set in the ratio 75:25. 
When observation in one class is higher than the observation in other classes then there exists a class imbalance. 
Imbalance data can hamper our model accuracy. Out of total observations, 80.39% of the people have fully paid the loan where as 19.61% are defaulters. So, to overcome this challenge, we used oversampling method which duplicates random records from the minority class. 

![image](https://user-images.githubusercontent.com/70087327/130553635-9f25a570-99b9-47e9-aca6-27d9d878d31d.png)

## Model Fitting

### Logistic Regression
Logistic regression which is also known as classification algorithm is used to describe data and explain the relationship between a dependent binary variable and one or more nominal, ordinal, interval or ratio-level independent variables. logistic regression helps us to make informed decisions. Here in our case, Based on variables loan_amt, term, int_Rate, annual_inc, loan_Status , we can predict whether customer will default or not using logistic regression. 

 
### Random Forest                                                                                                                                                              
Random forest is a commonly-used supervised machine learning algorithm which combines the output of multiple decision trees to reach a single result. Since random forest can handle both regression and classification tasks with a high degree of accuracy, it is a popular method among data scientists. Feature bagging also makes the random forest an effective tool for estimating missing values as it maintains accuracy when a portion of the data is missing. 
In our case, we tried different resampling methods before applying the random forest algorithm and compared the respective accuracy values to find the best model. 
 
 
### KNN 
K Nearest Neighbor is one of the easiest and simplest machine learning algorithm which is mainly used for classification problems in industry.  
KNN algorithm uses ‘feature similarity’ to predict the values of new datapoints which further means that the new data point will be assigned a value based on how closely it matches the points in the training set. KNN can be used in loan default prediction to predict weather an individual is fit for the loan approval? Does that individual have the characteristics similar to the defaulters one? 
 
 
### XG Boost
XG boost i.e. extreme gradient boosting is one of the well-known gradient boosting techniques having enhanced performance and speed in tree-based (sequential decision trees) machine learning algorithms. XGBoost has in-built regularization which prevents the model from overfitting. XGBoost has an in-built capability to handle missing values. 


## Hyperparameter tunning 
A Machine Learning model is defined as a mathematical model with a number of parameters that need to be learned from the data. By training a model with existing data, we are able to fit the model parameters. 
However, there is another kind of parameters, known as Hyperparameters, that cannot be directly learned from the regular training process. They are usually fixed before the actual training process begins. 
This method use all the possible permutation and combination of the parameters. So, In short Hyperparameter tuning is choosing a set of optimal hyperparameters for a learning algorithm.

![image](https://user-images.githubusercontent.com/70087327/130553947-832f5cf2-5e0b-49d0-a119-61ff0e0e1f93.png)
![image](https://user-images.githubusercontent.com/70087327/130553961-c818bfae-00ee-48d2-8149-f28e411f7516.png)
![image](https://user-images.githubusercontent.com/70087327/130553972-0b18115a-0fee-4b7f-afa9-ba88cd344184.png)

## Results
We First Built a model using the whole data set and then after extracting all the important features. A model was built on the train data.
The model passed the global test meaning at least one variable significantly contributes to the model. Then by t-test we found the significant variables. The optimal Cut-off value was 0.56. The model also passed the Hosmer lemeshow goodness of fit test. Diagnostic Check involved checking the VIF values and we concluded that there was no presence of Multicollinearity.

In the confusion matrix, The Reference values were showed Row wise and the Predicted classes column wise. The matrix also showed distribution of true Positive, true negative, false positive and False negative. From the confusion matrix, we can say that Logistic Regression has the Highest proportion of False positives and Negatives followed by KNN, XGBoost and Random Forest.
The same can be seen from the Confusion Matrix of Test data.  Further, we calculated accuracy and Random forest had the highest accuracy among all of them.
The Accuracy in both the sets training and testing dataset is approximately the same within different models.

![image](https://user-images.githubusercontent.com/70087327/132232744-035d6f3e-8fe0-4ee8-8047-20301d3b7a42.png)

Logistic Regression shows poor performance with 64 percent. Based on the Recall values, XGBoost stands out to be the best model. Since the F1 Score for XGBoost and random Forest is the same, therefore we will look at the ROC Curve for all the models and will consider the model with highest AUC score

Here we see the AUC score for the XG Boost model is the highest with 91% which can be seen in the blue curve. It implies that the model correctly classifies the defaulters from the non-defaulters exactly 91% of the time based on the features followed by RandomForestClassifier with 89%, KNN with 82% and Logistic Regression with 71%. 

![image](https://user-images.githubusercontent.com/70087327/132230882-c086d96e-11cf-431f-a8da-d24011b1bc9d.png)

## SHAP
Next we took a glance at the Important Features from the XGBosst model using the SHAP values and we found that interest rate, DTI, annual Income term, Grade and various Accounts are the important features which also was our initial Guess.

![image](https://user-images.githubusercontent.com/70087327/132230824-9a336d53-6b0d-4855-832b-641aec4bbf7a.png)

SHAP which stands for Shapley Additive explanations, in a nutshell are used whenever we have a complex model, in our case XGBoost, and we want to understand what decisions the model is making.
It quantifies the contribution that each feature brings to the prediction made by the model and is interpreted as follows :
Blue indicates lower values of the feature whereas red indicates higher values.
On the horizonal axis, the values to the left of 0 indicate negative impact on the customer defaulting and the values to the right indicate positive impact on the customer defaulting. 

![image](https://user-images.githubusercontent.com/70087327/132230847-efca118b-7590-4a5a-b24e-f3bc0823df9b.png)

For example, the bar for debt-to-income ratio or dti shows that for lower values of dti, the customer is less likely to default (which is indicated by the blue colour on the left) and for higher values of dti,  the customer is more likely to default (which is indicated by the red colour on the right).
Similarly, the bar for Annual income shows that for higher values of annual income, the customer is less likely to default and for lower values of annual income, the customer is more likely to default.

## Conclusion
The Conclusions drawn based on the findings of our analysis are that Interest rate, Debt-to-income ratio, annual income of the applicant, term of the loan, loan amount, grade allocated to the applicant and the number of accounts were the top driving factors in loan defaulting.

In the problem of loan defaulting, false negatives have a higher cost than false positives since if a defaulter is predicted as a non-defaulter the consequences are much more serious. Hence, we give preference to recall value.
Consequently, XGBoost among other techniques employed  gives the best
fit for predicting loan defaulters with an accuracy of 81.24%, AUC score of 0.91 and a recall value of 80%.

