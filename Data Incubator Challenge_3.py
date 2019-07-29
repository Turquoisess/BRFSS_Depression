# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 14:42:47 2019

@author: Siyu Zhang
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


big_frame = pd.read_sas("E:\LLCP2017.XPT")

selected_features = big_frame[['_AGEG5YR',
     'SEX',
     'EDUCA',
     'VETERAN3',
     'FRUIT2',
     'FRUITJU2',
     'FVGREEN1',
     'VEGETAB2',
     'EXEROFT1',
     'INCOME2',
     'MARITAL',
     '_BMI5',
     'SMOKE100',
     'SMOKDAY2',
     '_RFDRHV5',
     '_RFCHOL1',
     '_RFHYPE5',
     'ADDEPEV2']]
processed_features = selected_features.copy()

# check missing values in train dataset
total = processed_features.isnull().sum().sort_values(ascending=False)
percent = (processed_features.isnull().sum()/processed_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

#Age and Sex has no NAs,
#Education, 6 NAs as 'Not asked or missing', could be imputed as 'Refused', 9
processed_features['EDUCA'].fillna(9, inplace=True)
#narrow the education to 4 categories, value 1 for not graduate High School or Unknown, 2 for graduateed High School, 3 for attended college or technical school, 4 for graduated from college or technical school. 
processed_features.loc[processed_features['EDUCA'] <= 3, 'Education'] = 1
processed_features.loc[processed_features['EDUCA'] == 4, 'Education'] = 2
processed_features.loc[processed_features['EDUCA'] == 5, 'Education'] = 3
processed_features.loc[processed_features['EDUCA'] == 6, 'Education'] = 4
processed_features.loc[processed_features['EDUCA'] == 9, 'Education'] = 1
#Veteran, 11 NAs as 'Not asked or missing', imputed as 'Don't know/Not Sure', 7
processed_features['VETERAN3'].fillna(7, inplace=True)
#Fruit, 25432 NAs as 'Not asked or missing', imputed as 'Don't know/Not Sure', 777
processed_features['FRUIT2'].fillna(777, inplace=True)
#Fruit Juice, 26571 NAs as 'Not asked or missing', imputed as 'Don't know/Not Sure', 777
processed_features['FRUITJU2'].fillna(777, inplace=True)
#Vegetables, 27520 NAs as 'Not asked or missing', imputed as 'Don't know/Not Sure', 777
processed_features['FVGREEN1'].fillna(777, inplace=True)
#Other vegetables, 30293 NAs as 'Not asked or missing', imputed as 'Don't know/Not Sure', 777
processed_features['VEGETAB2'].fillna(777, inplace=True)
#Diet, narrow to 2 categories, value 1 for every week, 2 for more than a week or never or unknown
processed_features.loc[(processed_features['FRUIT2'] < 300) | (processed_features['FRUITJU2'] < 300) | (processed_features['FVGREEN1'] < 300) | (processed_features['VEGETAB2'] < 300) , 'Diet'] = 1
processed_features['Diet'].fillna(2, inplace=True)
#Exercise, narrow to 3 categories, value 1 for every week, 2 for every month, 3 for never or unknown
processed_features.loc[(processed_features['EXEROFT1'] > 100) & (processed_features['EXEROFT1'] < 200), 'Exercise'] = 1
processed_features.loc[(processed_features['EXEROFT1'] > 200) & (processed_features['EXEROFT1'] < 300), 'Exercise'] = 2
processed_features['Exercise'].fillna(3, inplace=True)
#Income, narrow to 4 categories, value 1 for less than $50,000, value 2 for $50,000 - $75,000, value 3 for greater than $75,000, value 4 for unknown
processed_features.loc[processed_features['INCOME2'] <= 6, 'Income'] = 1
processed_features.loc[processed_features['INCOME2'] == 7, 'Income'] = 2
processed_features.loc[processed_features['INCOME2'] == 8, 'Income'] = 3
processed_features['Income'].fillna(4, inplace=True)
#Marital, narrow to 2 categories, value 1 for married, value 2 for all other situations
processed_features.loc[processed_features['MARITAL'] == 1, 'Married'] = 1
processed_features['Married'].fillna(2, inplace=True)
#BMI, 36446 NAs, imputed with mean, narrow to 4 categories, value 1 for underweight, value 2 for normal weight, value 3 for overweight, value 4 for obese
processed_features['_BMI5'].fillna(processed_features['_BMI5'].mean(), inplace=True)
processed_features.loc[(processed_features['_BMI5'] < 1850), 'Build'] = 1
processed_features.loc[(processed_features['_BMI5'] >= 1850) & (processed_features['_BMI5'] < 2500), 'Build'] = 2
processed_features.loc[(processed_features['_BMI5'] >= 2500) & (processed_features['_BMI5'] < 3000), 'Build'] = 3
processed_features['Build'].fillna(4, inplace=True)
#Smoke, narrow to 4 categories, value 1 for Everyday smoker, value 2 for Someday smoker, value 3 for Former smoker or unknown, value 4 for Non-smoker
processed_features.loc[(processed_features['SMOKE100'] == 1) & (processed_features['SMOKDAY2'] == 1), 'Smoke'] = 1
processed_features.loc[(processed_features['SMOKE100'] == 1) & (processed_features['SMOKDAY2'] == 2), 'Smoke'] = 2
processed_features.loc[(processed_features['SMOKE100'] == 1) & (processed_features['SMOKDAY2'] == 3), 'Smoke'] = 3
processed_features.loc[(processed_features['SMOKE100'] == 2), 'Smoke'] = 4
processed_features['Smoke'].fillna(3, inplace=True)
#Alcohol, blood pressure has no NAs, cholesterol has 51571 NAs as 'Missing', imputed as 'No High Cholesterol'
processed_features['_RFCHOL1'].fillna(1, inplace=True)
#Depression, narrow to 2 categories, value 1 for yes, value 2 for all other situation
processed_features.loc[(processed_features['ADDEPEV2'] == 1), 'Depression'] = 1
processed_features['Depression'].fillna(0, inplace=True)


#Remove the raw data columns
processed_features = processed_features[['SEX',
     'Education',
     'VETERAN3',
     'Diet',
     'Exercise',
     'Income',
     'Married',
     'Build',
     'Smoke',
     '_RFDRHV5',
     '_RFCHOL1',
     '_RFHYPE5',
     'Depression']]
#check correlation
corrmat = processed_features.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

#Check missing values
total = processed_features.isnull().sum().sort_values(ascending=False)
percent = (processed_features.isnull().sum()/processed_features.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()

logreg = LogisticRegression()
scores_reg = []
train, test = train_test_split(processed_features, test_size=0.1)
X_train=train.loc[:,'SEX':'_RFHYPE5']
y_train=train['Depression']
X_test = test.loc[:,'SEX':'_RFHYPE5']
y_test = test['Depression']
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
scores_reg.append(logreg.score(X_test, y_test))
print('Accuracy of Logistic Regression on test set: {:.4f}'.format(scores_reg))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()

print(logreg.classes_)







