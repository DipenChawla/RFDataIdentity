import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
dataset = pd.read_csv('train_HK6lq50.csv')
df1 = pd.read_csv('test_2nAIblo.csv')

dataset['age'].fillna(dataset['age'].median(), inplace=True)
dataset['trainee_engagement_rating'].fillna(dataset['trainee_engagement_rating'].mode()[0], inplace=True)
dataset.isnull().sum()
#Pre-processing the test data
df1['age'].fillna(df1['age'].median(), inplace=True)
df1['trainee_engagement_rating'].fillna(df1['trainee_engagement_rating'].mode()[0], inplace=True)
df1.isnull().sum()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
for col in dataset.columns.values:
       # Encoding only categorical variables
       if dataset[col].dtypes=='object':
       # Using whole data to form an exhaustive list of levels
           data = dataset[col]
           labelencoder.fit(data.values)
           dataset[col]= labelencoder.transform(dataset[col])
X = dataset.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13,14]].values
onehotencoder = OneHotEncoder(categorical_features = [0,3,4,6,7,8,11])
X = onehotencoder.fit_transform(X).toarray()

labelencoder1 = LabelEncoder()
for col in df1.columns.values:
       # Encoding only categorical variables
       if df1[col].dtypes=='object':
       # Using whole data to form an exhaustive list of levels
           data = df1[col]
           labelencoder1.fit(data.values)
           df1[col]= labelencoder1.transform(df1[col])
X_df = df1.iloc[:,[2,3,4,5,6,7,8,9,10,11,12,13,14]].values
onehotencoder1 = OneHotEncoder(categorical_features = [0,3,4,6,7,8,11])
X_df = onehotencoder1.fit_transform(X_df).toarray()
y = dataset.iloc[:, 15].values
#
## Splitting the dataset into the Training set and Test set
#from sklearn.cross_validation import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
i=1
kf = StratifiedKFold(n_splits=5,random_state=42,shuffle=True)
for train_index,test_index in kf.split(X,y):
    print('\n{} of kfold {}'.format(i,kf.n_splits))
    xtr,xvl = X[train_index],X[test_index]
    ytr,yvl = y[train_index],y[test_index]
    model = RandomForestRegressor(n_estimators = 1000, max_depth = 30, min_samples_leaf = 4, random_state = 42, verbose = 1)
    model.fit(xtr,ytr)
    pred = model.predict(xvl)
    score = roc_auc_score(yvl,pred)
    print('roc_auc_score',score)
    i+=1
y_pred = model.predict(X_df)

submission=pd.read_csv("final3.csv")
submission['is_pass']=y_pred
submission.to_csv('dipenresult.csv', index=False)
