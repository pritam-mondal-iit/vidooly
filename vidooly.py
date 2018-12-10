# pritam will get placed by this!

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train=pd.read_csv('final_train.csv')

#cols=['views','likes','dislikes','comment']
X=train.iloc[:,[1,5,6,7,8,9,10,11,12]].values
Y=train.iloc[:,-1].values

from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)


import xgboost
xgb = xgboost.XGBRegressor(n_estimators=1000, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb.fit(X_train,y_train)
predictions1 = xgb.predict(X_test)

#Finding RMSE
from sklearn.metrics import mean_squared_error
import math
testScore1=math.sqrt(mean_squared_error(y_test,predictions1))
print(testScore1)

#10-fold Cross Validation for counteract overfitting
from sklearn.model_selection import cross_val_score
accuracies1=cross_val_score(estimator=xgb, X=X_train, y=Y, cv=10)
accuracies1.mean()
accuracies1.std()



from sklearn.preprocessing import RobustScaler
transformer = RobustScaler().fit(X)
X_train1 = transformer.fit_transform(X)
X_train1 = transformer.transform(X_train1)


#correlations matrix

cols= list(train.columns.values)

labels = []
values = []
for col in cols:
    labels.append(col)
    values.append(np.corrcoef(train[col].values, train['duration'].values)[0,1])
    print(col,values[-1])
ind = np.arange(len(labels))
width = 0.9
fig, ax = plt.subplots(figsize=(20,40))
rects = ax.barh(ind, np.array(values), color='y')
ax.set_yticks(ind+((width)/2.))
ax.set_yticklabels(labels, rotation='horizontal')
ax.set_xlabel("Correlation coefficient")
ax.set_title("Correlation coefficient")
plt.show()


import seaborn as sns

cols_to_use = ['killRank', 'greekFireItems', 'onFootDistance', 'weaponsUsed']

temp_df = train[cols]
corrmat = temp_df.corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()






'''
#Data Preprocessing steps
#Standardizing values
from sklearn.preprocessing import MinMaxScaler
minmax1 = MinMaxScaler()
X_train1 = minmax1.fit_transform(X)
X_train1 = minmax1.transform(X_train1)
'''
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train1, Y, test_size = 0.25, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#Finding RMSE for RandomForest Model 
from sklearn.metrics import mean_squared_error
import math
testScore2=math.sqrt(mean_squared_error(y_test,predictions2))
print(testScore2)

from sklearn.metrics import r2_score
r2_score(y_test, y_pred) 



import xgboost
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb.fit(X_train,y_train)
predictions1 = xgb.predict(X_test)

#Finding RMSE
from sklearn.metrics import mean_squared_error
import math
testScore1=math.sqrt(mean_squared_error(y_test,predictions1))
print(testScore1)

#10-fold Cross Validation for counteract overfitting
from sklearn.model_selection import cross_val_score
accuracies1=cross_val_score(estimator=xgb, X=X_train, y=Y, cv=10)
accuracies1.mean()
accuracies1.std()





'''
cols=['likes','dislikes','comments']
#for col in cols:
train['duration'].loc[train['duration'] < 10] = np.nan

train2=train.dropna()

train2.to_csv('final_train.csv')
'''




'''
l=[]

for i in train2['duration']:
    a = i.lstrip('PT').rstrip('S').replace('H','.').replace('M','.').split('.')
    try:    
        if len(a)==3:
            time = int(a[0])*60*60 + int(a[1])*60 + int(a[2])
        elif len(a)==2:
            time = int(a[0])*60 + int(a[1])
        else:
            time = int(a[0])
    except:
        time = int(a[0])*60
    l.append(time)


n = train2.columns[6]

train2.drop(n,axis=1,inplace=True)

train2[n] = l
'''

'''
s = pd.Series(train2['category'])
pd.get_dummies(s)

train3=pd.get_dummies(train2, columns=["category"])
cols=list(train3.columns.values)

train2.to_csv('data_train1.csv')

train3=train3.drop(['Week of year'],axis=1)
train3.to_csv('train_set1.csv',index=False)
'''
















