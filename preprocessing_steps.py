#null values if any
labels = []
values = []
for col in train.columns:
    labels.append(col)
    values.append(train[col].isnull().sum())
    print(col, values[-1])

#finding all unique values 
for col in train.columns:
    print(col,train[col].nunique())
    
    
    
    
'''
train=train.drop(['vidid'],axis=1)

from datetime import datetime

train['published']=train['published'].map(str)
train['published']=pd.to_datetime(train['published'],format='%d/%m/%Y')

train['Day of week']=train['published'].dt.dayofweek
train['Week of year']=train['published'].dt.weekofyear

##Converting comma values to float
cols=list(train.columns.values)

train=train.loc[2:,:]

train[cols[2]]=train[cols[2]].astype(int)

#converting to int
train[cols[3]]=train[cols[3]].str.replace('nan','-1').astype(str)


train[cols[3]]=train[cols[3]].astype(int)


train[cols[5]]=train[cols[5]].str.replace('F','-1').astype(str)


train[cols[5]]=train[cols[5]].astype(int)


train[cols[3]]=train[cols[3]].astype(int)
train[cols[4]]=train[cols[4]].astype(int)
train[cols[5]]=train[cols[5]].astype(int)

'''
    
    
    
    