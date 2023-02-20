import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import plot_tree 
df = pd.read_csv('Company_Data.csv')
df.head()
df.info()
df.shape
df.isnull().any()
sns.pairplot(data=df, hue = 'ShelveLoc')
df=pd.get_dummies(df,columns=['Urban','US'], drop_first=True)
df
df.info()
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
df['ShelveLoc']=df['ShelveLoc'].map({'Good':1,'Medium':2,'Bad':3})
df.head()
x=df.iloc[:,0:6]
y=df['ShelveLoc']
x
y
df['ShelveLoc'].unique()
df.ShelveLoc.value_counts()
colnames = list(df.columns)
colnames
x_train, x_test,y_train,y_test = train_test_split(x,y, test_size=0.2,random_state=40)
model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)
from sklearn import tree
tree.plot_tree(model);
fn=['Sales','CompPrice','Income','Advertising','Population','Price']
cn=['1', '2', '3']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(model,
               feature_names = fn, 
               class_names=cn,
               filled = True);
preds = model.predict(x_test)
pd.Series(preds).value_counts()
preds
pd.crosstab(y_test,preds)
np.mean(preds==y_test)
from sklearn.tree import DecisionTreeClassifier
model_gini = DecisionTreeClassifier(criterion='gini', max_depth=3)
model_gini.fit(x_train, y_train)
pred=model.predict(x_test)
np.mean(preds==y_test)
from sklearn.tree import DecisionTreeRegressor
array = df.values
X = array[:,0:3]
y = array[:,3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)
model = DecisionTreeRegressor()
model.fit(X_train, y_train)
model.score(X_test,y_test)
