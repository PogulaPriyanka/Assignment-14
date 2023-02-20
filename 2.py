import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import classification_report
from sklearn import preprocessing
df = pd.read_csv("Fraud_check.csv")
df.head()
df.tail()
df=pd.get_dummies(df,columns=['Undergrad','Marital.Status','Urban'], drop_first=True)
df["TaxInc"] = pd.cut(df["Taxable.Income"], bins = [10002,30000,99620], labels = ["Risky", "Good"])
print(df)
df = pd.get_dummies(df,columns = ["TaxInc"],drop_first=True)
df.tail(10)
import seaborn as sns
sns.pairplot(data=df, hue = 'TaxInc_Good')
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return (x)
df_norm = norm_func(df.iloc[:,1:])
df_norm.tail(10)
X = df_norm.drop(['TaxInc_Good'], axis=1)
y = df_norm['TaxInc_Good']
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
df_norm["income"]="<=30000"
df_norm.loc[df["Taxable.Income"]>=30000,"income"]="Good"
df_norm.loc[df["Taxable.Income"]<=30000,"income"]="Risky"
df.drop(["Taxable.Income"],axis=1,inplace=True)
df.rename(columns={"Undergrad":"undergrad","Marital.Status":"marital","City.Population":"population","Work.Experience":"experience","Urban":"urban"},inplace=True)
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
for column_name in df.columns:
    if df[column_name].dtype == object:
        df[column_name] = le.fit_transform(df[column_name])
    else:
        pass
features = df.iloc[:,0:5]
labels = df.iloc[:,5]
colnames = list(df.columns)
predictors = colnames[0:5]
target = colnames[5]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(features,labels,test_size = 0.2,stratify = labels)
from sklearn.ensemble import RandomForestClassifier as RF
model = RF(n_jobs = 3,n_estimators = 15, oob_score = True, criterion = "entropy")
model.fit(x_train,y_train)
model.estimators_
model.classes_
model.n_features_
model.n_classes_
model.n_outputs_
model.oob_score_
prediction = model.predict(x_train)
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_train,prediction)
np.mean(prediction == y_train)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_train,prediction)
pred_test = model.predict(x_test)
acc_test =accuracy_score(y_test,pred_test)
from sklearn.tree import export_graphviz
import pydotplus
from six import StringIO
tree = model.estimators_[5]
dot_data = StringIO()
export_graphviz(tree,out_file = dot_data, filled = True,rounded = True, feature_names = predictors ,class_names = target,impurity =False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
model = DecisionTreeClassifier(criterion = 'entropy',max_depth=3)
model.fit(x_train,y_train)
DecisionTreeClassifier(criterion='entropy', max_depth=3)
from sklearn import tree
tree.plot_tree(model);
colnames = list(df.columns)
colnames
fn=['population','experience','Undergrad_YES','Marital.Status_Married','Marital.Status_Single','Urban_YES']
cn=['1', '0']
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
