# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/bd063086-2aca-4445-a3a5-d0d06176ffad)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/dd6c04a2-220a-4d7d-bc0c-19f14d8559d5)
```
missing=data[data.isnull().any(axis=1)]
missing
```
![image](https://github.com/user-attachments/assets/ba879987-f5b9-4d63-b7f6-e2336686acf7)
```
data2=data.dropna(axis=0)
data2
```
![image](https://github.com/user-attachments/assets/273495bd-2989-410b-9ea8-ca77e6e495ef)
```
sal=data["SalStat"]

data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```
![image](https://github.com/user-attachments/assets/480ee6eb-15b2-4c13-b8cc-a62085fe6fa9)
```
sal2=data2['SalStat']

dfs=pd.concat([sal,sal2],axis=1)
dfs
```
![image](https://github.com/user-attachments/assets/f85e0bf5-7d19-49ee-bfc7-4ca2231becc9)

```
data2
```
![image](https://github.com/user-attachments/assets/5087dea9-a540-464f-8aa9-76f551d15e3f)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/e701f493-9b77-4a84-b2a7-64364921bba5)
![image](https://github.com/user-attachments/assets/da7bb69a-ba0f-42da-9e23-46db363e2154)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![image](https://github.com/user-attachments/assets/341292d9-3f84-45ca-b20c-44d4e6032115)

```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![image](https://github.com/user-attachments/assets/bb6a198f-1c87-47c4-b2b4-09813f018894)
```
y=new_data['SalStat'].values
print(y)
```
![image](https://github.com/user-attachments/assets/2b83a47f-014d-400d-b948-d9b0078a070c)
```
x=new_data[features].values
print(x)
```
![image](https://github.com/user-attachments/assets/8cf58ab8-5559-4580-b8b4-89a6d57a2dbd)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)

KNN_classifier=KNeighborsClassifier(n_neighbors = 5)

KNN_classifier.fit(train_x,train_y)
```
![image](https://github.com/user-attachments/assets/7ebefe88-a0a9-4a2d-a2ad-37538680deef)

```
prediction=KNN_classifier.predict(test_x)

confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/bf9870ec-2fb9-40a7-ae09-641ae602d90a)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```
![image](https://github.com/user-attachments/assets/39276eb5-d7ea-447f-b095-869f819d4121)
```
data.shape
```
![image](https://github.com/user-attachments/assets/9f201aa2-d1aa-435e-b1dc-666c57247763)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]

selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)

selected_feature_indices=selector.get_support(indices=True)

selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```
![image](https://github.com/user-attachments/assets/883d8e20-d8a1-42cc-a860-e2cc5badccf7)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```
![image](https://github.com/user-attachments/assets/4d1b7d55-9c5a-4e03-bef2-b034ae5ec60d)
```
tips.time.unique()
```
![image](https://github.com/user-attachments/assets/f870dca5-4317-42e4-9f7e-23c1ede97516)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![image](https://github.com/user-attachments/assets/c903d1b2-3fbb-4612-99d0-1c670a0beb65)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```
![image](https://github.com/user-attachments/assets/11529633-69cb-4607-b5ba-8c4b2c1e84b1)

# RESULT:
Thus the given data and perform Feature Scaling and Feature Selection process and save the
data to a file is Verified 
