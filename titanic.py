import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import mean_absolute_error
import seaborn as sns

sc = StandardScaler()
logmodel = LogisticRegression()

train_data = pd.read_csv("datasets/train.csv")
test_data = pd.read_csv("datasets/test.csv")
gender = pd.read_csv("datasets/gender_submission.csv")

sns.countplot(x="Pclass", data=train_data)

train_data.drop(["Cabin"], axis=1, inplace=True)
m = train_data["Age"].mean()
train_data["Age"].fillna(m, inplace=True)
train_data["Embarked"].fillna(0, inplace=True)
sex = pd.get_dummies(train_data["Sex"], drop_first=True, dtype=np.uint8)
pcl = pd.get_dummies(train_data["Pclass"], drop_first=True, dtype=np.uint8)
embark = pd.get_dummies(
    train_data["Embarked"], drop_first=True, dtype=np.uint8)

train_data = pd.concat([train_data, sex, pcl, embark], axis=1)
train_data.drop(["PassengerId", "Pclass", "Name", "Sex",
                "Ticket", "Embarked", "C"], axis=1, inplace=True)


test_data.drop(["Cabin"], axis=1, inplace=True)
age_mean = test_data["Age"].mean()
test_data["Age"].fillna(age_mean, inplace=True)
test_data["Embarked"].fillna(0, inplace=True)
test_data["Fare"].fillna(0, inplace=True)
sex1 = pd.get_dummies(test_data["Sex"], drop_first=True, dtype=np.uint8)
pcl1 = pd.get_dummies(test_data["Pclass"], drop_first=True, dtype=np.uint8)
embark1 = pd.get_dummies(
    test_data["Embarked"], drop_first=True, dtype=np.uint8)

test_data = pd.concat([test_data, sex1, pcl1, embark1], axis=1)
test_data.drop(["PassengerId", "Pclass", "Name", "Sex",
               "Ticket", "Embarked"], axis=1, inplace=True)


X_train = train_data.drop(["Survived"], axis=1)
y_train = train_data["Survived"]
X_test = test_data
y_test = gender.drop(["PassengerId"], axis=1)
X_train.columns = X_train.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

logmodel.fit(X_train, y_train)

prediction = logmodel.predict(X_test)

classification_report(y_test, prediction)

matrix = confusion_matrix(y_test, prediction)
acc = accuracy_score(y_test, prediction)

print("Confusion Matrix:")
print(matrix)
print("Accuracy Score:")
print(f"{round(acc, 2)*100}%")

ln_mse = mean_absolute_error(y_test, prediction)
ln_rmse = np.sqrt(ln_mse)
print(ln_mse, ln_rmse)
