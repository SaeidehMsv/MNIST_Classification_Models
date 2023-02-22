import pandas as pd
from sklearn.svm import SVC  
from sklearn.preprocessing import scale
import numpy as np



train_data = pd.read_csv("mnist_train.csv")
test_data = pd.read_csv("/mnist_test.csv")
#print(train_data.head())

y_train = train_data['label']
X_train= train_data.drop(columns = 'label')
X_train = X_train/255.0
X_train = scale(X_train)


y_test = test_data['label']
X_test= test_data.drop(columns = 'label')
X_test= X_test/255.0
X_test = scale(X_test)

#print(X_train[0]) 



#SVM
# linear model 
model_linear = SVC(kernel='linear')


model_linear.fit(X_train, y_train)
y_pred = model_linear.predict(X_test)
print("accuracy-linear:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")

# non-linear model 
non_linear_model = SVC(kernel='rbf')


non_linear_model.fit(X_train, y_train)
y_pred = non_linear_model.predict(X_test)
print("accuracy-non linear:", metrics.accuracy_score(y_true=y_test, y_pred=y_pred), "\n")









