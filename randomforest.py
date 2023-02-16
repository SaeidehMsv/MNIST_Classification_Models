
import pandas as pd
from sklearn.preprocessing import scale
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
import numpy as np



train_data = pd.read_csv("/home/saeideh/Desktop/mnist/mnist_train.csv") 
test_data = pd.read_csv("/home/saeideh/Desktop/mnist/mnist_test.csv")


y_train = train_data['label']
X_train= train_data.drop(columns = 'label')
X_train = X_train/255.0
X_train = scale(X_train)


y_test = test_data['label']
X_test= test_data.drop(columns = 'label')
X_test= X_test/255.0
X_test = scale(X_test)



#random forest
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)



rf.fit(X_train, y_train);
y_pred = rf.predict(X_test)
errors = abs(y_pred - y_test)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')
mape = 100 * (errors / y_test)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')



