# Load relevant packages and modules
import pandas as pd
from sklearn.svm import LinearSVC
import numpy as np

# Load cleaned and formatted train and test data files (see Clean_and_format.py)
train = pd.read_csv('train_cleaned.csv')
test = pd.read_csv('test_cleaned.csv')

# Create the target and features numpy arrays
features = ['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_0', 'Embarked_1', 'FamilySize', 'IsAlone', 'Title']
Y_train = train['Survived'].values
X_train = train[features].values
X_test = test[features].values

# Fit linear SVC model
linear_svc = LinearSVC()
linear_svc.fit(X_train, Y_train)
Y_pred = linear_svc.predict(X_test)
acc_linear_svc = round(linear_svc.score(X_train, Y_train) * 100, 2)
print(acc_linear_svc)

# Predict targets on test sample
my_prediction = linear_svc.predict(X_test)

# Create a data frame with two columns: PassengerId & Survived. Survived contains my predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

# Write Solution to a csv file with the name my_solution.csv
my_solution.to_csv('my_solution.csv', index_label=['PassengerId'])

