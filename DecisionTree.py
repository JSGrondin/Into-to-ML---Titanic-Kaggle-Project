# Load relevant packages and modules
import pandas as pd
from sklearn import tree
import numpy as np

# Load cleaned and formatted train and test data files (see Clean_and_format.py)
train = pd.read_csv('train_cleaned.csv')
test = pd.read_csv('test_cleaned.csv')

# Create the target and features numpy arrays
Y_train = train['Survived'].values
X_train = train[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_0', 'Embarked_1', 'FamilySize', 'IsAlone', 'Title']].values
X_test = test[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked_0', 'Embarked_1', 'FamilySize', 'IsAlone', 'Title']].values

# Fit decision tree
my_tree = tree.DecisionTreeClassifier(max_depth=4, min_samples_split=5, random_state=1)
my_tree = my_tree.fit(X_train, Y_train)

# Print feature importance and fitting score on training sample
print(my_tree.feature_importances_)
print(my_tree.score(X_train, Y_train))

# Predict targets on test sample
my_prediction = my_tree.predict(X_test)

# Create a data frame with two columns: PassengerId & Survived. Survived contains my predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

# Write Solution to a csv file with the name my_solution.csv
my_solution.to_csv('my_solution.csv', index_label=['PassengerId'])

