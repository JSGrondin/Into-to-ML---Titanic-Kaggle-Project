# Load relevant packages and modules
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Load cleaned and formatted train and test data files (see Clean_and_format.py)
train = pd.read_csv('train_cleaned.csv')
test = pd.read_csv('test_cleaned.csv')

# Create the target and features numpy arrays
train_target = train['Survived'].values
train_features = train[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch']].values
test_features = test[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch']].values

# Fit decision tree
my_forest = RandomForestClassifier(max_depth=5, min_samples_split=2, n_estimators=200, random_state=1)
my_forest = my_forest.fit(train_features, train_target)

# Print fitting score on training sample
print(my_forest.score(train_features, train_target))

# Predict targets on test sample
my_prediction = my_forest.predict(test_features)

# Create a data frame with two columns: PassengerId & Survived. Survived contains my predictions
PassengerId =np.array(test["PassengerId"]).astype(int)
my_solution = pd.DataFrame(my_prediction, PassengerId, columns = ["Survived"])

# Write Solution to a csv file with the name my_solution.csv
my_solution.to_csv('my_solution.csv', index_label=['PassengerId'])

