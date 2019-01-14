# Load relevant packages
import pandas as pd

#Disable warning from pandas for chained assignment
pd.options.mode.chained_assignment = None

# Load data in Dataframe with Pandas
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#--- Cleaning and Formatting Data
#Convert Sex categorical values to numerical value
train['Sex'] = train['Sex'].map( {'male': 0, 'female': 1} ).astype(int)
test['Sex'] = test['Sex'].map( {'male': 0, 'female': 1} ).astype(int)

#Impute missing Age variables (train sample)
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())

#Impute missing Embarked variables (train sample)
train['Embarked'] = train['Embarked'].fillna('S')

#Impute missing Fare variable (test sample)
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

#Convert Embarked categorical values to numerical values
train['Embarked'] = train['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
test['Embarked'] = test['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#identify titles in Names and convert to categorical values
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train['Title'] = train['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

train['Title'] = train['Title'].replace('Mlle', 'Miss')
train['Title'] = train['Title'].replace('Ms', 'Miss')
train['Title'] = train['Title'].replace('Mme', 'Mrs')
test['Title'] = test['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

test['Title'] = test['Title'].replace('Mlle', 'Miss')
test['Title'] = test['Title'].replace('Ms', 'Miss')
test['Title'] = test['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
train['Title'] = train['Title'].map(title_mapping)
train['Title'] = train['Title'].fillna(0)
test['Title'] = test['Title'].map(title_mapping)
test['Title'] = test['Title'].fillna(0)

#droping features: Name and PassengerId
train = train.drop(['Name', 'PassengerId'], axis=1)
test = test.drop(['Name'], axis=1)

#creating feature: FamilySize and IsAlone
train['FamilySize'] = train['SibSp'] + train['Parch'] + 1
test['FamilySize'] = test['SibSp'] + test['Parch'] + 1
train['IsAlone'] = 0
train.loc[train['FamilySize'] == 1, 'IsAlone'] = 1
test['IsAlone'] = 0
test.loc[train['FamilySize'] == 1, 'IsAlone'] = 1

#Save cleaned files to .csv
train.to_csv('train_cleaned.csv')
test.to_csv('test_cleaned.csv')