# Load relevant packages
import pandas as pd

#Disable warning from pandas for chained assignment
pd.options.mode.chained_assignment = None

# Load data in Dataframe with Pandas
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

#--- Cleaning and Formatting Data
#Convert Sex categorical values to numerical value
train['Sex'][train['Sex'] == 'male'] = 0
train['Sex'][train['Sex'] == 'female'] = 1
test['Sex'][test['Sex'] == 'male'] = 0
test['Sex'][test['Sex'] == 'female'] = 1

#Impute missing Age variables (train sample)
train['Age'] = train['Age'].fillna(train['Age'].median())
test['Age'] = test['Age'].fillna(test['Age'].median())

#Impute missing Embarked variables (train sample)
train['Embarked'] = train['Embarked'].fillna('S')

#Impute missing Fare variable (test sample)
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

#Convert Embarked categorical values to numerical values
train['Embarked'][train['Embarked'] == 'S'] = 0
train['Embarked'][train['Embarked'] == 'C'] = 1
train['Embarked'][train['Embarked'] == 'Q'] = 2
test['Embarked'][test['Embarked'] == 'S'] = 0
test['Embarked'][test['Embarked'] == 'C'] = 1
test['Embarked'][test['Embarked'] == 'Q'] = 2

train.to_csv('train_cleaned.csv')
test.to_csv('test_cleaned.csv')