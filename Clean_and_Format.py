# Load relevant packages
import pandas as pd

#Disable warning from pandas for chained assignment
pd.options.mode.chained_assignment = None

# Load data in Dataframe with Pandas
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#--- Cleaning and Formatting Data
#Convert Sex categorical values to numerical value
df_train['Sex'][df_train['Sex'] == 'male'] = 0
df_train['Sex'][df_train['Sex'] == 'female'] = 1
df_test['Sex'][df_test['Sex'] == 'male'] = 0
df_test['Sex'][df_test['Sex'] == 'female'] = 1

#Impute missing Age variables
df_train['Age'] = df_train['Age'].fillna(df_train['Age'].median())
df_test['Age'] = df_test['Age'].fillna(df_test['Age'].median())

#Impute missing Embarked variables
df_train['Embarked'] = df_train['Embarked'].fillna('S')
df_test['Embarked'] = df_train['Embarked'].fillna('S')

#Convert Embarked categorical values to numerical values
df_train['Embarked'][df_train['Embarked'] == 'S'] = 0
df_train['Embarked'][df_train['Embarked'] == 'C'] = 1
df_train['Embarked'][df_train['Embarked'] == 'Q'] = 2
df_test['Embarked'][df_test['Embarked'] == 'S'] = 0
df_test['Embarked'][df_test['Embarked'] == 'C'] = 1
df_test['Embarked'][df_test['Embarked'] == 'Q'] = 2

df_train.to_csv('train_cleaned.csv')
df_test.to_csv('test_cleaned.csv')