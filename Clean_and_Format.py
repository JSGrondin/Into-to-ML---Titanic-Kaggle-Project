# Load relevant packages
import pandas as pd
import numpy as np

#Disable warning from pandas for chained assignment
pd.options.mode.chained_assignment = None

# Load data in Dataframe with Pandas
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
combine = [train, test]

#--- Cleaning and Formatting Data
#Convert Sex categorical values to numerical value
for dataset in combine:
        dataset['Sex'] = dataset['Sex'].map( {'male': 0, 'female': 1} ).astype(int)

#Impute missing Age variables (using median from different Sex/Pclass group combinations)
guess_ages = np.zeros((2,3))
for dataset in combine:
        for i in range(0, 2):   #Sex
            for j in range(0, 3):       #Pclass
                guess_df = dataset[(dataset['Sex']==i) & (dataset['Pclass'] == j+1)]['Age'].dropna()
                age_guess = guess_df.median()
                guess_ages[i,j] = int(age_guess/0.5+0.5)*0.5

        for i in range(0, 2):
            for j in range(0, 3):
                dataset.loc[(dataset.Age.isnull()) & (dataset.Sex == i) & (dataset.Pclass == j+1),\
                        'Age'] = guess_ages[i,j]

        dataset['Age'] = dataset['Age'].astype(int)

#Impute missing Embarked variables (train sample)
train['Embarked'] = train['Embarked'].fillna('S')

#Impute missing Fare variable (test sample)
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

#Convert Embarked categorical values to numerical values
for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#identify titles in Names and convert to categorical values
for dataset in combine:
        dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
                'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

#creating feature: FamilySize and IsAlone
for dataset in combine:
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['IsAlone'] = 0
        dataset.loc[train['FamilySize'] == 1, 'IsAlone'] = 1

#droping features: Name and PassengerId
train = train.drop(['Ticket', 'Cabin', 'Name', 'PassengerId', 'SibSp', 'Parch'], axis=1)
test = test.drop(['Ticket', 'Cabin', 'Name', 'SibSp', 'Parch'], axis=1)

#train.head(10)
#test.head(10)

#Save cleaned files to .csv
train.to_csv('train_cleaned.csv')
test.to_csv('test_cleaned.csv')