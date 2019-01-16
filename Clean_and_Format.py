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

#Creating Age Bands to study correlation with Survived
#train['AgeBand'] = pd.cut(train['Age'], 5)
#train[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)

#Replace Age with ordinals based on age bands
for dataset in combine:
    dataset.loc[dataset['Age'] <= 16, 'Age'] = int(0)
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = int(1)
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = int(2)
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = int(3)
    dataset.loc[ dataset['Age'] > 64, 'Age'] = int(4)

#Impute missing Embarked variables (train sample)
train['Embarked'] = train['Embarked'].fillna('S')

#Convert Embarked categorical values to numerical values
for dataset in combine:
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

#Create dummy variables to replace categorical feature Embarked,
train = pd.concat([train, pd.get_dummies(train['Embarked'], prefix='Embarked', prefix_sep='_')], axis=1)
test = pd.concat([test, pd.get_dummies(test['Embarked'], prefix='Embarked', prefix_sep='_')], axis=1)
combine = [train, test]

#Impute missing Fare variable (test sample)
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

#Create FareBand from Fare feature
for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
    dataset['Fare'] = dataset['Fare'].astype(int)

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
train = train.drop(['Ticket', 'Cabin', 'Name', 'PassengerId', 'SibSp', 'Parch', 'Embarked', 'Embarked_2'], axis=1)
test = test.drop(['Ticket', 'Cabin', 'Name', 'SibSp', 'Parch', 'Embarked', 'Embarked_2'], axis=1)

#train.head(10)
#test.head(10)

#Save cleaned files to .csv
train.to_csv('train_cleaned.csv')
test.to_csv('test_cleaned.csv')