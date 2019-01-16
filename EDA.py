import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data in Dataframe with Pandas
#train = pd.read_csv('train.csv')
#test = pd.read_csv('test.csv')

# Load cleaned and formatted train and test data files (see Clean_and_format.py)
train = pd.read_csv('train_cleaned.csv')
test = pd.read_csv('test_cleaned.csv')

# Analyze by pivoting features - Pclass vs Survived
train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Analyze by pivoting features - Sex vs Survived
train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Analyze by pivoting features - Embarked vs Survived
#train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Analyze by pivoting features - Sex vs Survived
train_0=train[train['Embarked']==0]
train_1=train[train['Embarked']==1]
train_2=train[train['Embarked']==2]
train_0[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_1[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_2[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

# Analyze by visualizing data - Age vs Survived
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=20)

# Analyze by visualizing data - Pclass vs Survived
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();

# Correlating categorical features - Embarked, Pclass and Sex
grid = sns.FacetGrid(train, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep', hue_order = [1, 0])
grid.add_legend()

# Correlating categorical features with numerical features - Embarked, Survived and Sex
grid = sns.FacetGrid(train, row='Embarked', col='Survived', size=2.2, aspect=1.6)
grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)
grid.add_legend()