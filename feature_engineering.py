import pandas as pd
import numpy as np


def name(full):
    
    for dataset in full:
        dataset['NameLen'] = dataset['Name'].apply(lambda x: len(x))
        dataset['NameTitle'] = dataset['Name'].apply(lambda x: x.split(',')[1]).apply(lambda x: x.split()[0])
        
        dataset['NameTitle'] = dataset['NameTitle'].replace(['Lady.', 
                                                             'Countess.',
                                                             'Capt.', 
                                                             'Col.', 
                                                             'Don.', 
                                                             'Dr.', 
                                                             'Major.', 
                                                             'Rev.', 
                                                             'Sir.', 
                                                             'Jonkheer.',
                                                             'Dona.']
                                                            ,'Rare')
        # Error correction
        dataset['NameTitle'] = dataset['NameTitle'].replace('Mlle', 'Miss')
        dataset['NameTitle'] = dataset['NameTitle'].replace('Ms', 'Miss')
        dataset['NameTitle'] = dataset['NameTitle'].replace('Mme', 'Mrs')
        
        dataset.drop(['Name'], axis=1, inplace=True)
    
    return full


def impute_age(full):
    train = full[0]
    
    for dataset in full:
        # Keep a flag to signal if age was initially NaN
        dataset['AgeNull'] = dataset['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
        
        # Add mean of (title, pclass) age to impute missing values
        dataset['Age'] = train.groupby(['NameTitle', 'Pclass'])['Age'].transform(lambda x: x.fillna(int(x.mean())))
        
    
    return full


def impute_fare(full):
    train = full[0]
    
    for dataset in full:
        
        # Impute with training mean
        dataset['Fare'] = dataset['Fare'].fillna(train['Fare'].mean())
        
    return full

def impute_embarked(full):
    train = full[0]
    
    for dataset in full:
        dataset['Embarked'] = dataset['Embarked'].fillna(train['Embarked'].mode()[0])
    
    return full


def set_dummies(train, test, columns = ['Pclass', 'Sex', 'Embarked', 'NameTitle']):
    
    for column in columns:
        # Convert to ´string´
        train[column] = train[column].apply(lambda x: str(x))
        test[column] = test[column].apply(lambda x: str(x))
        
        # Check for columns in both sets
        good_cols = [column + '_' + i for i in train[column].unique() if i in test[column].unique()]
        
        train = pd.concat([train, pd.get_dummies(train[column], prefix = column)[good_cols]], axis = 1)
        test = pd.concat([test, pd.get_dummies(test[column], prefix = column)[good_cols]], axis = 1)
        
        # Drop old columns
        train.drop([column], axis=1, inplace=True)
        test.drop([column], axis=1, inplace=True)
    
    return train, test

