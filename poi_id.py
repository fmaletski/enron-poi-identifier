#!/usr/bin/python
import sys
import pickle
import pandas as pd
import numpy as np
sys.path.append("./tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi',
                 'exercised_stock_options',
                 'bonus',
                 'to_poi_ratio',
                 'expenses',
                 'from_poi_ratio',
                 'deferral_payments']

### Load the dictionary containing the dataset

with open("final_project_dataset_py3.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Remove outliers

data_dict.pop('TOTAL')

### Create dataframe

df = pd.DataFrame(data_dict)
df = df.transpose()
df.replace('NaN', 0, inplace=True)
### Create new features

df['from_poi_ratio'] = df['from_poi_to_this_person']/df['to_messages']
df['to_poi_ratio'] = df['from_this_person_to_poi']/df['from_messages']
df = df.fillna(0)

### Feature scaling

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[features_list] = scaler.fit_transform(df[features_list])

labels = df.poi
features = df[features_list[1:]]
### Store in my_dataset to export below

my_dataset = df[features_list].transpose().to_dict()

### Classifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier

clf1 = DecisionTreeClassifier(class_weight='balanced', criterion='entropy',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=19, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=42,
            splitter='best')

clf2 = SVC(C=5.4, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=3.9, kernel='sigmoid',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=0.001, verbose=False)

clf3 = KNeighborsClassifier(algorithm='ball_tree', leaf_size=1, metric='minkowski',
           metric_params=None, n_jobs=1, n_neighbors=3, p=2,
           weights='distance')

eclf = VotingClassifier(estimators=[('dt', clf1), ('svc', clf2), ('kn', clf3)], voting='hard')

dump_classifier_and_data(clf1, my_dataset, features_list)
