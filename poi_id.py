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
                 'to_poi_ratio',
                 'salary',
                 'bonus',
                 'deferred_income',
                 'total_stock_value',
                 'exercised_stock_options']

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
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

clf3 = SVC(C=11.6, cache_size=200, class_weight='balanced', coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=1.17, kernel='sigmoid',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=0.001, verbose=False)

clf2 = AdaBoostClassifier(algorithm='SAMME',
          base_estimator=DecisionTreeClassifier(class_weight='balanced', criterion='gini',
            max_depth=None, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=12, min_samples_split=2,
            min_weight_fraction_leaf=0.0, presort=False, random_state=None,
            splitter='best'),
          learning_rate=1.2, n_estimators=3, random_state=42)

clf1 = RandomForestClassifier(bootstrap=True, class_weight='balanced',
            criterion='gini', max_depth=None, max_features='auto',
            max_leaf_nodes=None, min_impurity_decrease=0.0,
            min_impurity_split=None, min_samples_leaf=2,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            n_estimators=9, n_jobs=1, oob_score=False, random_state=42,
            verbose=0, warm_start=False)

eclf = VotingClassifier(estimators=[('svc', clf3), ('ada', clf2), ('rf', clf1)], voting='hard')

### Evaluation metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer
mcc = make_scorer(matthews_corrcoef)
scorers = {'mcc': mcc, 'accuracy': 'accuracy', 'f1': 'f1', 
           'recall': 'recall', 'precision': 'precision'}

### CrossValidation
from sklearn.model_selection import GridSearchCV
parameters = {} # Using GridSearchCV just for CV
clf = GridSearchCV(eclf, parameters, scoring=scorers,
                   n_jobs=1, cv=3, refit='mcc', verbose=0)
clf.fit(features, labels)
print(clf.best_estimator_)
mcc = clf.cv_results_['mean_test_mcc'][clf.best_index_]
print('MCC:       {:0.4f}'.format(mcc))
f1 =  clf.cv_results_['mean_test_f1'][clf.best_index_]
print('F1:        {:0.4f}'.format(f1))
pre = clf.cv_results_['mean_test_precision'][clf.best_index_]
print('Precision: {:0.4f}'.format(pre))
rec = clf.cv_results_['mean_test_recall'][clf.best_index_]
print('Recall:    {:0.4f}'.format(rec))
acc = clf.cv_results_['mean_test_accuracy'][clf.best_index_]
print('Accuracy:  {:0.4f}'.format(acc))


dump_classifier_and_data(eclf, my_dataset, features_list)
