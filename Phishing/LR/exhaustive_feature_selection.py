import pandas as pd
import  numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from mlxtend.feature_selection import ExhaustiveFeatureSelector

paribas_data = pd.read_csv("../dataset/bnp-paribas-cardif-claims-management/train.csv", nrows=20000)

# Remove no-numeric and correlated columns
num_columns = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_columns = list(paribas_data.select_dtypes(include=num_columns).columns)
paribas_data = paribas_data[numerical_columns]

train_features, test_features, train_labels, test_labels = train_test_split(
    paribas_data.drop(labels=['target', 'ID'], axis=1),
    paribas_data['target'],
    test_size=0.2,
    random_state=41)

correlated_features = set()
correlation_matrix = paribas_data.corr()
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8: #0.8 is a correlation threshold value
            column_name = correlation_matrix.columns[i]
            correlated_features.add(column_name)

train_features.drop(labels=correlated_features, axis=1, inplace=True)
test_features.drop(labels=correlated_features,axis=1, inplace=True)

feature_selector = ExhaustiveFeatureSelector(RandomForestClassifier(n_jobs=-1),
                                             min_features=2,
                                             max_features=4,
                                             scoring='roc_auc',
                                             print_progress=True,
                                             cv=2)
features = feature_selector.fit(np.array(train_features.fillna(0)),train_labels)

filtered_features = train_features.columns[list(features.k_feature_idx_)]

clf = RandomForestClassifier(n_estimators=100, random_state=41, max_depth=3)
clf.fit(train_features[filtered_features].fillna(0), train_labels)

train_pred = clf.predict_proba(train_features[filtered_features].fillna(0))
print('Accuracy on training set: {}'.format(roc_auc_score(train_labels, train_pred[:,1])))

test_pred = clf.predict_proba(test_features[filtered_features].fillna(0))
print('Accuracy on test set: {}'.format(roc_auc_score(test_labels, test_pred [:,1])))