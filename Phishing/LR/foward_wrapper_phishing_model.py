import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import average_precision_score
from sklearn.metrics import classification_report
from sklearn import metrics



phishing_data = pd.read_csv("../dataset/dataset.csv", nrows=11055)

train_features, test_features, train_labels, test_labels = train_test_split(
    phishing_data.drop(labels=['Result', 'id'], axis=1),
    phishing_data['Result'],
    test_size=0.2,
    random_state=41)

labels = train_features.columns

# phishing_features = phishing_data.drop(labels=['Result', 'id'], axis=1)
# phishing_features.to_csv("../phishingdata/input.csv")
#
# label = phishing_data['Result']
# label.to_csv("../phishingdata/output.csv")
#
# input_data = np.genfromtxt('../phishingdata/input.csv',delimiter=',', dtype=np.int32)
# print(input_data)
#
# output_data = np.genfromtxt('../phishingdata/output.csv',delimiter=',', dtype=np.int32)
# print(output_data)
find_best = SelectKBest(score_func=f_classif, k=10)
fit = find_best.fit(train_features, train_labels)

np.set_printoptions(precision=3)
# print(fit.scores_.argsort())

train_features = fit.transform(train_features)
test_features = fit.transform(test_features)

top10_features_index = fit.scores_.argsort()[-10:][::-1]
top20_features_index = fit.scores_.argsort()[-20:][::-1]

print(fit.scores_)
print(fit.scores_.argsort())

for i in range(len(top10_features_index)):
    print(labels[top10_features_index[i]])

#
clf = RandomForestClassifier(n_estimators=100, random_state=41, max_depth=3)
clf.fit(train_features, train_labels)

train_pred_labels = clf.predict(train_features)
test_pre_labels = clf.predict(test_features)

train_pred = clf.predict_proba(train_features)

test_pred = clf.predict_proba(test_features)


#Results
target_names = ['Phishing', 'Legitimate']
print(classification_report(test_labels, test_pre_labels, target_names=target_names))

print('Accuracy on training set: {0:0.4f}'.format(roc_auc_score(train_labels, train_pred[:, 1])))

average_precision = average_precision_score(train_labels, train_pred[:, 1])
print('Average Precision for train data set: {0:0.4f} '.format(average_precision))

print('Accuracy on test set: {0:0.4f}'.format(roc_auc_score(test_labels, test_pred[:, 1])))
average_precision_test = average_precision_score(test_labels, test_pred[:, 1])
print('Average Precision for test data set: {0:0.4f} '.format(average_precision_test))

print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, test_pre_labels))
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, test_pre_labels))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, test_pre_labels)))



#For top 20 - features


# train_features, test_features, train_labels, test_labels = train_test_split(
#     phishing_data.drop(labels=['Result', 'id'], axis=1),
#     phishing_data['Result'],
#     test_size=0.2,
#     random_state=41)
#
# labels = train_features.columns







