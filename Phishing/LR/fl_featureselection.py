import pandas as pd
import  numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold


santandar_data = pd.read_csv("../dataset/santander-customer-satisfaction/train.csv")


#Spliting Data into training and samples
train_features, test_features, train_labels, test_labels=train_test_split(
    santandar_data.drop(labels=['TARGET'], axis=1),
    santandar_data['TARGET'],
    test_size=0.2,
    random_state=41
)
constant_filter = VarianceThreshold(threshold=0)
constant_filter.fit(train_features)

#Non-constant features - get_support() get non-constant columns
non_constant = len(train_features.columns[constant_filter.get_support()]) #length of the non constant feature array

#print(non_constant)
constant_filter_column = [column for column in train_features.columns
                          if column not in train_features.columns[constant_filter.get_support()]]

# for column in constant_filter_column:
#     print(column)

#Remove non constant features
#train_features = constant_filter.transform(train_features)
#test_features = constant_filter.transform(test_features)

train_features.drop(labels=constant_filter_column, axis=1, inplace=True)
test_features.drop(labels=constant_filter_column, axis=1, inplace=True)

#Removing Quasi-Constant features
qconstant_filter = VarianceThreshold(threshold=0.01)
qconstant_filter.fit(train_features)

#of non-Quasi constant
len(train_features.columns[qconstant_filter.get_support()])
qconstant_column = [qcolumn for qcolumn in train_features.columns
                    if qcolumn not in train_features.columns[qconstant_filter.get_support()] ]


# for column in qconstant_column:
#     print(column)

#Remove qconstant values
train_features = qconstant_filter.transform(train_features)
test_features = qconstant_filter.transform(test_features)

#===================Remove Duplicate Values=======================================#
santandar_data = pd.read_csv("../dataset/santander-customer-satisfaction/train.csv", nrows=20000)
d_train_features, d_test_features, d_train_labels, d_test_labels=train_test_split(
    santandar_data.drop(labels=['TARGET'], axis=1),
    santandar_data['TARGET'],
    test_size=0.2,
    random_state=41
)

d_train_featuresT = d_train_features.T

#duplicate() in pandas used to identify the duplicate rows from the dataframe

#print(d_train_featuresT.duplicated().sum())

unique_features = d_train_featuresT.drop_duplicates(keep= 'first').T

#print(unique_features)

duplicate_features = [d_column for d_column in d_train_features.columns
                      if d_column not in unique_features.columns]

#print(duplicate_features)

#===================Remove correlated Features=======================================#

paribas_data = pd.read_csv("../dataset/bnp-paribas-cardif-claims-management/train.csv", nrows=20000)

#To find correlation, we oly need the numerical features in our dataset

#Filter out all other features, except the numeric ones
num_colums = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
numerical_column = list(paribas_data.select_dtypes(include=num_colums).columns)
paribas_data = paribas_data[numerical_column]

#Splitting the traing and testing data
p_train_features, p_test_features, p_train_labels, p_test_labels = train_test_split(
    paribas_data.drop(labels=['target', 'ID'], axis=1),
    paribas_data['target'],
    test_size=0.2,
    random_state=41)

#Removing Correlated Features using corr() method

correlated_feature = set()
correlation_matrix = paribas_data.corr()


for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > 0.8: #0.8 is a correlation threshold value
            column_name = correlation_matrix.columns[i]
            correlated_feature.add(column_name)

print(len(correlated_feature))
print(correlated_feature)

#Remove correlated feature from the dataset
p_train_features.drop(labels= correlated_feature, axis=1, inplace=True)
p_test_features.drop(labels=correlated_feature, axis=1, inplace=True)






