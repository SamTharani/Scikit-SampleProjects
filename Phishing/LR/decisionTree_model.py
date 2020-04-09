import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pprint import pprint

dataset = pd.read_csv('../dataset/Phishing.csv')

dataset = dataset.drop(columns='id')
# print(dataset)

def entropy(target_col):
    """
    Calculate the entropy of a dataset.
    The only parameter of this function is the target_col parameter which specifies the target column
    """
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy

# print(entropy(dataset['URL_Length']))

def InfoGain(data, split_attribute_name, target_name="CLASS_LABEL"):
    """
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """
    # Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])

    ##Calculate the entropy of the dataset

    # Calculate the values and the corresponding counts for the split attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    # Calculate the weighted entropy
    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    # Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


"""
1. Find the Information Gain of every features
2. Store in an array
3. Find the top 10 Information Gain features
"""
labels = dataset.columns
IG_list = []
best10_columns = []
for i in range(len(labels)-1):
    IG_list.append(InfoGain(dataset,labels[i],labels[48]))
IG_array = np.array(IG_list)
print(IG_array)
index_top_10Features = IG_array.argsort()[-10:][::-1]

print("###### Top ten features are #####")
for i in range(len(index_top_10Features)):
    best10_columns.append(labels[index_top_10Features[i]])

feature_column = pd.DataFrame(best10_columns,['URL_Features'])


cols = [col for col in dataset.columns if col not in ['id','CLASS_LABEL']]

url_feature = dataset[cols]

label = dataset['CLASS_LABEL']

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(url_feature,label, test_size = 0.30, random_state = 10)

filteredTrain_feature = pd.DataFrame()
for f in range(feature_column.shape[0]):
    filteredTrain_feature[feature_column.iloc[f]] = data_train[feature_column.iloc[f]]
filteredTrain_feature.to_csv('./dataset/ig_filtered_train.csv')

filteredTest_feature = pd.DataFrame()
for f in range(feature_column.shape[0]):
    filteredTest_feature[feature_column.iloc[f]] = data_test[feature_column.iloc[f]]
filteredTest_feature.to_csv('./dataset/ig_filtered_test.csv')

#print(dataset.iloc[:,index_top_10Features[0]])


