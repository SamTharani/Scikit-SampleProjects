#import necessary modules
import pandas as pd
import matplotlib.pyplot as plot
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ClassificationReport
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np


# read phishing URL data
url_data = pd.read_csv('./dataset/Phishing.csv') #create datafram (tabular data-structure: first row is a column name , first column contains id for each row )

#print(url_data.head(n=2)) #view the first few records
#print(url_data.tail(n=10)) #view the last few records
#print(url_data.dtypes)

#set the background colour of the plot to white
#sns.set(style="whitegrid", color_codes=True)

#setting the plot size for all plots
#sns.set(rc={'figure.figsize':(11.7,8.27)})
#create a countplot
#sns.countplot('NumNumericChars', data=url_data, hue='CLASS_LABEL')

#Reomve the top and down margin
#sns.despine(offset=10, trim=True)
#plot.show()

#split training and testing data
# Filter the features from the target column

cols = [col for col in url_data.columns if col not in ['id','CLASS_LABEL']]

url_feature = url_data[cols]

label = url_data['CLASS_LABEL']

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(url_feature,label, test_size = 0.30, random_state = 10)

#data_train.to_csv('./dataset/trainf.csv')
#target_train.to_csv('./dataset/trainl.csv')
#data_test.to_csv('./dataset/testf.csv')
#target_test.to_csv('./dataset/testl.csv')

#Train the model

#create GaussianNB object
gnb = GaussianNB()
#train the algorithm on training data and predict using the testing data
pred = gnb.fit(data_train, target_train).predict(data_test)

#Confusion matrix
print("Confusion matrix for Naive-Bayes : ",confusion_matrix(pred, target_test))

# names = np.unique(pred)
# sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False,
#             xticklabels=names, yticklabels=names)
# plot.xlabel('Truth')
# plot.ylabel('Predicted')
# plot.show()

#print the accuracy score of the model
print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred, normalize = True))

#create LinearSVC model
svc_model = LinearSVC(random_state=0)

#train the algorithm on training data and predict using the testing data
pred = svc_model.fit(data_train, target_train).predict(data_test)

#Confusion matrix
print("Confusion matrix for LinearSVC : ",confusion_matrix(pred, target_test))

#print the accuracy score of the model
print("LinearSVC accuracy : ",accuracy_score(target_test, pred, normalize = True))

#create object of the KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)

#Train the algorithm
neigh.fit(data_train, target_train)

# predict the response
pred = neigh.predict(data_test)

#Confusion matrix
print("Confusion matrix for KNeighbors : ",confusion_matrix(pred, target_test))

# evaluate accuracy
print ("KNeighbors accuracy score : ",accuracy_score(target_test, pred))


# #plot classification report
# # Instantiate the classification model and visualizer
#
# #Gaussian Navi Base
# visualizer = ClassificationReport(gnb, classes=['Ligitimate','Phishing'])
# visualizer.fit(data_train, target_train) # Fit the training data to the visualizer
# visualizer.score(data_test, target_test) # Evaluate the model on the test data
# #g = visualizer.poof() # Draw/show/poof the data
#
# #LinearSVC
# visualizer = ClassificationReport(svc_model, classes=['Ligitimate','Phishing'])
# visualizer.fit(data_train, target_train) # Fit the training data to the visualizer
# visualizer.score(data_test, target_test) # Evaluate the model on the test data
# #lsvc = visualizer.poof() # Draw/show/poof the data
#
# #KNeighborsClassifier
# visualizer = ClassificationReport(neigh, classes=['Ligitimate','Phishing'])
# visualizer.fit(data_train, target_train) # Fit the training data to the visualizer
# visualizer.score(data_test, target_test) # Evaluate the model on the test data
# #knn = visualizer.poof() # Draw/show/poof the data


#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=f_classif, k=10)
fit = bestfeatures.fit(data_train,target_train)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(data_train.columns)



#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
#print(featureScores)
featureScores.columns = ['URL_features','Score']  #naming the dataframe columns
#print(featureScores.nlargest(20,'Score'))  #print 10 best features

best10_features = featureScores.nlargest(15,'Score')
#best10_features.to_csv('./dataset/best10.csv')

# Naive-Bayes accuracy :  0.837
# LinearSVC accuracy :  0.9183333333333333
# KNeighbors accuracy score :  0.8713333333333333

#                           URL_features        Score
# 47  PctExtNullSelfRedirectHyperlinksRT  3028.933077
# 34          FrequentDomainNameMismatch  1971.202716
# 4                              NumDash  1138.258721
# 38                   SubmitInfoToEmail  1027.151102
# 33       PctNullSelfRedirectHyperlinks   944.585365
# 29                       InsecureForms   745.967495
# 0                              NumDots   682.739637
# 26                    PctExtHyperlinks   550.028317
# 24                   NumSensitiveWords   505.983345
# 39                       IframeOrFrame   399.649120
# 2                            PathLevel   363.937885
# 45              AbnormalExtFormActionR   221.524336
# 43                         UrlLengthRT   199.513030
# 20                      HostnameLength   194.873075
# 5                    NumDashInHostname   168.570028
# 10                  NumQueryComponents   154.485877
# 25                   EmbeddedBrandName   152.045080
# 32                  AbnormalFormAction   135.219340
# 16                           IpAddress   120.645341
# 18                       DomainInPaths   106.725700


best10_columns = best10_features['URL_features']

filteredTrain_feature = pd.DataFrame()


for f in range(best10_columns.shape[0]):
    filteredTrain_feature[best10_columns.iloc[f]] = data_train[best10_columns.iloc[f]]
filteredTrain_feature.to_csv('./dataset/filtered20_train.csv')

filteredTest_feature = pd.DataFrame()
for f in range(best10_columns.shape[0]):
    filteredTest_feature[best10_columns.iloc[f]] = data_test[best10_columns.iloc[f]]
filteredTest_feature.to_csv('./dataset/filtered20_test.csv')

#create GaussianNB object
gnb = GaussianNB()
#train the algorithm on training data and predict using the testing data
pred1 = gnb.fit(filteredTrain_feature, target_train).predict(filteredTest_feature)
#Confusion matrix
print("Confusion matrix for Naive-Bayes : ",confusion_matrix(pred1, target_test))
#print the accuracy score of the model
print("Naive-Bayes accuracy : ",accuracy_score(target_test, pred1, normalize = True))

#create LinearSVC model
svc_model = LinearSVC(random_state=0)

#train the algorithm on training data and predict using the testing data
pred1 = svc_model.fit(filteredTrain_feature, target_train).predict(filteredTest_feature)
#Confusion matrix
print("Confusion matrix for LinearSVC : ",confusion_matrix(pred1, target_test))
#print the accuracy score of the model
print("LinearSVC accuracy : ",accuracy_score(target_test, pred1, normalize = True))

#create object of the KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
#Train the algorithm
neigh.fit(filteredTrain_feature, target_train)
# predict the response
pred1 = neigh.predict(filteredTest_feature)
#Confusion matrix
print("Confusion matrix for KNeighbors : ",confusion_matrix(pred1, target_test))
# evaluate accuracy
print ("KNeighbors accuracy score : ",accuracy_score(target_test, pred1))

#New Classification result
# Naive-Bayes accuracy :  0.8176666666666667
# LinearSVC accuracy :  0.8876666666666667
# KNeighbors accuracy score :  0.9563333333333334


#Feature importance
model = ExtraTreesClassifier()
model.fit(data_train,target_train)
#print(model.feature_importances_) #use inbuilt class feature_importances of tree based classifiers
#plot graph of feature importances for better visualization
# feat_importances = pd.Series(model.feature_importances_, index=data_train.columns)
# feat_importances.nlargest(10).plot(kind='barh')
#plot.show()


#plot co-orelation matrix

corrmat = filteredTrain_feature.corr()
#cg = sns.heatmap(corrmat, cmap ="YlGnBu", annot = True,linewidths = 0.1)



#plot heat map
cg = sns.clustermap(corrmat, cmap ="YlGnBu", linewidths = 0.1, annot = True,fmt='.1g')
plot.setp(cg.ax_heatmap.yaxis.get_majorticklabels())
plot.show()









