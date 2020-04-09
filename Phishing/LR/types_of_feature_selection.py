import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.feature_selection import chi2
from sklearn.model_selection import train_test_split

# uses the chi squared (chi^2) statistical test for non-negative features to select 4 of the best features from the Pima Indians onset of diabetes dataset.

#load data
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataFrame = pd.read_csv(url,names=names)
array = dataFrame.values
features = array[:,0:8]
labels = array[:,8]

#feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(features,labels)
# summarize scores
np.set_printoptions(precision=3)
#print(fit.scores_)
#print(features)
new_features = fit.transform(features)
# summarize selected features
#print(new_features)
# summarize selected features
#print(new_features[0:5,:])





