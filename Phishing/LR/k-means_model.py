import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

phishing_data = pd.read_csv("../dataset/dataset.csv", nrows=11055)

train_features, test_features, train_labels, test_labels = train_test_split(
    phishing_data.drop(labels=['Result', 'id'], axis=1),
    phishing_data['Result'],
    test_size=0.2,
    random_state=41)


kmeans = KMeans(n_clusters=2)
kmeans.fit(train_features)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

print(centroids)
print(labels)



