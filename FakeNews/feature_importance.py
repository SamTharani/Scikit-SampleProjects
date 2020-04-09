from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
import numpy as np
#from senti_classifier import senti_classifier
import os
import xlrd 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
#from senti_classifier import senti_classifier
import csv
import matplotlib.pyplot as plt
import math


from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier


def remove_non_ascii_1(text):
  return ''.join([i if ord(i) < 128 else ' ' for i in text])

#in_file = open('fake.csv','r')
#out_file = open('results_fake.csv', 'w')

#in_file.readline()

# Build a classification task using 3 informative features
#X, y = make_classification(n_samples=1000,
#                           n_features=10,
#                           n_informative=3,
#                           n_redundant=0,
#                           n_repeated=0,
#                           n_classes=2,
#                           random_state=0,
#                           shuffle=False)

#print X, y

analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(score)))
    return score

X_list = []
Y_list = []

X2_list = []
Y2_list = []

loc = ("fake.xls") 
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 

num_recs = sheet.nrows - 1
training_num_recs = int(math.floor(num_recs * 0.8))
test_num_recs = num_recs - training_num_recs

for i in range(1,training_num_recs): 
#for i in range(1,sheet.nrows): 
  spam_score = float(sheet.cell_value(i, 12))
  fake_cls = sheet.cell_value(i, 19)
  num_replies = int(sheet.cell_value(i, 14))
  num_participants = int(sheet.cell_value(i, 15))
  num_likes = int(sheet.cell_value(i, 16))
  num_comments = int(sheet.cell_value(i, 17))
  num_shares = int(sheet.cell_value(i, 18))
  if spam_score >= 0.7:
    fake_class = 1
  else:
    fake_class = 0

  news = sheet.cell_value(i, 5)
  title = sheet.cell_value(i, 4)
  news = remove_non_ascii_1(news)
  title = remove_non_ascii_1(title)
  scores = sentiment_analyzer_scores(news)
  scores2 = sentiment_analyzer_scores(title)

  author = sheet.cell_value(i, 2)
  published = sheet.cell_value(i,3)
  language = sheet.cell_value(i,6)
  thread_title = sheet.cell_value(i,11) 
  site_url = sheet.cell_value(i,8)
  country = sheet.cell_value(i,9)

  this_rec_attrs = [num_replies, num_participants, num_likes, num_comments, num_shares, scores['pos'], scores['neg'], scores2['pos'], scores2['neg']]
  X_list.append(this_rec_attrs)
  Y_list.append(fake_class) 
  #Y_list.append(fake_class)

  this_rec_attrs2 = [num_comments] #author, published, language, title, news, site_url, country]
  X2_list.append(this_rec_attrs2)
  Y2_list.append(fake_class) 
  #Y_list.append(fake_class)

#with open('fake.csv', 'r') as f:
#    f.readline()
#    data = f.readlines()
#    print len(data)
##for line in in_file:
#    #joint_char = ","
#    for line in data:
#      line_list = [col for col in line.strip().split(",")] #csv.reader(f, delimiter=',')]
#      #line_list = line.split(joint_char)
#      print line_list, len(line_list), line_list[-8]
#      if line_list[-8] == '""':
#        spam_score = 0.0
#      else:
#        #print line_list[-8]
#        spam_score = float(line_list[-8])
#      if spam_score > 0.7:
#        fake_class = 1
#      else:
#        fake_class = 0
#      num_replies = int(line_list[-6])
#      num_participants = int(line_list[-5])
#      num_likes = int(line_list[-4])
#      num_comments = int(line_list[-3])
#      num_shares = int(line_list[-2])
#      this_rec_attrs = [num_replies, num_participants, num_likes, num_comments, num_shares]
#      #print [fake_class], [this_rec_attrs]
#      X_list.append(this_rec_attrs)
#      Y_list.append(fake_class)
      
#print X_list
y = np.array(Y_list)
X = np.array(X_list)
      #print X, y

print(X, y)

y2 = np.array(Y2_list)
X2 = np.array(X2_list)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(X, y)

#forest2 = ExtraTreesClassifier(n_estimators=250,
#                              random_state=0)
#forest2.fit(X2, y2)

importances = forest.feature_importances_

std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)

indices = np.argsort(importances)[::-1]

#importances2 = forest2.feature_importances_

#std2 = np.std([tree.feature_importances_ for tree in forest2.estimators_],
#             axis=0)

#indices2 = np.argsort(importances2)[::-1]

corr_predictions = 0
#corr_predictions2 = 0
for i in range(training_num_recs, sheet.nrows): 
#for i in range(1,sheet.nrows): 
  spam_score = float(sheet.cell_value(i, 12))
  fake_cls = sheet.cell_value(i, 19)
  num_replies = int(sheet.cell_value(i, 14))
  num_participants = int(sheet.cell_value(i, 15))
  num_likes = int(sheet.cell_value(i, 16))
  num_comments = int(sheet.cell_value(i, 17))
  num_shares = int(sheet.cell_value(i, 18))
  if spam_score >= 0.7:
    fake_class = 1
  else:
    fake_class = 0

  news = sheet.cell_value(i, 5)
  title = sheet.cell_value(i, 4)
  news = remove_non_ascii_1(news)
  title = remove_non_ascii_1(title)
  scores = sentiment_analyzer_scores(news)
  scores2 = sentiment_analyzer_scores(title)

  author = sheet.cell_value(i, 2)
  published = sheet.cell_value(i,3)
  language = sheet.cell_value(i,6)
  thread_title = sheet.cell_value(i,11) 
  site_url = sheet.cell_value(i,8)
  country = sheet.cell_value(i,9)

  this_rec_attrs = [num_replies, num_participants, num_likes, num_comments, num_shares, scores['pos'], scores['neg'], scores2['pos'], scores2['neg']]
  #X_list.append(this_rec_attrs)
  #Y_list.append(fake_class) 
  #Y_list.append(fake_class)

  this_rec_attrs2 = [num_comments] #,author, published, language, title, news, site_url, country]
  #X2_list.append(this_rec_attrs2)
  #Y2_list.append(fake_class) 
  #Y_list.append(fake_class)

  if forest.predict([this_rec_attrs]) == [fake_class]:
    corr_predictions += 1

  #if forest2.predict([this_rec_attrs2]) == [fake_class]:
  #  corr_predictions2 += 1

print(corr_predictions, test_num_recs)
#print(corr_predictions2, test_num_recs)

#with open('fake.csv', 'r') as f:
#    f.readline()
#    data = f.readlines()
#    print len(data)
##for line in in_file:
#    #joint_char = ","
#    for line in data:
#      line_list = [col for col in line.strip().split(",")] #csv.reader(f, delimiter=',')]
#      #line_list = line.split(joint_char)
#      print line_list, len(line_list), line_list[-8]
#      if line_list[-8] == '""':
#        spam_score = 0.0
#      else:
#        #print line_list[-8]
#        spam_score = float(line_list[-8])
#      if spam_score > 0.7:
#        fake_class = 1
#      else:
#        fake_class = 0
#      num_replies = int(line_list[-6])
#      num_participants = int(line_list[-5])
#      num_likes = int(line_list[-4])
#      num_comments = int(line_list[-3])
#      num_shares = int(line_list[-2])
#      this_rec_attrs = [num_replies, num_participants, num_likes, num_comments, num_shares]
#      #print [fake_class], [this_rec_attrs]
#      X_list.append(this_rec_attrs)
#      Y_list.append(fake_class)
      
#print X_list
#y2 = np.array(Y2_list)
#X2 = np.array(X2_list)
#      #print X, y
#
#print(X2, y2)
#
#
#
#labels = []
#
#for x in range(len(indices)):
#  if indices[x] == 0:
#   labels.append('replies')
#  elif indices[x] == 1:
#    labels.append('participants')
#  elif indices[x] == 2:
#   labels.append('likes')
#  elif indices[x] == 3:
#    labels.append('comments')
#  elif indices[x] == 4:
#    labels.append('shares')
#  elif indices[x] == 5:
#    labels.append('news_pos')
#  elif indices[x] == 6:
#    labels.append('news_neg')
#  elif indices[x] == 7:
#    labels.append('title_pos')
#  elif indices[x] == 8:
#    labels.append('title_neg')
#
#print(indices, labels )
#
## Print the feature ranking
#print("Feature ranking:")
#
#for f in range(X.shape[1]):
#    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
#
#
# Plot the feature importances of the forest
#plt.figure()
#plt.title("Feature importances for spam socre")
#plt.bar(range(X.shape[1]), importances[indices],
#       color="r", yerr=std[indices], align="center")
#plt.xticks(range(X.shape[1]), labels, rotation=30) #labels)
##plt.xticks(ticks, labels)
#plt.xlim([-1, X.shape[1]])
#plt.savefig('feature_importance_fake_spam_score.eps')
#plt.show()
##
#
#
#
##out_file.write(truth+','+str(pos_score)+','+str(neg_score)+os.linesep)
#
##out_file.close()
