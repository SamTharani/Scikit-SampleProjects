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
import pandas as pd
import seaborn as sns
import color

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

loc = ("fake.xls") 
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 

#out_file = open('fake_news_results.csv', 'w')
##out_file.write('pos_news_score'+','+'neg_news_score'+','+'pos_title_score'+','+'neg_title_score'+','+'replies'+','+'comments'+','+'likes'+','+'participants'+','+'shares'+os.linesep)#
#
#for i in range(1,sheet.nrows): 
#  spam_score = float(sheet.cell_value(i, 12))
#  fake_cls = sheet.cell_value(i, 19)
#  num_replies = int(sheet.cell_value(i, 14))
#  num_participants = int(sheet.cell_value(i, 15))
#  num_likes = int(sheet.cell_value(i, 16))
#  num_comments = int(sheet.cell_value(i, 17))
# num_shares = int(sheet.cell_value(i, 18))
#  if spam_score >= 0.7:
#    fake_class = 1
#  else:
#    fake_class = 0
#
#  news = sheet.cell_value(i, 5)
#  title = sheet.cell_value(i, 4)
#  news = remove_non_ascii_1(news)
#  title = remove_non_ascii_1(title)
#  scores = sentiment_analyzer_scores(news)
#  scores2 = sentiment_analyzer_scores(title)
#
#  author = sheet.cell_value(i, 2)
#  published = sheet.cell_value(i,3)
#  language = sheet.cell_value(i,6)
#  thread_title = sheet.cell_value(i,11) 
#  site_url = sheet.cell_value(i,8)
#  country = sheet.cell_value(i,9)
#
#  #X_list.append(scores[)
#  #Y_list.append(fake_class) 
#
#  out_file.write(str(scores['pos'])+','+str(scores['neg'])+','+str(scores2['pos'])+','+str(scores2['neg'])+','+str(num_replies)+','+str(num_comments)+','+str(num_likes)+','+str(num_participants)+','+str(num_shares)+','+str(spam_score)+os.linesep)
#
#out_file.close()
##
##
#
#
# Step 1 - Make a scatter plot with square markers, set column names as labels

n_colors = 256 # Use 256 colors for the diverging color palette
palette = sns.diverging_palette(20, 220, n=n_colors) # Create the palette
color_min, color_max = [-1, 1] # Range of values that will be mapped to the palette, i.e. min and max possible correlation
def value_to_color(val):
    val_position = float((val - color_min)) / (color_max - color_min) # position of value in the input range, relative to the length of the input range
    ind = int(val_position * (n_colors - 1)) # target index in the color palette
    return palette[ind]


def heatmap(x, y, size, color):
    #fig, ax = plt.subplots()

    plot_grid = plt.GridSpec(1, 15, hspace=0.2, wspace=0.1) # Setup a 1x15 grid
    ax = plt.subplot(plot_grid[:,:-1]) # Use the leftmost 14 columns of the grid for the main plot
    
    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]:p[0] for p in enumerate(x_labels)} 
    y_to_num = {p[1]:p[0] for p in enumerate(y_labels)} 
    
    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num),
        y=y.map(y_to_num),
        s=size_scale, #size * size_scale,
        c=color.apply(value_to_color), # Vector of square color values, mapped to color palette
        marker='s'
    )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

    ax.grid(False, 'major')
    ax.grid(True, 'minor')
    ax.set_xticks([t + 0.5 for t in ax.get_xticks()], minor=True)
    ax.set_yticks([t + 0.5 for t in ax.get_yticks()], minor=True)
    ax.set_xlim([-0.5, max([v for v in x_to_num.values()]) + 0.5]) 
    ax.set_ylim([-0.5, max([v for v in y_to_num.values()]) + 0.5])


    # Add color legend on the right side of the plot
    ax = plt.subplot(plot_grid[:,-1]) # Use the rightmost column of the plot
    col_x = [0]*len(palette) # Fixed x coordinate for the bars
    bar_y=np.linspace(color_min, color_max, n_colors) # y coordinates for each of the n_colors bars

    bar_height = bar_y[1] - bar_y[0]
    ax.barh(
        y=bar_y,
        width=[5]*len(palette), # Make bars 5 units wide
        left=col_x, # Make bars start at 0
        height=bar_height,
        color=palette,
        linewidth=0
    )

    ax.set_xlim(1, 2) # Bars are going from 0 to 5, so lets crop the plot somewhere in the middle
    ax.grid(False) # Hide grid
    ax.set_facecolor('white') # Make background white
    ax.set_xticks([]) # Remove horizontal ticks
    ax.set_yticks(np.linspace(min(bar_y), max(bar_y), 3)) # Show vertical ticks for min, middle and max
    ax.yaxis.tick_right() # Show vertical ticks on the right 
    plt.savefig('feature_correlation_2.eps')
    plt.show()
    
data = pd.read_csv('fake_news_results.csv')
columns = ['pos_news_score','neg_news_score','pos_title_score','neg_title_score','replies','comments','likes','participants','shares','spam_score'] 
corr = data[columns].corr()
corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
corr.columns = ['x', 'y', 'value']
heatmap(
    x=corr['x'],
    y=corr['y'],
    size=corr['value'].abs(),
    color = corr['value']
)



# ...



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
