import pandas as  pd
from sklearn.preprocessing import LabelEncoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plot
import seaborn as sns
from sklearn.model_selection import train_test_split


#Load fake news data from facebook

fake_data = pd.read_csv('./fakedata/sentiment/barstool-sports.csv')


################Lable Datasets#######################

# df = pd.DataFrame(fake_data)
# #Scrap status messages from the dataset
# status_message = fake_data['status_message']
# #print(status_message)
# #status_message.to_csv('./fakedata/buzzfeed_status.csv')
#
# #Analyse the sentiments of the status message
# analyser = SentimentIntensityAnalyzer()
#
# def sentiment_analyzer_scores(sentence):
#     score = analyser.polarity_scores(sentence)
#     return score
#
#
# labels = []
#
# for s in status_message:
#     score = sentiment_analyzer_scores(str(s))
#     if score.get('compound') >= 0.05:
#         labels.append('positive')
#     elif score.get('compound') > -0.05 and score.get('compound') < 0.05:
#         labels.append('neutral')
#     else:
#         labels.append('negative')
# #print (labels)
# df['label'] = labels
# sentiment_data = pd.DataFrame({'status_message':status_message,
#                                'label':labels})
#df.to_csv('./fakedata/sentiment/food-network_posts.csv')

################ End #######################
cols = [col for col in fake_data.columns if col not in ['label','status_message']]
columns = fake_data.columns.tolist()

sentiment_feature = fake_data[cols]


#print(sentiment_feature)

label = fake_data['label']
#print(label)

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(sentiment_feature,label, test_size = 0.30, random_state = 10)
# data_train.to_csv('./fakedata/train.csv')
# target_train.to_csv('./fakedata/trlabel.csv')
# data_test.to_csv('./fakedata/test.csv')
# target_test.to_csv('./fakedata/tslabel.csv')


def heatmap(x, y, size):
    fig, ax = plot.subplots()

    # Mapping from column names to integer coordinates
    x_labels = [v for v in sorted(x.unique())]
    y_labels = [v for v in sorted(y.unique())]
    x_to_num = {p[1]: p[0] for p in enumerate(x_labels)}
    y_to_num = {p[1]: p[0] for p in enumerate(y_labels)}

    size_scale = 500
    ax.scatter(
        x=x.map(x_to_num),  # Use mapping for x
        y=y.map(y_to_num),  # Use mapping for y
        s=size * size_scale,  # Vector of square sizes, proportional to size parameter
        marker='s'  # Use square as scatterplot marker
    )

    # Show column labels on the axes
    ax.set_xticks([x_to_num[v] for v in x_labels])
    ax.set_xticklabels(x_labels, rotation=45, horizontalalignment='right')
    ax.set_yticks([y_to_num[v] for v in y_labels])
    ax.set_yticklabels(y_labels)

corr = sentiment_feature.corr()
print(corr)
# corr = pd.melt(corr.reset_index(), id_vars='index') # Unpivot the dataframe, so we can get pair of arrays for x and y
# corr.columns = ['x', 'y', 'value']
# heatmap(
#     x=corr['x'],
#     y=corr['y'],
#     size=corr['value'].abs(),
# )

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220,10,as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr,cmap=cmap,vmax=1,center=0,square=True,
            linewidth=.5, cbar_kws={'shrink': .5})
# Set up the matplotlib figure
f, ax = plot.subplots(figsize=(11,9))
ax.set_title('Multi-Collinearity of Features')
plot.show()