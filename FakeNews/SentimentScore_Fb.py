import pandas as  pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt


#Load fake news data from facebook

sentiment_data = pd.read_csv('./fakedata/sentiment/fake.csv')

df = pd.DataFrame(sentiment_data)

#Scrap status messages from the dataset
status_message = sentiment_data['text']
label = sentiment_data['type']
analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score

def avg(x, len):
  sum = 0
  for i in x:
     sum += i
  return round(sum/len,4)

sentiment_score = []

for s in status_message:
    score = sentiment_analyzer_scores(str(s))
    sentiment_score.append(score)

sentiment_text = pd.DataFrame(status_message)
sentiment_val = pd.DataFrame(sentiment_score)

sentiment_GRFN = pd.concat([sentiment_text,sentiment_val,label],axis=1)

#concat two dataframes for better visualization
#sentimentScores = pd.concat([sentiment_text,sentiment_val],axis=1)
#sentimentScoresforanalysis = pd.concat([sentiment_val,label],axis=1)

sentiment_GRFN.to_csv('./datasets/SentimentScore/GRFN_content.csv')
#sentimentScoresforanalysis.to_csv('./datasets/SentimentScore/FA-KES_analysis_title.csv')

sentiment_analysis = pd.read_csv('./datasets/SentimentScore/GRFN_content.csv')
#sentiment_analysis = pd.read_csv('./datasets/SentimentScore/FA-KES_analysis_title.csv')

neg = sentiment_analysis['neg']
pos = sentiment_analysis['pos']
fakeOrReal = sentiment_analysis['type']

# fakes = [] #choose label as one
# real = []  #choose label as zero

#fakes = sentiment_analysis.query('type == `fake`').index.tolist()

fakes = sentiment_analysis.loc[sentiment_analysis['type'] == 'fake']
fake_index = fakes.iloc[:,0]

# real = sentiment_analysis.query('labels == 0').index.tolist()
#
fakeforneg = neg[fake_index]
fakeforpos = pos[fake_index]
#
# realforneg = neg[real]
# realforpos = pos[real]

plt.plot(fakeforneg,'r',fakeforpos,'b')
plt.ylabel('Sentiment Score For Fake GRFN Content')
plt.show()

# #Article Title
# print(fakeforneg.max()) #0.858
# print(fakeforneg.min()) #0.0
# print(avg(fakeforneg,len(fakeforneg))) #0.4006
#
# print(fakeforpos.max()) #0.231
# print(fakeforpos.min()) #0.0
# print(avg(fakeforpos,len(fakeforpos))) #0.0071

#Article Content
# print(fakeforneg.max()) #0.409
# print(fakeforneg.min()) #0.0220
# print(avg(fakeforneg,len(fakeforneg))) #0.1865
#
# print(fakeforpos.max()) #0.182
# print(fakeforpos.min()) #0.0
# print(avg(fakeforpos,len(fakeforpos))) #0.0301



