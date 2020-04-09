import pandas as  pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
import subprocess

analyser = SentimentIntensityAnalyzer()


def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score

#Average of the single sentances for news article
def avg(x, len):
  sum = 0
  for i in x:
     sum += i
  return round(sum/len,4)


sentiment_score = []
status_msg = []
sentences = []
rpath = "datasets/articles/2018-03-03"

rlpath = "./datasets/SentimentScore/NELA-GT/2018-03-03/"

for dir in os.listdir(rpath):
    flpath = rpath+"/"+dir

    # Sentiment value for the sentences of news article
    for file in os.listdir(flpath):
        files = open(flpath+"/"+file,"r")
        for line in files:
            sen = line.split('\r\n')
            for s in sen:
                if s != '':
                    sentences.append(s)
                    status_msg.append(s)
                    score = sentiment_analyzer_scores(s)
                    sentiment_score.append(score)
        sentiment_text = pd.DataFrame(status_msg)
        sentiment_val = pd.DataFrame(sentiment_score)
        # concat two dataframes for better visualization
        sentimentScores = pd.concat([sentiment_text,sentiment_val],axis=1)
        sentimentScores.to_csv(rlpath+file+'.csv')




    sentiment_article = pd.DataFrame(columns=['text','compound','neu','neg','pos'])

    #Sentiment value for news article
    for file in os.listdir(flpath):
        files = open(flpath+"/"+file,"r")
        sentiment_article['text'] = [files.read()]

        sentiment_data = pd.read_csv(rlpath+file+'.csv')

        compound = sentiment_data['compound']
        neutral = sentiment_data['neu']
        negative = sentiment_data['neg']
        positive = sentiment_data['pos']

        sentiment_article['compound'] = [avg(compound, len(compound))]
        sentiment_article['neu'] = [avg(neutral, len(neutral))]
        sentiment_article['neg'] = [avg(negative, len(negative))]
        sentiment_article['pos'] = [avg(positive, len(positive))]

        sentiment_article.to_csv(rlpath+file+'_avg.csv')

subprocess.call("./copy_file.sh")
