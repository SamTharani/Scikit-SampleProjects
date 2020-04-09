import pandas as  pd
import unicodedata
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plot


#Load fake news data from facebook

fake_data = pd.read_csv(filepath_or_buffer='./fakedata/sentiment/FA-KES-Dataset.csv', encoding='latin1')




################Lable Datasets#######################

df = pd.DataFrame(fake_data)

#Scrap status messages from the dataset
status_message = fake_data['article_title']

label = fake_data['labels']

analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    return score


sentiment = []

for s in status_message:
    score = sentiment_analyzer_scores(str(s))
    if score.get('compound') >= 0.05:
        sentiment.append('positive')
    elif score.get('compound') > -0.05 and score.get('compound') < 0.05:
        sentiment.append('neutral')
    else:
        sentiment.append('negative')

df['sentiment'] = sentiment
sentiment_fake = pd.DataFrame({'sentiment':sentiment,'label':label})


fake = pd.DataFrame(columns=['sentiment', 'label'])
real = pd.DataFrame(columns=['sentiment', 'label'])


for i in range(sentiment_fake.shape[0]):
    dt = sentiment_fake.iloc[i]
    if dt['label'] == 0:
        fake = fake.append({'sentiment': dt['sentiment'], 'label': 0}, ignore_index=True)
    else:
        real = real.append({'sentiment': dt['sentiment'], 'label': 1}, ignore_index=True)




# for i in status_message:
#     s = re.findall(r"[ADFJMNOS]\w* [\d]{1,2} [ADFJMNOS]\w* [\d]{4}", i)
#     #    re.sub(r'[ADFJMNOS]\w* [\d]{1,2} [ADFJMNOS]\w* [\d]{4}', r'', i)
#     listToString(s)
#     if(listToString(s)!=""):
#         utf8string = i.encode("latin1")
#         utf8string.replace(listToString(s),"")
#         print(i)

    # s = re.findall(r"[ADFJMNOS]\w* [\d]{1,2} [ADFJMNOS]\w* [\d]{4}",i)
    # i = unicodedata.normalize('NFKD', i).encode('ascii', 'ignore')

    # def listToString(s):
    #     # initialize an empty string
    #     str1 = " "
    #
    #     # return string
    #     return (str1.join(s))
