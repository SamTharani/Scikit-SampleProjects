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

def remove_non_ascii_1(text):
  return ''.join([i if ord(i) < 128 else ' ' for i in text])

analyser = SentimentIntensityAnalyzer()

def sentiment_analyzer_scores(sentence):
    score = analyser.polarity_scores(sentence)
    #print("{:-<40} {}".format(sentence, str(score)))
    return score

#in_file = open('fake.csv','r')
#in_file.readline()
#
#lines = in_file.read()
#split_char = '"'+os.linesep
#print split_char
#lines_list = lines.split(split_char)
#
#print len(lines_list)
#
#for line in lines_list:
#    joint_char = ','
#    line_list = line.split(',')
#    #print line_list[5]
#    news = line_list[5]
#    news = remove_non_ascii_1(news)
#    sentences = news.split('.')
#    pos_score, neg_score = senti_classifier.polarity_scores(sentences)
#    print pos_score, neg_score



#in_file2 = open('Articles.csv','r')
#
#in_file2.readline()
#
#for line in in_file2:
#    joint_char = '"'+','
#    line_list = line.split(joint_char)
#    #print line_list[5]
#    news = line_list[0].split('"')[1]
#    news = remove_non_ascii_1(news)
#    sentences = news.split('.')
#    pos_score, neg_score = senti_classifier.polarity_scores(sentences)
#    print pos_score, neg_score

out_file4 = open('results_fake_news.csv', 'w')
loc = ("./datasets/fake-news/train2.xls") 
wb = xlrd.open_workbook(loc) 
sheet = wb.sheet_by_index(0) 
for i in range(1,sheet.nrows): 
  #for j in range(sheet.ncols): 
    #print(sheet.cell_value(0, i)) 
  news = sheet.cell_value(i, 3)
  title = sheet.cell_value(i, 1)
  truth = str(sheet.cell_value(i, 4))
  news = remove_non_ascii_1(news)
  title = remove_non_ascii_1(title)
  #sentences = news.split('.')
  scores = sentiment_analyzer_scores(news)
  scores2 = sentiment_analyzer_scores(title)
  #print truth, pos_score, neg_score
  out_file4.write(truth+','+str(scores['pos'])+','+str(scores['neu'])+','+str(scores['neg'])+','+str(scores['compound'])+str(scores2['pos'])+','+str(scores2['neu'])+','+str(scores2['neg'])+','+str(scores2['compound'])+os.linesep)
out_file4.close()


in_file3 = open('liar_dataset/train.tsv','r')
out_file3 = open('results_liar.csv', 'w')

in_file3.readline()

ddd = 0
for line in in_file3:
    ddd += 1
    joint_char = '\t'
    line_list = line.split(joint_char)
    #print line_list[5]
    news = line_list[2]
    truth = line_list[1]
    news = remove_non_ascii_1(news)
    #sentences = news.split('.')
    scores = sentiment_analyzer_scores(news)
    #print truth, pos_score, neg_score
    print(ddd)
    out_file3.write(truth+','+str(scores['pos'])+','+str(scores['neu'])+','+str(scores['neg'])+','+str(scores['compound'])+os.linesep)

out_file3.close()




#out_file4 = open('results_fake_news_news_text_emotion.csv', 'w')
#out_file5 = open('results_fake_news_title_text_emotion.csv', 'w')
#loc = ("./fake.xls") 
#wb = xlrd.open_workbook(loc) 
#sheet = wb.sheet_by_index(0) 
#ddd = 0
#for i in range(1,sheet.nrows): 
#  ddd += 1
#  #for j in range(sheet.ncols): 
#    #print(sheet.cell_value(0, i)) 
#  uid = sheet.cell_value(i, 0)
#  spam_score = float(sheet.cell_value(i, 12))
#  fake_cls = sheet.cell_value(i, 19)
#  if spam_score >= 0.7:
#    fake_class = 1
#  else:
#    fake_class = 0
#  news = sheet.cell_value(i, 5)
#  title = sheet.cell_value(i, 4)
#  news = remove_non_ascii_1(news)
#  title = remove_non_ascii_1(title)
#  #sentences = news.split('.')
#  #sentences2 = title.split('.')
#  scores = sentiment_analyzer_scores(news)
#  #print fake_cls, fake_class, pos_score, neg_score
#  out_file4.write(uid + ',' + fake_cls + ',' + str(fake_class)+','+str(scores['pos'])+','+str(scores['neu'])+','+str(scores['neg'])+','+str(scores['compound'])+os.linesep)
#  scores2 = sentiment_analyzer_scores(title)
#  #print fake_cls, fake_class, pos_score, neg_score, pos_score2, neg_score2, str(ddd), sheet.nrows
#  out_file4.write(uid + ',' + fake_cls + ',' + str(fake_class)+','+str(scores2['pos'])+','+str(scores2['neu'])+','+str(scores2['neg'])+','+str(scores2['compound'])+os.linesep)
#out_file4.close()
#out_file5.close()

