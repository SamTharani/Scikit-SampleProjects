import matplotlib.pyplot as plt

res_dict = {}
res_dict2 = {}

with open('results_fake_news_title_text_emotion.csv', 'r') as f:
  #first_line = f.readline()
  i = 0
  for line in f:
    line_list = line.split(',')
    #values = [float(s) for s in line.split(',')]
    fake_cls = line_list[-5]
    pos_sent = float(line_list[-4])
    neg_sent = float(line_list[-2])
    if fake_cls not in res_dict:
      res_dict[fake_cls] = [pos_sent]
    else:
      res_dict[fake_cls] += [pos_sent]
    if fake_cls not in res_dict2:
      res_dict2[fake_cls] = [neg_sent]
    else:
      res_dict2[fake_cls] += [neg_sent]

ticks = [x+1 for x in range(len(res_dict))]
labels = []
data = []
for rr in res_dict:
  labels.append(rr)
  if rr == '1':
    data += res_dict[rr]


ticks2 = [x+1 for x in range(len(res_dict2))]
labels2 = []
data2 = []
for rr in res_dict2:
  labels2.append(rr)
  if rr == '1':
    data2+=res_dict2[rr]

print(len(data))

plot_data = [data[0:167], data2[0:167]]

#print(len(res_dict['1']), len(res_dict['0']), len(res_dict2['1']), len(res_dict2['0']))

green_diamond = dict(markerfacecolor='g', marker='D')
plt.boxplot(plot_data, flierprops=green_diamond) #,  showfliers=False) #, label='Cuckoo filter')
#plt.boxplot(Y2) #, label='Google-RAPPOR')
#plt.boxplot(Y3) #, label='Apple-CMS')
#plt.xlabel('Number of records queried')
plt.ylabel('Sentiment score')
#plt.yscale('log')
plt.title('Sentiment scores of fake news title')
plt.xticks(ticks, ['pos', 'neg'])
plt.ylim(ymax=1.0)
#plt.xticks(rotation=25)
plt.legend(loc='lower left')
plt.savefig('pos_sentiment_smnews_news_fakenewstitle.eps')
plt.show()

#red_square = dict(markerfacecolor='r', marker='s')
#plt.boxplot(data2, flierprops=red_square) #,  showfliers=False) #, label='Cuckoo filter')
##plt.boxplot(Y2) #, label='Google-RAPPOR')
##plt.boxplot(Y3) #, label='Apple-CMS')
##plt.xlabel('Number of records queried')
#plt.ylabel('Negative sentiment score')
##plt.yscale('log')
#plt.title('Negative sentiment scores of news')
#plt.xticks(ticks2, labels2)
##plt.ylim(ymax=1.0)
##plt.xticks(rotation=25)
#plt.legend(loc='lower left')
#plt.savefig('neg_sentiment_smnews_news.eps')
#plt.show()
