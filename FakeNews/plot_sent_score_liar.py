import matplotlib.pyplot as plt

res_dict = {}
res_dict2 = {}

with open('results_liar.csv', 'r') as f:
  #first_line = f.readline()
  i = 0
  for line in f:
    line_list = line.split(',')
    #print line_list
    #values = [float(s) for s in line.split(',')]

    fake_cls = line_list[0]
    pos_sent = float(line_list[1])
    neg_sent = float(line_list[3])
    neu_sent = float(line_list[2])
    com_score = float(line_list[4])


    #pos_sent = float(line_list[1])
    #neg_sent = float(line_list[3])
    #neu_sent = float(line_list[2])
    #com_score = float(line_list[4][0:6])
    if fake_cls not in res_dict:
      res_dict[fake_cls] = [pos_sent]
    else:
      res_dict[fake_cls] += [pos_sent]
    if fake_cls not in res_dict2:
      res_dict2[fake_cls] = [neg_sent]
    else:
      res_dict2[fake_cls] += [neg_sent]

#ticks = [x+1 for x in range(len(res_dict))]
labels = []
data = []
for rr in res_dict:
  #print rr
  if rr == 'half-true':
    labels.append('Reliable')
    data+=res_dict[rr]
  elif rr == 'mostly-true':
    labels.append('Reliable')
    data += res_dict[rr]
  elif rr == 'true':
    labels.append('Reliable')
    data += res_dict[rr]
  elif rr == 'false':
    labels.append('Unreliable')
    #data += res_dict[rr]
  elif rr == 'barely-true':
    labels.append('Unreliable')
    #data += res_dict[rr]
  elif rr == 'pants-fire':
    labels.append('Unreliable')
    #data += res_dict[rr]


#ticks2 = [x+1 for x in range(len(res_dict2))]
labels2 = []
data2 = []
for rr in res_dict2:
  #print rr
  if rr == 'half-true':
    labels2.append('Reliable')
    data2+=res_dict2[rr]
  elif rr == 'mostly-true':
    labels2.append('Reliable')
    data2 += res_dict2[rr]
  elif rr == 'true':
    labels2.append('Reliable')
    data2 += res_dict2[rr]
  elif rr == 'false':
    labels2.append('Unreliable')
    #data2 += res_dict2[rr]
  elif rr == 'barely-true':
    labels2.append('Unreliable')
    #data2 += res_dict2[rr]
  elif rr == 'pants-fire':
    labels2.append('Unreliable')
    #data2 += res_dict2[rr]

print(len(data))
data = data[0:4496]
data2 = data2[0:4496]
plt_data = [data, data2]
print(len(data), sum(data), len(data2), sum(data2))

green_diamond = dict(markerfacecolor='g', marker='D')
plt.boxplot(plt_data, flierprops=green_diamond) #,  showfliers=False) #, label='Cuckoo filter')
#plt.boxplot(Y2) #, label='Google-RAPPOR')
#plt.boxplot(Y3) #, label='Apple-CMS')
#plt.xlabel('Number of records queried')
plt.ylabel('Sentiment score')
#plt.yscale('log')
plt.title('Sentiment scores of legitimate news')
plt.xticks([1, 2], ['pos', 'neg'])
#plt.ylim(ymax=1.0)
#plt.xticks(rotation=25)
plt.legend(loc='lower left')
plt.savefig('sentiment_liar_real.eps')
plt.show()

#red_square = dict(markerfacecolor='r', marker='s')
#plt.boxplot(data2, flierprops=red_square) #,  showfliers=False) #, #label='Cuckoo filter')
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
#plt.savefig('neg_sentiment_large_newstitle_data.eps')
#plt.show()
