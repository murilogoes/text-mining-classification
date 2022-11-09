import csv
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np

count_0 =0
count_1 =0

list_0 =[]
list_1 =[]

documents = []
labels    = []

with open('data/bos2.csv','r') as csvfile:
    next(csvfile)
    reader = csv.reader(csvfile, delimiter =',')
    for row in reader:
        if int(row[1] )==0:
            count_0+=1
            list_0.append(row[0])
        else:
            count_1+=1
            list_1.append(row[0])

        documents.append(row[0])
        labels.append(row[1])

    print(str(count_0) + " " + str(count_1))
objects = ('Non-police Context', 'Police Context')
y_pos = np.arange(len(objects))
performance = [count_0, count_1]

plt.bar(y_pos, performance, align='center',alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Amount')
plt.title('Amount per Class')

plt.show()