from asyncore import write
from dataclasses import field
import re
import csv

def not_empty(s):
	return len(s) > 0

f = open('/home/xuchenhao/graph_generation/meeting/220125/data/DBLP/edgelist.txt', 'r')
s = f.read()
f.close()
l = re.split('\s', s)
l = list(filter(not_empty, l))

max = 0
for i in range(len(l)):
	if (int(l[i]) > max):
		max = int(l[i])

index = [0 for _ in range(max + 1)]
for i in range(len(l)):
	if (i % 3 != 2):
		index[int(l[i])] = 1

counter = 0
for i in range(len(index)):
	if (index[i] > 0):
		index[i] = counter
		counter += 1

f_new = open('/home/xuchenhao/graph_generation/meeting/220125/data/DBLP/edgelist_new.txt', 'w')
for i in range(len(l)):
	if (i % 3 != 2):
		f_new.write(str(index[int(l[i])]))
		f_new.write(' ')
	else:
		f_new.write(str(int(l[i])-1))
		f_new.write('\n')

f_new.close()

f_csv = open('/home/xuchenhao/graph_generation/meeting/220125/data/DBLP/dblp_15t_1909n_adj.csv', 'w', newline='')
field_names = ['source', 'target', 'weight', 'time']
writer = csv.writer(f_csv)
writer.writerow(field_names)
row = []
for i in range(len(l)):
	if (i % 3 == 0 and i != 0):
		writer.writerow(row)
		writer.writerow(row[1:2] + row[:1] + row[2:])
		row.clear()
	if (i % 3 != 2):
		row.append(str(index[int(l[i])]))
	else:
		row.append('1')
		row.append(str(int(l[i])-1))

f_csv.close()