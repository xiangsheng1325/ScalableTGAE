import numpy as np
import re

def not_empty(s):
	return len(s) > 0

f = open('/home/xuchenhao/graph_generation/meeting/220114/data/DBLP/edgelist_new.txt', 'r')
s = f.read()
f.close()

l = re.split('\s', s)
l = list(filter(not_empty, l))

mat = np.zeros((15, 1909, 1909))
# for i in range(15):
# 	for j in range(1909):
# 		mat[i][j][j] = 1.
for i in range(int(len(l) / 3)):
	# 有向图
	# for j in range(int(l[3 * i + 2]), 15):
	# 	mat[j][int(l[3 * i])][int(l[3 * i + 1])] = 1.
	# 	mat[j][int(l[3 * i + 1])][int(l[3 * i])] = 1.
	mat[int(l[3 * i + 2])][int(l[3 * i])][int(l[3 * i + 1])] = 1.

np.save('/home/xuchenhao/graph_generation/meeting/220125/data/DBLP/DBLP_evolve.npy', mat)
