import pandas as pd

s1 = "data/DBLP/DBLP_raw_param_undir_selfloop.csv"
s2 = "/home/xuchenhao/graph_generation/experiment/TGAE/tables/DBLP_steps=120_eo=0.99_lr=0.05_H=128_s.csv"
o  = "/home/xuchenhao/graph_generation/experiment/TGAE/results/DBLP_steps=120_eo=0.99_lr=0.05_H=128_s.csv"

d1 = pd.read_csv(s1).drop(columns=["Unnamed: 0"])
d2 = pd.read_csv(s2).drop(columns=["Unnamed: 0"])
d3 = abs(d1 - d2) / d1
s1 = d3.mean()
s2 = d3.median()
s1.name = 'mean'
s2.name = 'median'
d3 = d3.append(s1)
d3 = d3.append(s2)

d3.to_csv(o)
