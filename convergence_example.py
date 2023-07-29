from cube import CubeDomain
domain = CubeDomain(2)

import pickle as pk

# dump_base = "N2pocket_D11_M1_cn0_0"
# dump_base = "N2s29k_D14_M1_cn0_0"
dump_base = "N2s120_D11_M1_cn0_9"

with open(f"rce/{dump_base}_mdb.pkl", "rb") as f:
    mdb = pk.load(f)
with open(f"rce/{dump_base}_con.pkl", "rb") as f:
    con = pk.load(f)[0]

import numpy as np

num_rule = np.empty(len(con.augment_incs)+1, dtype=int)
num_wild = np.empty(len(con.augment_incs)+1, dtype=int)

for i,inc in enumerate(con.augment_incs):
    num_rule[i] = con.rule_counts[inc]
    num_wild[i] = (mdb.tamed[:num_rule[i]] > inc).sum()

num_rule[-1] = num_rule[-2]
num_wild[-1] = num_wild[-2]

I = (num_rule == num_rule[-1]).argmax()
print(mdb.wildcards[:2,:])
print(mdb.tamed[:2,:])
print(mdb.wildcards[-2:,:])
print(mdb.tamed[-2:,:])
print(num_wild[-10:])
print(num_rule[-10:])
print(len(num_rule))
print(I)

import matplotlib.pyplot as pt

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 18
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\boldmath'

pt.subplot(2,1,1)
pt.plot(con.augment_incs + [con.num_incs], num_rule, 'k-')
pt.ylabel("Rule count")
pt.subplot(2,1,2)
pt.plot(con.augment_incs + [con.num_incs], num_wild, 'k-')
pt.plot(con.augment_incs + [con.num_incs], num_rule * domain.state_size(), 'k:')
# pt.xlim([con.augment_incs[I], con.num_incs])
pt.ylabel("Wildcards | Total")
pt.xlabel("Incorporations")
pt.tight_layout()
pt.show()


