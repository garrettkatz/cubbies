from cube import CubeDomain
domain = CubeDomain(2)

from tree import SearchTree
tree = SearchTree(domain, max_depth=4)

action_map = {v:k for k,v in domain.get_action_map().items()}

import matplotlib.pyplot as pt
import matplotlib.patches as mp

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 18
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\boldmath'

pairs = list(tree.rooted_at(domain.solved_state()))[::(tree.size()//6)]

# fig, axs = pt.subplots(len(pairs), 1, figsize=(4,8))
# for s, (path, state) in enumerate(pairs):
#     domain.render(state, axs[s], x0=0, y0=0, text=False)
#     axs[s].text(3, -.75, ",".join([action_map[a] for a in domain.reverse(path)]))
#     axs[s].axis("equal")
#     axs[s].axis('off')

fig = pt.figure(figsize=(4,8))
ax = pt.gca()
for r,(path, state) in enumerate(pairs):
    domain.render(state, ax, x0=0, y0=-4*r, text=False)
    pt.text(3, -4*r-.75, ",".join([action_map[a] for a in domain.reverse(path)]))

ax.axis("equal")
ax.axis('off')

# pt.tight_layout()
pt.savefig("god_alg.png")
pt.show()

