from cube import CubeDomain
domain = CubeDomain(2)

import matplotlib.pyplot as pt
import matplotlib.patches as mp

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 18
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\boldmath'

actions = [
    (0,1,1),
    (1,1,1),
    (0,1,1),
    (2,1,1),
    (1,1,1),
    (0,1,1),
    (2,1,1),
]
state = domain.execute(actions, domain.solved_state())
states = domain.intermediate_states(domain.reverse(actions), state)

pt.figure(figsize=(6, 3))
ax = pt.gca()

for s, state in enumerate(states):

    domain.render(state, ax, x0=s*9, y0=0, text=False)
    if s + 1 < len(states):
        ax.add_patch(mp.FancyArrow(
            s*9 + 3, 0, 3, 0, color='k', length_includes_head=True,
            head_width=1, head_length=1,
            alpha = 1))

s = 2
for a,action in [(-5, (1,1,1)), (5, (2,1,1))]:
    domain.render(domain.perform(action, states[s]), ax, x0=(s+1)*9 - 3, y0=a, text=False)
    ax.add_patch(mp.Circle(
        ((s+1)*9 - 3, a), 3, color='w', alpha = .75))
    ax.add_patch(mp.FancyArrow(
        s*9 + 3, 0, 2, .6*a, color='k', length_includes_head=True,
        head_width=1, head_length=1,
        alpha = .25))

pt.plot([0, 0], [8, 12], 'k--')
ax.text(7, 10, "$m_{r_0}$")
pt.plot([18, 18], [8, 12], 'k--')
ax.text(21.5, 10, r"$\textbf{p}'$")
pt.plot([27, 27], [8, 12], 'k--')
ax.text(39, 10, "$m_{r_1}$")
pt.plot([54, 54], [8, 12], 'k--')

ax.text(0, 3, "$s^{(0)}$")
# ax.text(17, 3, "$m_{r_0}(s^{(0)})$")
ax.text(26, 3, "$s'$")
ax.text(53, 3, "$s^*$")

import numpy as np
for n,s in enumerate([0, 3]):
    w = (np.random.rand(*states[s].shape) < .5)
    domain.render(states[s]*(1-w), ax, x0=s*9, y0=-10, text=False)
    ax.text(s*9 + 3, -11, r"$S_{r_%d} \vee W_{r_%d}$" % (n,n))

ax.axis("equal")
ax.axis('off')
pt.tight_layout()
pt.savefig("example_alg.pdf")
pt.show()
