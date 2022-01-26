import cube
domain = cube.CubeDomain(2)

import pickle as pk

# with open("rce/N2pocket_D11_M1_cn0_0_mdb.pkl", "rb") as f:
with open("rce/N2s29k_D14_M1_cn0_0_mdb.pkl", "rb") as f:
    mdb = pk.load(f)

action_map = {v:k for k,v in domain.get_action_map().items()}

import matplotlib.pyplot as pt
import matplotlib.patches as mp

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.size'] = 18
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = r'\boldmath'

# fig, axs = pt.subplots(6, 1, figsize=(4,8))
# for r in range(6):
#     domain.render(mdb.prototypes[r] * (1 - mdb.wildcards[r]), axs[r], x0=0, y0=0, text=False)
#     axs[r].text(3, -.75, ",".join([action_map[a] for a in mdb.macros[r]]))
#     axs[r].axis("equal")
#     axs[r].axis('off')

# r = 1
# domain.render(mdb.wildcards[r] * cube._K, axs[r], x0=-4, text=False)
# axs[r].axis("equal")
# axs[r].axis('off')

RA = list(range(6))
RB = [0,1,2,6,7,8]

for ab in (0,1):

    fig = pt.figure(figsize=(14,8))
    ax = pt.gca()
    for i,r in enumerate([RA, RB][ab]):
        domain.render(mdb.prototypes[r] * (1 - mdb.wildcards[r]), ax, x0=0, y0=-4*i, text=False)
        pt.text(3, -4*i-.75, ",".join([action_map[a] for a in mdb.macros[r]]))
        # domain.render((1 - mdb.wildcards[r]) * cube._K, ax, x0=-7, y0=-4*i, text=False)
        # domain.render(mdb.prototypes[r], ax, x0=-12, y0=-4*i, text=False)
    
    # pt.text(-12.5, 3, r"$S_{r}$")
    # pt.text(-7.5, 3, r"$W_{r}$")
    # pt.text(-2, 3, r"$S_{r} \vee W_{r}$")
    # pt.text(10, 3, r"$m_{r}$")
    
    ax.axis("equal")
    ax.axis('off')
    
    # pt.tight_layout()
    pt.savefig(f"rcons_grow_{ab}.png")
    pt.show()

