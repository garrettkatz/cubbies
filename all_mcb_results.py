import os, glob
import pickle as pk
import numpy as np
import matplotlib.pyplot as pt
from matplotlib import rcParams
from cube import CubeDomain

rcParams['font.family'] = 'serif'
rcParams['font.size'] = 11
rcParams['text.usetex'] = True

cube_strs = ["s5040", "s29k"]
max_actions = {"s5040": 30, "s29k": 50}
histories = {}
for cube_str in cube_strs:

    cube_size, _, tree_depth = CubeDomain.parameters(cube_str)

    max_depth = 1
    color_neutral = False
    max_forks = 256
    backtrack_delta = 32

    dump_dir = "mcb"
    dump_base = "N%d%s_D%d_M%d_cn%d_mf%d_bd%d" % (
        cube_size, cube_str, tree_depth, max_depth, color_neutral, max_forks, backtrack_delta)

    num_repetitions = len(glob.glob(os.path.join(dump_dir, dump_base + "_*_hst.pkl")))

    histories[cube_str] = []
    for rep in range(num_repetitions):
        print(f"loading rep {rep}")
        fname = os.path.join(dump_dir, dump_base + f"_{rep}_hst.pkl")
        if not os.path.exists(fname): continue
        with open(fname, "rb") as f:
            (history, weights, tree_size, total_time) = pk.load(f)
        histories[cube_str].append(history)

# objective space
pt.figure(figsize=(3.5, 2))
for c, cube_str in enumerate(cube_strs):
    pt.subplot(1,2,c+1)

    bests = []
    all_rules, all_lengths, all_colors = [], [], []
    for rep in range(len(histories[cube_str])):
        (num_backtracks, σy, y, samples, rule_counts, fork_incs, best_forks, num_incs, augments) = zip(*histories[cube_str][rep])
        num_rules, avg_length = [], []
        for s in range(len(samples)):
            num_rules.append(rule_counts[s])
            solved, length = samples[s]
            avg_length.append(np.where(solved, length, max_actions[cube_str]).mean())
        best = np.argmax(σy)
        bests.append((num_rules[best], avg_length[best]))
        colors = [(1 - σy[i] / σy[best],)*3 for i in range(len(σy))]
        all_rules += num_rules
        all_lengths += avg_length
        all_colors += colors

        if cube_str == "s5040" and rep % 6 != 0:  continue # sub-sample 5040 data for smaller image file size

        # pt.scatter(num_rules[::2], avg_length[::2], 20, marker='.', color=colors)
        pt.plot(num_rules[::2], avg_length[::2], '.', color=(.5,)*3)

    # # shaded by objective value
    # all_rules = np.array(all_rules)
    # all_lengths = np.array(all_lengths)
    # all_colors = np.array(all_colors)
    # idx = np.argsort(all_colors[:,0])[::-1]
    # all_rules = all_rules[idx]
    # all_lengths = all_lengths[idx]
    # all_colors = all_colors[idx]
    # buckets = np.linspace(0, len(all_colors), 10).astype(int)
    # for b in range(len(buckets)-1):
    #     pt.scatter(
    #         all_rules[buckets[b]:buckets[b+1]],
    #         all_lengths[buckets[b]:buckets[b+1]],
    #         20, marker='.',
    #         color=all_colors[buckets[b]:buckets[b+1]])

    print(f"{cube_str}: avg {np.mean(all_rules)},{np.mean(all_lengths)}, best {np.min(all_rules)},{np.min(all_lengths)}")
    print(f" relative improvement: {1 - np.min(all_rules)/np.mean(all_rules)}, {1 - np.min(all_lengths)/np.mean(all_lengths)}")

    num_rules, avg_length = zip(*bests)
    # pt.scatter(num_rules, avg_length, 20, marker='.', color='k')
    pt.plot(num_rules, avg_length, '.', color='k')
    pt.xlabel("Rule count")
    pt.title(cube_str[1:])
    if c == 0: pt.ylabel("Avg. Soln. Len.")

pt.tight_layout()
pt.savefig(f"mcb_pareto.pdf")
pt.show()
pt.close()
