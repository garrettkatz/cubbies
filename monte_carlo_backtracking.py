import numpy as np
from constructor import Constructor

def mcbt(σ, evaluate, max_rules, alg, max_actions, γ, ema_threshold, domain, scrambler, backtrack_delta=1, max_forks=None, verbose=False):

    mdb_best = Constructor.init_macro_database(domain, max_rules=max_rules)
    con_best = Constructor(alg, max_actions, γ, ema_threshold)
    σ_best = -np.inf
    num_backtracks = 0

    history = []
    mdb, con = mdb_best, con_best
    best, inc = 0, 0

    while num_backtracks < len(con_best.augment_incs):

        num_augment_incs = len(con_best.augment_incs)
        if len(history) > 0:
            rewind_index = num_augment_incs - num_backtracks
            inc = con_best.augment_incs[rewind_index]

            mdb = mdb_best.copy().rewind(inc)
            con = con_best.copy().rewind(inc)
            scrambler.rewind(inc)
        else:
            rewind_index = 0

        success = con.run_passes(mdb, scrambler)
        augments = tuple(con.augment_incs[rewind_index:])
        y, samples = evaluate(mdb, alg)
        σy = σ(y)
        history.append((num_backtracks, σy, y, samples, mdb.num_rules, inc, best, con.num_incs, augments))

        if σy <= σ_best:
            num_backtracks += backtrack_delta
        else:
            mdb_best, con_best, σ_best = mdb, con, σy
            if verbose: print(f" {len(history)}:{num_backtracks}/{num_augment_incs} {σ_best=} ({mdb_best.num_rules} rules)")
            num_backtracks = 1
            best = len(history) - 1

        if len(history) == max_forks: break

    if verbose: print(f" {len(history)}:{num_backtracks}/{num_augment_incs} {σ_best=} ({mdb_best.num_rules} rules)")
        
    return mdb_best, history

def evaluate_factory(max_actions, domain, tree, num_problems):

    def problem():
        idx = np.random.choice(tree.size())
        state = domain.solved_state()[tree.permutations()[idx]]
        return state

    def evaluate(mdb, alg):
        solved = np.empty(num_problems)
        length = np.empty(num_problems)
        for p in range(num_problems):
            state = problem()
            solved[p], plan, _, _ = alg.run(max_actions, mdb, state)
            length[p] = sum([
                len(actions) + len(macro)
                for _, actions, macro in plan])

        # godliness = np.where(solved, 1 / np.maximum(1, length), 0).mean()
        godliness = 1 - np.where(solved, length, max_actions).mean() / max_actions
        folksiness = 1 - mdb.num_rules / tree.size()

        return (godliness, folksiness), (solved, length)

    return evaluate

def σ_factory():

    # normalized weights on positive unit sphere
    weights = np.random.normal(size=2)
    # weights = np.array([0.01, 0.99]) # folksiness
    weights = np.fabs(weights)
    weights /= np.sqrt(np.sum(weights**2))

    def σ_HV(objectives):
        return np.maximum(0, np.array(objectives) / weights).min(axis=-1) ** 2

    return σ_HV, weights

if __name__ == "__main__":

    import os
    import pickle as pk
    from cube import CubeDomain

    do_cons = False
    showresults = True
    confirm = False

    γ = 0.9
    ema_threshold = 1.0
    max_depth = 1
    color_neutral = False

    cube_str = "s120"
    max_forks = 300
    backtrack_delta = 1
    num_repetitions = 20
    max_actions = 30

    # cube_str = "s5040"
    # max_forks = 256
    # backtrack_delta = 32
    # num_repetitions = 128
    # max_actions = 30

    # cube_str = "s29k"
    # max_forks = 256
    # backtrack_delta = 32
    # num_repetitions = 16
    # max_actions = 50

    # cube_str = "pocket"
    cube_size, valid_actions, tree_depth = CubeDomain.parameters(cube_str)

    dump_dir = "mcb"
    dump_base = "N%d%s_D%d_M%d_cn%d_mf%d_bd%d" % (
        cube_size, cube_str, tree_depth, max_depth, color_neutral, max_forks, backtrack_delta)

    if do_cons or confirm:

        print("making states...")
        from tree import SearchTree
        domain = CubeDomain(cube_size, valid_actions)
        tree = SearchTree(domain, tree_depth)
        assert tree.depth() == tree_depth
        max_rules = tree.size()

        print("done.")

        from algorithm import Algorithm

        bfs_tree = SearchTree(domain, max_depth)
        alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)

    if do_cons:

        if not os.path.exists(dump_dir): os.mkdir(dump_dir)

        num_problems = 120
        evaluate = evaluate_factory(max_actions, domain, tree, num_problems)

        fewest_steps = np.inf
        fewest_rules = np.inf

        from time import perf_counter
        from scramblers import AllScrambler

        for rep in range(num_repetitions):

            start = perf_counter()

            scrambler = AllScrambler(domain, tree)

            σ, weights = σ_factory()
            mdb_best, history = mcbt(σ, evaluate, max_rules, alg, max_actions, γ, ema_threshold, domain, scrambler, backtrack_delta, max_forks, verbose=True)

            (num_backtracks, σy, y, samples, num_rules, fork_incs, best_forks, num_incs, augments) = zip(*history)
            y = np.array(y)
            best = np.argmax(σy)
            # steps = 1/y[:,0]
            steps = (1 - y[:,0])*max_actions
            rules = (1 - y[:,1])*tree.size()

            fewest_steps = min(fewest_steps, steps.min())
            fewest_rules = min(fewest_rules, rules.min())

            total_time = perf_counter() - start

            # status update
            print(f"{rep} of {num_repetitions} ({total_time:.2f}s): σ(y)={σy[best]}, s,r={steps[best]:.3f},{rules[best]:.0f} >= {fewest_steps:.3f},{fewest_rules:.0f}")

            mdb_best = mdb_best.shrink_wrap()
            with open(os.path.join(dump_dir, dump_base + f"_{rep}_mdb.pkl"), "wb") as f: pk.dump(mdb_best, f)
            with open(os.path.join(dump_dir, dump_base + f"_{rep}_hst.pkl"), "wb") as f: pk.dump((history, weights, tree.size(), total_time), f)

    if confirm:

        folksy_mdb = None
        for rep in range(num_repetitions):
            print(f"loading rep {rep}")
            with open(os.path.join(dump_dir, dump_base + f"_{rep}_mdb.pkl"), "rb") as f: mdb_best = pk.load(f)
            if folksy_mdb is None or mdb_best.num_rules < folksy_mdb.num_rules: folksy_mdb = mdb_best
        print("folksy num_rules:", folksy_mdb.num_rules)

        print("confirming...")
        correct = True
        for i in range(tree.size()):
            state = domain.solved_state()[tree.permutations()[i]]
            success, _, _, _ = alg.run(max_actions, folksy_mdb, state)
            correct = correct and success
        print(f"correct = {correct}")

        print("folksy num_rules:", folksy_mdb.num_rules)

    if showresults:

        import matplotlib.pyplot as pt
        from matplotlib import rcParams
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 11
        rcParams['text.usetex'] = True

        histories = []
        for rep in range(num_repetitions):
            print(f"loading rep {rep}")
            fname = os.path.join(dump_dir, dump_base + f"_{rep}_hst.pkl")
            if not os.path.exists(fname): continue
            with open(fname, "rb") as f:
                (history, weights, tree_size, total_time) = pk.load(f)
            histories.append(history)
            # if rep == 50: break

        # objective space
        pt.figure(figsize=(3.5, 3.25))
        bests = []
        for rep in range(len(histories)):
            (num_backtracks, σy, y, samples, rule_counts, fork_incs, best_forks, num_incs, augments) = zip(*histories[rep])
            num_rules, avg_length = [], []
            for s in range(len(samples)):
                num_rules.append(rule_counts[s])
                solved, length = samples[s]
                avg_length.append(np.where(solved, length, max_actions).mean())
            best = np.argmax(σy)
            bests.append((num_rules[best], avg_length[best]))
            # pt.subplot(1,2,1)
            pt.plot(num_rules, avg_length, 'o', color=(.5,)*3)
            # pt.subplot(1,2,2)
            # y = np.array(y)
            # pt.plot(y[:,1], y[:,0], 'o', color=(.5,)*3)
            # pt.plot(y[best,1], y[best,0], 'o', color=(0,)*3)
        num_rules, avg_length = zip(*bests)
        # pt.subplot(1,2,1)
        pt.plot(num_rules, avg_length, 'o', color=(0,)*3)
        pt.xlabel("Rule count")
        pt.ylabel("Avg. Soln. Length")
        # pt.subplot(1,2,2)
        # pt.xlabel("Folksiness")
        # pt.ylabel("Godliness")
        pt.tight_layout()
        pt.savefig(f"ftb_{cube_str}_pareto.pdf")
        pt.show()
        pt.close()

        # # scalarization curves
        # initials, finals = [], []
        # for rep in range(len(histories)):
        #     mdb_best = mdb_bests[rep]
        #     history = histories[rep]

        #     num_backtracks, y, σy = zip(*history)
        #     y = np.array(y)
        #     σy = np.array(σy)

        #     w = 10
        #     ma = σy.cumsum()
        #     ma = (ma[w:] - ma[:-w])/w

        #     pt.subplot(1,4,1)
        #     # pt.plot(ma)
        #     # pt.ylabel("moving average σ")
        #     pt.plot(σy)
        #     pt.ylabel("σ")
        #     pt.xlabel("fork")

        #     pt.subplot(1,4,2)
        #     pt.plot(num_backtracks)
        #     pt.ylabel("num backtracks")
        #     pt.xlabel("fork")

        #     initials.append(σy[0])
        #     finals.append(σy.max())

        # pt.subplot(1,4,3)
        # pt.plot(initials, finals, 'k.')
        # pt.xlabel("initial σ")
        # pt.ylabel("best σ")

        # pt.subplot(1,4,4)
        # pt.hist((initials, finals))
        # pt.xlabel("σ")
        # pt.ylabel("freq")
        # pt.legend(["initial", "best"])

        # pt.show()

        # # scalarization tree
        # rep = 0
        # with open(os.path.join(dump_dir, dump_base + f"_{rep}_hst.pkl"), "rb") as f:
        #     (history, weights, tree_size, total_time) = pk.load(f)

        # (num_backtracks, σy, y, samples, rule_counts, fork_inc, best_fork, num_incs, augments) = zip(*history)

        # for n in range(len(history)):
        #     i = best_fork[n]
        #     while fork_inc[n] < fork_inc[i]: i = best_fork[i]
        #     pt.plot(
        #         [fork_inc[i], fork_inc[i], num_incs[n]],
        #         [σy[i], σy[n], σy[n]],
        #         '-', color=(.75 - .5 * n / len(history),)*3)
        #         # '-', color=(.75 * n / len(leaves),)*3)
        #         # '-', color=(.5,)*3)

        # n = np.argmax(σy)
        # inc = num_incs[n]
        # while True:
        #     i = best_fork[n]
        #     while fork_inc[n] < fork_inc[i]: i = best_fork[i]
        #     pt.plot(
        #         [fork_inc[i], fork_inc[i], inc],
        #         [σy[i], σy[n], σy[n]],
        #         '-', color=(.0,)*3)
        #     if n == 0: break
        #     n = i
        #     inc = fork_inc[i]

        # pt.xlabel("Number of modifications")
        # pt.ylabel("Scalarization value")

        # # pt.savefig("ftb_%s.pdf" % dump_name)
        # pt.show()

        # augment tree
        fig = pt.figure(figsize=(3.5, 5))
        gs = fig.add_gridspec(1,3)
        rep = 0
        with open(os.path.join(dump_dir, dump_base + f"_{rep}_hst.pkl"), "rb") as f:
            (history, weights, tree_size, total_time) = pk.load(f)

        (num_backtracks, σy, y, samples, rule_counts, fork_inc, best_fork, num_incs, augments) = zip(*history)

        # pt.subplot(1,2,1)
        ax = fig.add_subplot(gs[0,:2])
        for n in range(len(history)):
            # color = (.5,)*3
            color = (0,)*3
            pt.plot(
                [fork_inc[n], fork_inc[n], num_incs[n]],
                [best_fork[n], n, n],
                '-', color=color)
            pt.plot(augments[n], [n]*len(augments[n]), '+', color=color)

        n = np.argmax(σy)
        pt.plot(num_incs[n], n, 'ko')

        pt.ylim([-1, len(history)])
        pt.xlabel("Incorporations")
        pt.ylabel("Forks")

        # pt.subplot(1,2,2)
        ax = fig.add_subplot(gs[0,2])
        # pt.barh(np.arange(len(history)), σy, height=.25, ec='k', fc='k')
        pt.plot(σy, np.arange(len(history)), 'k-')
        pt.yticks([], [])
        pt.ylim([-1, len(history)])
        # pt.xlabel("σ(y)")
        pt.xlabel("$\sigma(y)$")

        pt.tight_layout()
        pt.savefig(f"ftb_{cube_str}_mcbt.pdf")
        pt.show()

        
    
