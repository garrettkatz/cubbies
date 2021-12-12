import numpy as np
from constructor import Constructor

def mcbt(σ, evaluate, max_rules, alg, max_actions, γ, ema_threshold, domain, scrambler, verbose=False):

    mdb_best = Constructor.init_macro_database(domain, max_rules=max_rules)
    con_best = Constructor(alg, max_actions, γ, ema_threshold)
    σ_best = -np.inf
    num_backtracks = 0

    history = []
    mdb, con = mdb_best, con_best

    while num_backtracks < len(con_best.augment_incs):

        num_augment_incs = len(con_best.augment_incs)
        if len(history) > 0:
            rewind_index = num_augment_incs - num_backtracks
            inc = con_best.augment_incs[rewind_index]

            mdb = mdb_best.copy().rewind(inc)
            con = con_best.copy().rewind(inc)
            scrambler.rewind(inc)

        success = con.run_passes(mdb, scrambler)
        y = evaluate(mdb, alg)
        history.append((num_backtracks, y, σ(y)))

        if σ(y) <= σ_best:
            num_backtracks += 1
        else:
            mdb_best, con_best, σ_best = mdb, con, σ(y)
            if verbose: print(f" {len(history)}:{num_backtracks}/{num_augment_incs} {σ_best=} ({mdb_best.num_rules} rules)")
            num_backtracks = 1

    return mdb_best, history

def evaluate_factory(max_actions, domain, tree, num_problems):

    def problem():
        idx = np.random.choice(tree.size())
        state = domain.solved_state()[tree.permutations()[idx]]
        return state

    def evaluate(mdb, alg):
        godliness = []

        for p in range(num_problems):
            state = problem()
            solved, plan, _, _ = alg.run(max_actions, mdb, state)
            solution_length = sum([
                len(actions) + len(macro)
                for _, actions, macro in plan])
            godliness.append(0 if not solved else 1 / max(1, solution_length))

        godliness = np.mean(godliness)
        folksiness = 1 - mdb.num_rules / tree.size()

        return godliness, folksiness

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

    do_cons = True
    showresults = True
    confirm = False

    γ = 0.9
    ema_threshold = 1.0
    max_depth = 1
    max_actions = 20
    color_neutral = False

    num_repetitions = 1

    cube_str = "s120"
    # cube_str = "s5040"
    # cube_str = "s29k"
    # cube_str = "pocket"
    cube_size, valid_actions, tree_depth = CubeDomain.parameters(cube_str)

    dump_dir = "mcb"
    dump_base = "N%d%s_D%d_M%d_cn%d" % (
        cube_size, cube_str, tree_depth, max_depth, color_neutral)

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

        num_problems = 64
        evaluate = evaluate_factory(max_actions, domain, tree, num_problems)

        fewest_steps = np.inf
        fewest_rules = np.inf

        from scramblers import AllScrambler

        for rep in range(num_repetitions):

            scrambler = AllScrambler(domain, tree)

            σ, weights = σ_factory()
            mdb_best, history = mcbt(
                σ, evaluate, max_rules, alg, max_actions, γ, ema_threshold, domain, scrambler,
                verbose=True)

            num_backtracks, y, σ_y = zip(*history)
            y = np.array(y)
            best = np.argmax(σ_y)
            steps = 1/y[:,0]
            rules = (1 - y[:,1])*tree.size()

            fewest_steps = min(fewest_steps, steps.min())
            fewest_rules = min(fewest_rules, rules.min())

            # status update
            print(f"{rep} of {num_repetitions}: σ(y)={σ_y[best]}, s,r={steps[best]:.3f},{rules[best]:.0f} >= {fewest_steps:.3f},{fewest_rules:.0f}")

            with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "wb") as f:
                pk.dump((mdb_best, history), f)

    if showresults:

        import matplotlib.pyplot as pt

        # scalarization curves
        mdb_bests, histories = [], []
        for rep in range(num_repetitions):
            print(rep)
            with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "rb") as f:
                (mdb_best, history) = pk.load(f)
            mdb_bests.append(mdb_best)
            histories.append(history)
            if rep == 50: break

        initials, finals = [], []
        for rep in range(len(histories)):
            mdb_best = mdb_bests[rep]
            history = histories[rep]

            num_backtracks, y, σ_y = zip(*history)
            y = np.array(y)
            σ_y = np.array(σ_y)

            w = 10
            ma = σ_y.cumsum()
            ma = (ma[w:] - ma[:-w])/w

            pt.subplot(1,4,1)
            # pt.plot(ma)
            # pt.ylabel("moving average σ")
            pt.plot(σ_y)
            pt.ylabel("σ")
            pt.xlabel("fork")

            pt.subplot(1,4,2)
            pt.plot(num_backtracks)
            pt.ylabel("num backtracks")
            pt.xlabel("fork")

            initials.append(σ_y[0])
            finals.append(σ_y.max())

        pt.subplot(1,4,3)
        pt.plot(initials, finals, 'k.')
        pt.xlabel("initial σ")
        pt.ylabel("best σ")

        pt.subplot(1,4,4)
        pt.hist((initials, finals))
        pt.xlabel("σ")
        pt.ylabel("freq")
        pt.legend(["initial", "best"])

        pt.show()

        # # scalarization tree
        # rep = 0
        # with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "rb") as f:
        #     (mdb_best, history) = pk.load(f)

        # num_backtracks, y, σ_y = zip(*history)
        # y = np.array(y)
        # σ_y = np.array(σ_y)

        # local_best = [0]
        # for i in range(1, len(σ_y)):
        #     if σ_y[i] > local_best[-1]:
        #         local_best.append(σ_y[i])
        
    if confirm:

        folksy_mdb = None
        folksy_σ = 0
        for rep in range(num_repetitions):
            print(rep)
            with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "rb") as f:
                (mdb_best, history) = pk.load(f)
            num_backtracks, y, σ_y = zip(*history)
            if folksy_σ < max(σ_y):
                folksy_σ = max(σ_y)
                folksy_mdb = mdb_best

        print("num_rules:", folksy_mdb.num_rules)
        
        for i in range(tree.size()):
            state = domain.solved_state()[tree.permutations()[i]]
            success, plan, rule_indices, triggerers = alg.run(max_actions, folksy_mdb, state)
            assert success
        print("success!")
