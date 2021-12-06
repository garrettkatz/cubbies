import numpy as np
from constructor import Constructor

def mcbt(σ, evaluate, max_rules, alg, max_actions, γ, ema_threshold, domain, scrambler):

    mdb_best = Constructor.init_macro_database(domain, max_rules=max_rules)
    con_best = Constructor(alg, max_actions, γ, ema_threshold)
    σ_best = -np.inf
    backtracks = 0

    history = []
    mdb, con = mdb_best, con_best

    while backtracks < mdb_best.num_rules:

        if len(history) > 0:
            mdb = mdb_best.copy().rewind(con_best.num_incs - backtracks)
            con = con_best.copy().rewind(con_best.num_incs - backtracks)
            scrambler.rewind(con_best.num_incs - backtracks)

        success = con.run_passes(mdb, scrambler)
        y = evaluate(mdb, alg)
        history.append((y, σ(y)))

        if σ(y) <= σ_best:
            backtracks += 1
        else:
            mdb_best, con_best, σ_best = mdb, con, σ(y)
            backtracks = 1

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
        folkliness = 1 - mdb.num_rules / tree.size()

        return godliness, folkliness

    return evaluate

def σ_factory():

    # normalized weights on positive unit sphere
    weights = np.random.normal(size=2)
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

    γ = 0.9
    ema_threshold = 1.0
    max_depth = 1
    max_actions = 20
    color_neutral = False

    num_repetitions = 100

    cube_str = "s120"
    # cube_str = "s5040"
    # cube_str = "s29k"
    # cube_str = "pocket"
    cube_size, valid_actions, tree_depth = CubeDomain.parameters(cube_str)

    dump_dir = "mcb"
    dump_base = "N%d%s_D%d_M%d_cn%d" % (
        cube_size, cube_str, tree_depth, max_depth, color_neutral)

    if do_cons:

        if not os.path.exists(dump_dir): os.mkdir(dump_dir)

        print("making states...")
        from tree import SearchTree
        domain = CubeDomain(cube_size, valid_actions)
        tree = SearchTree(domain, tree_depth)
        assert tree.depth() == tree_depth
        max_rules = tree.size()

        print("done. running search...")

        from algorithm import Algorithm
        from scrambler import AllScrambler

        bfs_tree = SearchTree(domain, max_depth)
        alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)

        num_problems = 32
        evaluate = evaluate_factory(max_actions, domain, tree, num_problems)

        fewest_steps = np.inf
        fewest_rules = np.inf

        for rep in range(num_repetitions):

            scrambler = AllScrambler(domain, tree)

            σ, weights = σ_factory()
            mdb_best, history = mcbt(
                σ, evaluate, max_rules, alg, max_actions, γ, ema_threshold, domain, scrambler)

            y, σ_y = zip(*history)
            y = np.array(y)
            best = np.argmax(σ_y)
            steps = 1/y[:,0]
            rules = (1 - y[:,1])*tree.size()

            fewest_steps = min(fewest_steps, steps.min())
            fewest_rules = min(fewest_rules, rules.min())

            # status update
            print(f"{rep} of {num_repetitions}: σ(y)={σ_y[best]}, s,r={steps[best],rules[best]} >= {fewest_steps},{fewest_rules}")

            with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "wb") as f:
                pk.dump((mdb_best, history), f)
