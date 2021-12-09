import os
import pickle as pk
import numpy as np
from cube import CubeDomain

if __name__ == "__main__":

    do_cons = True
    showresults = True

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

    max_runs = 2**32
    branch_factor = 2

    dump_dir = "bnb"
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
        from constructor import Constructor
        from scramblers import AllScrambler

        bfs_tree = SearchTree(domain, max_depth)
        alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)

        for rep in range(num_repetitions):
    
            scrambler = AllScrambler(domain, tree)
            mdb = Constructor.init_macro_database(domain, max_rules=max_rules)
            con = Constructor(alg, max_actions, γ, ema_threshold)

            rule_bound = max_rules
            branch_counters = []
            branch_point = 0
            mdb_best = None

            bound_history = []
            success_history = []
            branch_history = []

            for run in range(max_runs):

                # run branch to termination
                mdb.max_rules = rule_bound
                success = con.run_passes(mdb, scrambler)

                # save if new best
                if success and mdb.num_rules < rule_bound:
                    rule_bound = mdb.num_rules
                    mdb_best = mdb.copy()

                # status update
                print(f"({rep},{run}) of ({num_repetitions},{max_runs}): {success=}, {rule_bound=}, {branch_point=}")

                # set counters in new branch
                branch_counters += [0] * (len(con.augment_incs) - 1 - branch_point)
                # print(branch_counters)

                # update branch counters and branch point
                for b in reversed(range(len(branch_counters))):
                    branch_counters[b] += 1
                    if branch_counters[b] < branch_factor: break
                    branch_counters[b] = 0
                if b == 0 and branch_counters[0] == 0: break
                branch_point = b

                # start new branch for next run
                branch_counters = branch_counters[:branch_point+1]
                mdb.rewind(con.augment_incs[branch_point])
                con.rewind(con.augment_incs[branch_point])
                scrambler.rewind(con.augment_incs[branch_point])

                bound_history.append(rule_bound)
                success_history.append(success)
                branch_history.append(branch_point)

            with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "wb") as f:
                pk.dump((mdb_best, bound_history, success_history, branch_history), f)

    if showresults:

        import matplotlib.pyplot as pt

        num_rules = []
        for rep in range(num_repetitions):
            with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "rb") as f:
                (mdb, bounds, success, branches) = pk.load(f)
            
            pt.subplot(2, num_repetitions, rep+1)
            pt.plot(bounds)
            pt.subplot(2, num_repetitions, num_repetitions + rep+1)
            pt.plot(branches)
        pt.show()



