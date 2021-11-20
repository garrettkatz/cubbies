import os
import pickle as pk
import numpy as np
from cube import CubeDomain
from tree import SearchTree
from macro_database import MacroDatabase
from algorithm import Algorithm
from constructor import Constructor

if __name__ == "__main__":

    do_cons = True
    showresults = True

    γ = 0.9
    ema_threshold = 0.9999
    max_depth = 1
    max_actions = 20
    color_neutral = False

    num_repetitions = 30
    num_restarts = 2**11

    # cube_str = "s120"
    cube_str = "s5040"
    # cube_str = "s29k"
    # cube_str = "pocket"
    cube_size, valid_actions, tree_depth = CubeDomain.parameters(cube_str)

    dump_dir = "lvo"
    dump_base = "N%d%s_D%d_M%d_cn%d" % (
        cube_size, cube_str, tree_depth, max_depth, color_neutral)

    if do_cons:

        if not os.path.exists(dump_dir): os.mkdir(dump_dir)

        print("making states...")
        domain = CubeDomain(cube_size, valid_actions)
        tree = SearchTree(domain, tree_depth)
        assert tree.depth() == tree_depth
        print("done. running restarts...")
        
        bfs_tree = SearchTree(domain, max_depth)
        all_scrambles = Constructor.make_all_scrambles(domain, tree)
        alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)

        for rep in range(num_repetitions):
            rule_caps = [1,1]
            for restart in range(num_restarts):
    
                if restart >= len(rule_caps):
                    rule_caps.append(2 * rule_caps[-1])
                    rule_caps += rule_caps
                # max_rules = 2 * rule_caps[restart]
                max_rules = rule_caps[restart]
    
                print(f"({rep},{restart}) of ({num_repetitions},{num_restarts}): {max_rules=}")
                    
                mdb = Constructor.init_macro_database(domain, max_rules=max_rules)
                con = Constructor(alg, max_actions, γ, ema_threshold)
                success = con.run_passes(mdb, all_scrambles)
    
                if success:
                    with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "wb") as f:
                        pk.dump((con, mdb, restart, rule_caps), f)
                    print(f"success: {mdb.num_rules=}")
                    break

    if showresults:

        import matplotlib.pyplot as pt

        num_rules = []
        for rep in range(num_repetitions):
            with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "rb") as f:
                (con, mdb, restart, rule_caps) = pk.load(f)
                print(mdb.num_rules)
                num_rules.append(mdb.num_rules)
        print(np.mean(num_rules))
        print(np.std(num_rules))

