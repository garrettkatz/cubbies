import os
import pickle as pk
from cube import CubeDomain
from tree import SearchTree
from macro_database import MacroDatabase
from algorithm import Algorithm
from constructor import Constructor

if __name__ == "__main__":

    do_cons = True
    showresults = False

    γ = 0.9
    ema_threshold = 0.9999
    max_depth = 1
    max_actions = 20
    color_neutral = False

    num_rollouts = 10000

    cube_str = "s120"
    # cube_str = "s5040"
    # cube_str = "s29k"
    # cube_str = "pocket"
    cube_size, valid_actions, tree_depth = CubeDomain.parameters(cube_str)

    dump_dir = "roe"
    dump_base = "N%d%s_D%d_M%d_cn%d" % (
        cube_size, cube_str, tree_depth, max_depth, color_neutral)

    if do_cons:

        print("making scaled up states...")
        domain = CubeDomain(cube_size, valid_actions)
        tree = SearchTree(domain, tree_depth)
        assert tree.depth() == tree_depth
        print("done. running rollouts...")
        
        bfs_tree = SearchTree(domain, max_depth)
        all_scrambles = Constructor.make_all_scrambles(domain, tree)
        alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)
    
        folksiness = []
    
        for r in range(num_rollouts):
    
            mdb = Constructor.init_macro_database(domain, max_rules=tree.size())
            cons = Constructor(alg, max_actions, γ, ema_threshold)
            cons.run_passes(mdb, all_scrambles)
    
            folksiness.append(mdb.num_rules)
            
            print(f"{r=} of {num_rollouts}: {folksiness[-1]}")

        with open(os.path.join(dump_dir, dump_base + ".pkl"), "wb") as f: pk.dump(folksiness, f)

    if showresults:

        with open(os.path.join(dump_dir, dump_base + ".pkl"), "rb") as f: folksiness = pk.load(f)

        import matplotlib.pyplot as pt
        pt.hist(folksiness)
        pt.show()
