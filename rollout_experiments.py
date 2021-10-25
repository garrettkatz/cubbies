import os
import pickle as pk
import numpy as np
from cube import CubeDomain
from tree import SearchTree
from macro_database import MacroDatabase
from algorithm import Algorithm
from constructor import Constructor

if __name__ == "__main__":

    do_cons = False
    showresults = True

    γ = 0.9
    ema_threshold = 0.9999
    max_depth = 1
    max_actions = 20
    color_neutral = False

    num_repetitions = 1000
    num_rollouts = 1000

    cube_str = "s120"
    # cube_str = "s5040"
    # cube_str = "s29k"
    # cube_str = "pocket"
    cube_size, valid_actions, tree_depth = CubeDomain.parameters(cube_str)

    dump_dir = "roe"
    dump_base = "N%d%s_D%d_M%d_cn%d" % (
        cube_size, cube_str, tree_depth, max_depth, color_neutral)

    if do_cons:

        if not os.path.exists(dump_dir): os.mkdir(dump_dir)

        print("making states...")
        domain = CubeDomain(cube_size, valid_actions)
        tree = SearchTree(domain, tree_depth)
        assert tree.depth() == tree_depth
        print("done. running rollouts...")
        
        bfs_tree = SearchTree(domain, max_depth)
        all_scrambles = Constructor.make_all_scrambles(domain, tree)
        alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)

        for rep in range(num_repetitions):

            mdb = Constructor.init_macro_database(domain, max_rules=tree.size())
            con = Constructor(alg, max_actions, γ, ema_threshold)
            con.run_passes(mdb, all_scrambles)

            inc = np.random.choice(con.num_incs)
            mdb.rewind(inc)
            con.rewind(inc)

            folksiness = []
            ema_histories = []
        
            for ro in range(num_rollouts):
        
                mdb_ro = mdb.copy()
                con_ro = con.copy()
                con_ro.run_passes(mdb_ro, all_scrambles)
        
                folksiness.append(mdb_ro.num_rules)
                ema_histories.append(con_ro.ema_history)
                
                print(
                    f"({rep}, {ro}) of ({num_repetitions}, {num_rollouts}): \
                    {con.ema_history[con.num_incs]} ~> {mdb.num_rules} v {mdb_ro.num_rules}")
    
            with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "wb") as f:
                pk.dump((con, mdb, folksiness, ema_histories), f)

    if showresults:

        import matplotlib.pyplot as pt

        # for rep in range(num_repetitions):
        #     with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "rb") as f:
        #         (con, mdb, folksiness, ema_histories) = pk.load(f)
            
        #     pt.subplot(2,2,1)
        #     x, y, yerr = con.num_incs, np.mean(folksiness), np.std(folksiness)
        #     pt.errorbar(x, y, yerr, c='k')
        #     pt.plot(x, y, 'ko')
        #     pt.xlabel("Branch inc")
        #     pt.ylabel("Folksiness")

        #     pt.subplot(2,2,2)
        #     x, y, yerr = con.ema_history[-1], np.mean(folksiness), np.std(folksiness)
        #     pt.errorbar(x, y, yerr, c='k')
        #     pt.plot(x, y, 'ko')
        #     pt.xlabel("Branch ema")
        #     pt.ylabel("Folksiness")

        #     pt.subplot(2,2,3)
        #     x, y = con.num_incs, np.std(folksiness)
        #     pt.scatter(x, y, c='k')
        #     pt.xlabel("Branch inc")
        #     pt.ylabel("Dev Folksiness")

        #     pt.subplot(2,2,4)
        #     x, y = con.ema_history[-1], np.std(folksiness)
        #     pt.scatter(x, y, c='k')
        #     pt.xlabel("Branch ema")
        #     pt.ylabel("Dev Folksiness")

        # pt.show()

        # for rep in range(num_repetitions):
        #     with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "rb") as f:
        #         (con, mdb, folksiness, ema_histories) = pk.load(f)

        #     estimates = [np.mean(folksiness[:n+1]) for n in range(len(folksiness))]
        #     errors = np.fabs(np.array(estimates) - np.mean(folksiness))

        #     # pt.subplot(2,2,1)
        #     pt.plot(errors, c=(con.ema_history[-1],)*3, marker='.', linestyle='-')
        #     pt.xlabel("Num rollouts")
        #     pt.ylabel("Folksiness error")

        # pt.show()

        for rep in range(num_repetitions):
            with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "rb") as f:
                (con, mdb, folksiness, ema_histories) = pk.load(f)

            best = np.min(folksiness)
            error = best - mdb.num_rules

            # pt.subplot(2,2,1)
            pt.scatter(con.ema_history[-1], error, color='k', marker='.')
            pt.xlabel("Branchpoint ema")
            pt.ylabel("Min folksiness increase")

        pt.show()
