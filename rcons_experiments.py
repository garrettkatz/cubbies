import numpy as np

if __name__ == "__main__":

    import os
    import pickle as pk
    from cube import CubeDomain

    do_cons = False
    showresults = False
    confirm = True

    γ = 0.99
    ema_threshold = 1.1
    max_depth = 1
    max_actions = 30
    color_neutral = False

    exhaust = False

    num_repetitions = 1

    # cube_str = "s120"
    # cube_str = "s5040"
    # cube_str = "s29k"
    cube_str = "pocket"
    cube_size, valid_actions, tree_depth = CubeDomain.parameters(cube_str)

    dump_dir = "rce"
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

        from time import perf_counter

    if do_cons:

        if not os.path.exists(dump_dir): os.mkdir(dump_dir)

        from constructor import Constructor
        from scramblers import AllScrambler, FolkScrambler

        for rep in range(num_repetitions):

            start = perf_counter()
        
            scrambler = AllScrambler(domain, tree)
            # scrambler = FolkScrambler(domain, tree, max_actions - max_depth)
            mdb = Constructor.init_macro_database(domain, max_rules)
            con = Constructor(alg, max_actions, γ, ema_threshold)

            success_rates = [1 / tree.size()]
            checks = [1]
            correct = False
            while not correct:

                # run more incorporations
                print("  running to augment...")
                unmaxed = con.run_to_augment(mdb, scrambler)

                print("  checking correct...")
                # check if correct on all states, short-circuit if not exhaustive
                success_count = 0
                correct = True
                first_fail = tree.size()
                for t in range(tree.size()):
                    state = domain.solved_state()[tree.permutations()[t]]
                    success, _, _, _ = alg.run(max_actions, mdb, state)
                    correct = correct and success
                    success_count = success_count + success
                    if not correct:
                        first_fail = min(first_fail, t)
                        if not exhaust: break

                success_rate = success_count / tree.size() if exhaust else 1 - 1/first_fail

                checks.append(first_fail)
                success_rates.append(success_rate)
                print(f" {con.num_incs} incs: First {first_fail} of {tree.size()} correct, {success_rate:.2f}")

                # stop early if rules maxed out
                if not unmaxed: break

            total_time = perf_counter() - start
            print(f"{total_time:.2f}s, {mdb.num_rules} rules")

            with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "wb") as f:
                pk.dump((mdb, con, checks, success_rates, total_time), f)

    if confirm:

        rep = 0
        with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "rb") as f:
            (mdb, con, checks, success_rates, total_time) = pk.load(f)

        start = perf_counter()
        correct = True
        success_count = 0
        for t in range(tree.size()):
            if t % 1000 == 0: print(f"{t} of {tree.size()}")
            state = domain.solved_state()[tree.permutations()[t]]
            success, _, _, _ = alg.run(max_actions, mdb, state)
            correct = correct and success
            success_count = success_count + success

        success_rate = success_count / tree.size()
        total_time = perf_counter() - start
        print(f"success rate: {success_count} / {tree.size()} = {success_rate} (correct={correct}), {total_time:.2f}s, {mdb.num_rules} rules")

    if showresults:

        import matplotlib.pyplot as pt
        from matplotlib import rcParams
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 11
        rcParams['text.usetex'] = True

        # show one cons run
        for rep in range(num_repetitions):
            with open(os.path.join(dump_dir, dump_base + f"_{rep}.pkl"), "rb") as f:
                (mdb, con, checks, success_rates, total_time) = pk.load(f)

            pt.figure(figsize=(3.5, 1.5))
            # pt.plot(con.augment_incs, success_rates, '-', color=(.75,)*3, label="Success Rate")
            pt.step(con.augment_incs, success_rates, '-', where="post", color=(.75,)*3, label="Success Rate")
            pt.plot(con.ema_history, '-', color=(0,)*3, label="EMA")
            pt.legend(fontsize=10, loc='lower right')
            pt.ylabel("Correct")
            pt.xlabel("Number of incorporations")
            pt.yticks([0, 0.5, 1.0])
            pt.tight_layout()
            if rep == 0: pt.savefig(f"rcons_{dump_base}_{rep}.pdf")
            pt.show()
            pt.close()

