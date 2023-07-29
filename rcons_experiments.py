import numpy as np

if __name__ == "__main__":

    import os
    import pickle as pk
    from cube import CubeDomain

    do_cons = True
    showresults = True
    confirm = True

    γ = 0.99
    ema_threshold = 1.1
    max_depth = 1
    max_actions = 30
    color_neutral = False

    verbose = True

    cube_str = "s120"
    num_confirm_incs = 1000
    num_repetitions = 10

    cube_str = "s5040"
    num_confirm_incs = 50
    num_repetitions = 5

    cube_str = "s29k"
    num_confirm_incs = 50
    num_repetitions = 2

    cube_str = "pocket" # can be "s120", "s5040", "s29k", or "pocket"
    num_confirm_incs = 10
    num_repetitions = 1

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
        from constructor import Constructor
        from scramblers import AllScrambler, FolkScrambler, NonScrambler

        if not os.path.exists(dump_dir): os.mkdir(dump_dir)

        for rep in range(num_repetitions):

            if do_cons:

                start = perf_counter()
                verbose_prefix = f"rep {rep} of {num_repetitions} rcons: " if verbose else None
            
                # scrambler = AllScrambler(domain, tree)
                scrambler = NonScrambler(domain, tree)
                mdb = Constructor.init_macro_database(domain, max_rules)
                con = Constructor(alg, max_actions, γ, ema_threshold)
                unmaxed = con.run_passes(mdb, scrambler, verbose_prefix)
    
                total_time = perf_counter() - start
                print(f"rep {rep} of {num_repetitions}, {con.num_incs} incs ({total_time:.2f}s): success = {unmaxed}, {mdb.num_rules} rules")
    
                mdb = mdb.shrink_wrap()
                with open(os.path.join(dump_dir, dump_base + f"_{rep}_mdb.pkl"), "wb") as f: pk.dump(mdb, f)
                with open(os.path.join(dump_dir, dump_base + f"_{rep}_con.pkl"), "wb") as f:
                    pk.dump((con, unmaxed, total_time, mdb.num_rules), f)

            if confirm:
    
                with open(os.path.join(dump_dir, dump_base + f"_{rep}_mdb.pkl"), "rb") as f: mdb = pk.load(f)
                with open(os.path.join(dump_dir, dump_base + f"_{rep}_con.pkl"), "rb") as f:
                    (con, unmaxed, total_time, num_rules) = pk.load(f)
        
                if len(con.augment_incs) > num_confirm_incs:
                    idx = np.linspace(0, len(con.augment_incs)-1, num_confirm_incs).astype(int)
                    confirm_incs = tuple(con.augment_incs[i] for i in idx)
                else:
                    confirm_incs = tuple(con.augment_incs)
        
                confirm_times = []
                success_counts = []
                rule_counts = []
                corrects = []
                emas = []
                
                for i,inc in enumerate(reversed(confirm_incs)):
        
                    mdb = mdb.rewind(inc)
        
                    start = perf_counter()
                    success_count = 0
                    correct = True
                    for t in range(tree.size()):
                        # if t % 1000 == 0: print(f" conf inc {inc} ({i} of {len(confirm_incs)}): state {t} of {tree.size()}")
                        state = domain.solved_state()[tree.permutations()[t]]
                        success, _, _, _ = alg.run(max_actions, mdb, state)
                        correct = correct and success
                        success_count = success_count + success
            
                    success_rate = success_count / tree.size()
                    confirm_time = perf_counter() - start
                    print(f"rep {rep} of {num_repetitions}, conf inc {inc} ({i} of {len(confirm_incs)}): success rate = {success_count} / {tree.size()} = {success_rate:.3f} (correct={correct}), {confirm_time:.2f}s, {mdb.num_rules} rules")
                    
                    confirm_times.insert(0, confirm_time)
                    success_counts.insert(0, success_count)
                    corrects.insert(0, correct)
                    rule_counts.insert(0, mdb.num_rules)
                    emas.insert(0, con.ema_history[inc])
        
                with open(os.path.join(dump_dir, dump_base + f"_{rep}_cnf.pkl"), "wb") as f:
                    pk.dump((confirm_incs, confirm_times, success_counts, corrects, rule_counts, emas, tree.size()), f)

    if showresults:

        import glob
        import matplotlib.pyplot as pt
        from matplotlib import rcParams
        rcParams['font.family'] = 'serif'
        rcParams['font.size'] = 11
        rcParams['text.usetex'] = True

        num_reps = min(num_repetitions, len(glob.glob(os.path.join(dump_dir, dump_base + "_*_con.pkl"))))

        # run times and rule table sizes
        all_run_times, all_rule_counts, all_inc_counts = [], [], []
        for rep in range(num_reps):
            print(f"loading {rep}")
            with open(os.path.join(dump_dir, dump_base + f"_{rep}_con.pkl"), "rb") as f:
                (con, unmaxed, total_time, num_rules) = pk.load(f)
            with open(os.path.join(dump_dir, dump_base + f"_{rep}_cnf.pkl"), "rb") as f:
                (confirm_incs, confirm_times, success_counts, corrects, rule_counts, emas, tree_size) = pk.load(f)
            
            print(f"done. {total_time/60:.2f}min runtime, {num_rules} rules")
            all_run_times.append(total_time)
            all_rule_counts.append(num_rules)
            all_inc_counts.append(con.num_incs)

        print(f"Avg run time: {np.mean(all_run_times)/60} +/- {np.std(all_run_times)/60} min")
        print(f"Avg num rule: {np.mean(all_rule_counts)} +/- {np.std(all_rule_counts)} of {tree_size} ({np.mean(all_rule_counts)/tree_size*100:.2f}\%)")
        print(f"Avg num incs: {np.mean(all_inc_counts)} +/- {np.std(all_inc_counts)} incs")

        # show one representative cons run
        buckets = 1000000
        augment_indicator = np.zeros(con.num_incs + buckets - (con.num_incs % buckets))
        augment_indicator[con.augment_incs] = 1
        augment_density = augment_indicator.reshape((buckets, -1)).mean(axis=1)
        success_rates = tuple(count / tree_size for count in success_counts)
        pt.figure(figsize=(3.5, 1.5))
        pt.plot(np.linspace(0, con.num_incs, buckets), augment_density, '-', color=(.5,)*3, label="Modification")
        pt.plot(confirm_incs, success_rates, 'ko-', label="Correctness")
        pt.xlabel("Incorporation")
        pt.ylabel("Rate")
        pt.legend(loc='center right')
        pt.tight_layout()
        pt.savefig(f"rcons_{dump_base}_{rep}.pdf", bbox_inches='tight')
        pt.show()

        # convergence histograms
        pt.figure(figsize=(3.5, 1.5))
        pt.subplot(1,3,1)
        pt.hist(np.array(all_inc_counts)/10**7, color=(1,)*3, ec='k', rwidth=.75, align="left")
        pt.xlabel("Incs (10M)")
        pt.ylabel("Count")
        pt.ylim([0, 20])
        pt.subplot(1,3,2)
        pt.hist(np.array(all_run_times)/60, color=(1,)*3, ec='k', rwidth=.75, align="left")
        pt.xlabel("Time (min)")
        pt.ylim([0, 20])
        pt.yticks([])
        pt.subplot(1,3,3)
        pt.hist((np.array(all_rule_counts)/10**5).round(1), color=(1,)*3, ec='k', rwidth=.75, align="left")
        pt.xlabel("Rules (100K)")
        pt.ylim([0, 20])
        pt.yticks([])
        pt.tight_layout()
        pt.savefig("rcons_%s_hist.pdf" % cube_str, bbox_inches='tight')
        pt.show()
        

        # data = []
        # varphi_stops, varphi_threshes = [], []
        # all_success_rates, all_emas = [], []
        # # for rep in range(num_repetitions):        
        # for rep in range(len(glob.glob(os.path.join(dump_dir, dump_base + "_*_con.pkl")))):
        # # for rep in range(2):
        #     print(f"loading {rep}")
        #     with open(os.path.join(dump_dir, dump_base + f"_{rep}_con.pkl"), "rb") as f:
        #         (con, unmaxed, total_time, num_rules) = pk.load(f)
        #     with open(os.path.join(dump_dir, dump_base + f"_{rep}_cnf.pkl"), "rb") as f:
        #         (confirm_incs, confirm_times, success_counts, corrects, rule_counts, emas, tree_size) = pk.load(f)

        #     print(f"processing {rep}")
        #     data.append(confirm_incs[-1])
            
        #     thresh = .99
        #     near_inc = np.argmax([ema > thresh for ema in con.ema_history])
        #     near_aug = np.argmax([inc > near_inc for inc in confirm_incs]) - 1
        #     if con.ema_history[near_inc] > thresh:
        #         varphi_stops.append(confirm_incs[near_aug])
        #         varphi_threshes.append(con.ema_history[near_inc])
        #     else:
        #         print(rep, max(con.ema_history), con.ema_history[near_inc])
            
        #     all_success_rates += [count / tree_size for count in success_counts]
        #     all_emas += emas

        # fig = pt.figure(figsize=(3.5, 3))
        # gs = fig.add_gridspec(2,2)
        # ax = fig.add_subplot(gs[0,:])
        # # pt.hist(data, bins = np.arange(0,max(data),500), color=(1,)*3, ec='k', rwidth=.75, align="left")
        # pt.hist(data, color=(1,)*3, ec='k', rwidth=.75, align="left")
        # pt.xlabel("Convergence Incs")
        # pt.ylabel("Frequency")

        # ax = fig.add_subplot(gs[1,0])
        # pt.plot(varphi_stops, varphi_threshes, 'k.')
        # pt.xlabel("0.99 Convergence Time")
        # pt.ylabel("EMA")
        # # pt.xticks([0, 5000])

        # ax = fig.add_subplot(gs[1,1])
        # # idx = np.random.permutation(len(all_emas))[:1000]
        # # pt.plot(np.array(all_success_rates)[idx], np.array(all_emas)[idx], 'k.')
        # pt.plot(all_success_rates, all_emas, 'k.')
        # pt.xlabel("Correctness")
        # pt.ylabel("EMA")

        # pt.tight_layout()
        # pt.savefig("rcons_%s_hist.pdf" % cube_str)
        # pt.show()


        # # show augment incs
        # pt.figure(figsize=(3.5, 1.5))
        # for rep in range(num_reps):
        #     with open(os.path.join(dump_dir, dump_base + f"_{rep}_con.pkl"), "rb") as f:
        #         (con, unmaxed, total_time, num_rules) = pk.load(f)
        #     with open(os.path.join(dump_dir, dump_base + f"_{rep}_cnf.pkl"), "rb") as f:
        #         (confirm_incs, confirm_times, success_counts, corrects, rule_counts, emas, tree_size) = pk.load(f)

        #     augment_incs = np.array(con.augment_incs)
        #     diffs = augment_incs[1:] - augment_incs[:-1]
        #     # pt.plot(diffs)
        #     pt.plot(np.log(diffs))
    
        #     # pt.plot(con.augment_incs)
        #     # pt.plot(np.log(con.augment_incs), 'k-')

        # pt.show()

        # # show one cons run with simple averaging
        # for rep in range(num_repetitions):
        #     if rep == 1: break
        #     with open(os.path.join(dump_dir, dump_base + f"_{rep}_con.pkl"), "rb") as f:
        #         (con, unmaxed, total_time, num_rules) = pk.load(f)
        #     with open(os.path.join(dump_dir, dump_base + f"_{rep}_cnf.pkl"), "rb") as f:
        #         (confirm_incs, confirm_times, success_counts, corrects, rule_counts, emas, tree_size) = pk.load(f)
    
        #     success_rates = tuple(count / tree_size for count in success_counts)
        #     confirm_incs += (con.num_incs-1,)
        #     success_rates += (success_rates[-1],)
    
        #     pt.figure(figsize=(3.5, 1.5))
        #     # plot_incs = confirm_incs if cube_str == "pocket" else con.augment_incs
        #     plot_incs = confirm_incs
        #     for inc in plot_incs[:-1]:
        #         pt.plot([inc, inc], [0, 1], '-', color=(.85,)*3)
        #     pt.plot(plot_incs, success_rates, '--', color=(0,)*3, label="Success Rate")

        #     # ma = np.zeros(con.num_incs)
        #     # augment_incs = tuple(con.augment_incs) + (con.num_incs-1,)
        #     # for i in range(2, len(augment_incs)-1):
        #     #     for inc in range(augment_incs[i], augment_incs[i+1]):
        #     #         ma[inc] = 1 - 1/(inc - augment_incs[i-1])
        #     # # pt.plot(plot_incs, ma[list(plot_incs)], 'k-', label="Moving Average")
        #     # pt.plot(augment_incs, ma[list(augment_incs)], 'k-', label="Moving Average")
            
        #     avg = np.zeros(con.num_incs)
        #     augment_incs = tuple(con.augment_incs) + (con.num_incs-1,)
        #     for j in range(1,len(augment_incs)-1):
        #         for inc in range(augment_incs[j], augment_incs[j+1]):
        #             avg[inc] = 1 - j/inc
        #     pt.plot(avg, 'k-', label="Average")
        #     # pt.plot(augment_incs, ma[list(augment_incs)], 'k-', label="Moving Average")

        #     pt.legend(fontsize=10, loc='lower right')
        #     # pt.plot([con.num_incs - tree_size]*2, [0, 1], 'k--')
        #     pt.ylabel("Correct")
        #     pt.xlabel("Number of incorporations")
        #     pt.yticks([0, 0.5, 1.0])
        #     pt.tight_layout()
        #     # if rep == 0: pt.savefig(f"rcons_{dump_base}_{rep}.pdf")
        #     pt.show()
        #     pt.close()

        # # show one cons run with prob model
        # for rep in range(num_repetitions):
        #     if rep == 1: break
        #     with open(os.path.join(dump_dir, dump_base + f"_{rep}_con.pkl"), "rb") as f:
        #         (con, unmaxed, total_time, num_rules) = pk.load(f)
        #     with open(os.path.join(dump_dir, dump_base + f"_{rep}_cnf.pkl"), "rb") as f:
        #         (confirm_incs, confirm_times, success_counts, corrects, rule_counts, emas, tree_size) = pk.load(f)
    
        #     static_steps = tuple(con.augment_incs[i+1] - con.augment_incs[i] - 1 for i in range(len(con.augment_incs)-1))
        #     static_steps += (con.num_incs - con.augment_incs[-1],)
    
        #     success_rates = tuple(count / tree_size for count in success_counts)
        #     confirm_incs += (con.num_incs,)
        #     success_rates += (success_rates[-1],)
    
        #     thresh_probs = []
        #     thresh = .99
        #     # thresh = 1
        #     N = tree_size
        #     for inc in range(con.num_incs):
        #         last_augment = max(i for i in con.augment_incs if i <= inc)
        #         t = inc - last_augment
        #         R = con.rule_counts[inc]
    
        #         numer = N/(t+1) * (1 - thresh**(t+1)) + 1
        #         denom = N/(t+1) * (1 - (R / N)**(t+1)) + 1
        #         thresh_probs.append(numer / denom)
    
        #     pt.figure(figsize=(3.5, 1.5))
        #     pt.plot(confirm_incs, success_rates, '-o', color=(.75,)*3, label="Success Rate")
        #     # pt.step(
        #     #     confirm_incs, success_rates, '-o', where="post", color=(.75,)*3, label="Success Rate")
        #     # pt.plot(con.ema_history, '-', color=(0,)*3, label="EMA")
        #     # pt.plot(confirm_incs[:-1], thresh_probs, '--', color=(0,)*3, label=f"Pr(k/N $\geq$ {thresh:.2f})")
        #     pt.plot(thresh_probs, '--', color=(0,)*3, label=f"Pr(k/N $\geq$ {thresh:.2f})")
        #     pt.legend(fontsize=10, loc='lower right')
        #     pt.plot([con.num_incs - tree_size]*2, [0, 1], 'k--')
        #     pt.ylabel("Correct")
        #     pt.xlabel("Number of incorporations")
        #     pt.yticks([0, 0.5, 1.0])
        #     pt.tight_layout()
        #     if rep == 0: pt.savefig(f"rcons_{dump_base}_{rep}.pdf")
        #     pt.show()
        #     pt.close()

            
