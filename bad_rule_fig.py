import numpy as np

if __name__ == "__main__":

    import os
    import pickle as pk
    from cube import CubeDomain

    γ = 0.99
    ema_threshold = 1.1
    max_depth = 1
    max_actions = 30
    color_neutral = False

    verbose = True

    cube_str = "s120"
    cube_size, valid_actions, tree_depth = CubeDomain.parameters(cube_str)

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
    from scramblers import AllScrambler

    scrambler = AllScrambler(domain, tree)
    mdb = Constructor.init_macro_database(domain, max_rules)
    con = Constructor(alg, max_actions, γ, ema_threshold)

    # unmaxed = con.run_passes(mdb, scrambler, verbose_prefix)
    done = False
    num_rules = mdb.num_rules
    while True:

        new_pass, (s, a) = scrambler.next_instance()

        if new_pass:
            # if done stayed true for entire previous pass, finished successfully
            if done: break
            # otherwise, reset done for the new pass
            done = True

        rq = mdb.query(s[0])

        ϕ, maxed_out = con.incorporate(mdb, s, a)
        done = ϕ and done

        if not ϕ and mdb.num_rules == num_rules:
            if (mdb.tamed[rq] < con.num_incs-2).sum() > 1:
                break
        
        num_rules = mdb.num_rules

    mdb.rewind(con.num_incs - 2)

    print(mdb.num_rules)

    r = mdb.query(s[0])
    M = max_actions - (con.alg.max_depth + len(mdb.macros[r]))
    v, p, r_, s_ = con.alg.run(M, mdb, mdb.apply_rule(r, s[0]))
    assert not v

    plan = ()
    idxb = ()
    for (sym, actions, macro) in p:
        idxb += (len(plan) + len(actions),)
        plan += tuple(actions) + tuple(macro)

    print(p)
    sb = [s[0]] + domain.intermediate_states(plan, s[0])

    r_.insert(0, r)
    s_.insert(0, s[0])
    # ω = self.make_ω(mdb, r_, s_)
    # (n, k) = ω[np.random.choice(len(ω))]
    # mdb.disable(r_[n], k, tamed=self.num_incs)

    vg, pg, r_g, s_g = con.alg.run(M, mdb, mdb.prototypes[r])
    print(vg)
    plan = ()
    idxg = ()
    for (sym, actions, macro) in pg:
        idxg += (len(plan) + len(actions),)
        plan += tuple(actions) + tuple(macro)

    sg = [mdb.prototypes[r]] + domain.intermediate_states(plan, mdb.prototypes[r])
    r_g.insert(0, r)
    s_g.insert(0, s[0])

    print(len(sb))
    print(len(sg))

    import matplotlib.pyplot as pt
    import matplotlib.patches as mp

    from matplotlib import rcParams
    rcParams['font.family'] = 'serif'
    rcParams['font.size'] = 12
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = r'\boldmath'

    pt.figure(figsize=(12, 6)) # for presentation
    ax = pt.gca()

    for row, states in enumerate((sb[:len(sg)+3], sg)):
    
        for s, state in enumerate(states):
        
            domain.render(state, ax, x0=s*9, y0=-row*10, text=False)
            if s + 1 < len(states):
                ax.add_patch(mp.FancyArrow(
                    s*9 + 3, -row*10, 3, 0, color='k', length_includes_head=True,
                    head_width=1, head_length=1,
                    alpha = 1))

    for ii, rr, ss in zip(idxg, r_g, s_g):
        domain.render(ss * (1 - mdb.wildcards[rr]), ax, x0=ii*9, y0=-row*5, text=False)
        # if s + 1 < len(states):
        #     ax.add_patch(mp.FancyArrow(
        #         s*9 + 3, -row*9, 3, 0, color='k', length_includes_head=True,
        #         head_width=1, head_length=1,
        #         alpha = 1))
    # domain.render(s_g[0] * (1 - mdb.wildcards[r_g[0]]), ax, x0=0, y0=-row*5, text=False)

    pt.text(-9, 0, r"$s^{(0)}$")
    pt.text(-9, -10, r"$S_r$")
    pt.text(-9, -5, r"$S_r \vee W_r$")


    ax.axis("equal")
    ax.axis("off")
    pt.show()


