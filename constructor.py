import numpy as np
import matplotlib.pyplot as pt

class Constructor:
    def __init__(self, alg, max_actions, scramble, γ, ema_threshold):
        self.alg = alg
        self.max_actions = max_actions
        self.scramble = scramble
        self.γ = γ
        self.ema_threshold = ema_threshold
        self.ema_history = [0.]
        self.num_incs = 0

    def make_ω(self, mdb, r_, s_):
        return [(n,k)
            for n in range(len(r_))
            for k in np.flatnonzero(mdb.prototypes[r_[n]] != s_[n])]

    def make_τ(self, mdb, s):
        queries = [mdb.query(s[t]) for t in range(len(s))]
        return [(t,r)
            for (t,r) in enumerate(queries)
            if r != None and (s[t] == mdb.prototypes[r]).all() and self.alg.max_depth + t + mdb.costs[r] <= self.max_actions]

    def incorporate(self, mdb, s, a):
        ϕ = True

        r = mdb.query(s[0])
        if r != None:
            M = self.max_actions - (self.alg.max_depth + len(mdb.macros[r]))
            v, p, r_, s_ = self.alg.run(M, mdb, mdb.apply_rule(r, s[0]))
            if not v:
                ϕ = False
                r_.insert(0, r)
                s_.insert(0, s[0])
                ω = self.make_ω(mdb, r_, s_)
                (n, k) = ω[np.random.choice(len(ω))]
                mdb.disable(r_[n], k, tamed=self.num_incs)

        result = self.alg.rule_search(mdb, s[0])
        if result == False:
            ϕ = False
            τ = self.make_τ(mdb, s)
            (t, r) = τ[np.random.choice(len(τ))]
            mdb.add_rule(s[0], a[:t], t + mdb.costs[r], added=self.num_incs)

        self.num_incs += 1
        return ϕ

    def run(self, mdb):
        ema = self.ema_history[self.num_incs]
        while ema < self.ema_threshold:
            s, a = self.scramble(self.max_actions - self.alg.max_depth)
            ϕ = self.incorporate(mdb, s, a)
            ema = self.γ * ema + (1. - self.γ) * int(ϕ)
            self.ema_history.append(ema)

    def run_passes(self, mdb, all_scrambles, verbose=True):
        done = False
        while not done:
            done = True
            for i, (s, a) in enumerate(all_scrambles()):
                ϕ = self.incorporate(mdb, s, a)
                done = ϕ and done
            if verbose: print(f"{mdb.num_rules} rules, {self.num_incs} incs, done = {done}")

if __name__ == "__main__":

    import numpy as np

    γ = 0.9
    ema_threshold = 0.9999

    max_depth = 1
    max_actions = 20
    color_neutral = False

    from cube import CubeDomain
    domain = CubeDomain(2)
    solved = domain.solved_state()

    from tree import SearchTree
    bfs_tree = SearchTree(domain, max_depth)

    from macro_database import MacroDatabase
    mdb = MacroDatabase(domain, 3)
    mdb.add_rule(solved, (), 0, added=-1)
    for w in range(domain.state_size()): mdb.disable(0, w, tamed=-1)

    from algorithm import Algorithm
    alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)

    scramble = lambda _: None # placeholder
    cons = Constructor(alg, max_actions, scramble, γ, ema_threshold)
    ϕ = cons.incorporate(mdb, [solved], [])
    assert ϕ
    assert mdb.num_rules == 1

    # within max depth
    s, a = [domain.perform((0,1,1), solved), solved], [(0,1,3)]
    ϕ = cons.incorporate(mdb, s, a)
    assert ϕ
    assert mdb.num_rules == 1

    # outside max depth
    a = [(0,1,1), (1,1,1), (2,1,1)]
    s = [solved] + domain.intermediate_states(a, solved)
    ϕ = cons.incorporate(mdb, s[::-1], domain.reverse(a))
    assert not ϕ
    assert mdb.num_rules == 2

    # outside max depth with intermediate tau endpoint
    for r in range(mdb.num_rules):
        for w in range(domain.state_size()):
            mdb.disable(r, w)
    a = [(0,1,1), (1,1,1), (2,1,1), (1,1,1), (0,1,1), (1,1,1)]
    s = [solved] + domain.intermediate_states(a, solved)
    ϕ = cons.incorporate(mdb, s[::-1], domain.reverse(a))
    assert not ϕ
    assert mdb.num_rules == 3

    ### Integration tests
    cube_size = 2
    valid_actions = (
        (0,1,1), (0,1,2), (0,1,3),
        (1,1,2), 
    )
    tree_depth = 11

    domain = CubeDomain(cube_size, valid_actions)
    tree = SearchTree(domain, tree_depth)
    assert tree.depth() == tree_depth
    
    def scramble(num_actions):
        idx = np.random.choice(tree.size())
        s0 = domain.solved_state()[tree.permutations()[idx]]
        a = domain.reverse(tree.paths()[idx])
        s = [s0] + domain.intermediate_states(a, s0)
        return s, a

    def all_scrambles():
        for idx in range(tree.size()):
            s0 = domain.solved_state()[tree.permutations()[idx]]
            a = domain.reverse(tree.paths()[idx])
            s = [s0] + domain.intermediate_states(a, s0)
            yield s, a

    ### test persistent prototype chains
    mdb = MacroDatabase(domain, tree.size())
    mdb.add_rule(solved, (), 0, added=-1)
    for w in range(domain.state_size()): mdb.disable(0, w, tamed=-1)

    alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)
    cons = Constructor(alg, max_actions, scramble, γ, ema_threshold)

    for i in range(500):
        s, a = scramble(cons.max_actions - cons.alg.max_depth)
        ϕ = cons.incorporate(mdb, s, a)
        for r in range(1, mdb.num_rules):
            print(i,r)
            s_ = mdb.apply_rule(r, mdb.prototypes[r])
            assert (s_ == mdb.prototypes[:r]).all(axis=1).any()

    # run constructor to ema convergence without crashing
    for reps in range(30):
        mdb = MacroDatabase(domain, tree.size())
        mdb.add_rule(solved, (), 0, added=-1)
        for w in range(domain.state_size()): mdb.disable(0, w, tamed=-1)

        alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)
        cons = Constructor(alg, max_actions, scramble, γ, ema_threshold)
        cons.run(mdb)

    # run constructor to true convergence with all scrambles
    from time import perf_counter
    rep_times = []
    for reps in range(30):
        print(f"check {reps}")

        start = perf_counter()
        mdb = MacroDatabase(domain, tree.size())
        mdb.add_rule(solved, (), 0, added=-1)
        for w in range(domain.state_size()): mdb.disable(0, w, tamed=-1)
    
        alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)
        cons = Constructor(alg, max_actions, scramble, γ, ema_threshold)
        cons.run_passes(mdb, all_scrambles)

        rep_times.append(perf_counter() - start)
    
        for state in tree.states_rooted_at(domain.solved_state()):
            success, plan, rule_indices, triggerers = alg.run(max_actions, mdb, state)
            assert success

    print(f"avg time = {np.mean(rep_times)}")
    
    ### Scale up domain
    print("making scaled up states...")

    # pocket cube: one axis with quarter twists, two with half twists
    # 5040 states, reached in max_depth=13
    cube_size = 2
    valid_actions = (
        (0,1,1), (0,1,2), (0,1,3),
        (1,1,2),
        (2,1,2),
    )
    cube_str = "s5040"
    tree_depth = 13

    # # pocket cube: two axes with quarter twists, one fixed
    # # 29k states, reached in max_depth=14
    # cube_size = 2
    # valid_actions = (
    #     (0,1,1), (0,1,2), (0,1,3),
    #     (1,1,1), (1,1,2), (1,1,3),
    # )
    # cube_str = "s29k"
    # tree_depth = 14

    # # full pocket cube, all actions allowed, ~4m states
    # cube_size = 2
    # valid_actions = None
    # cube_str = "full"
    # tree_depth = 11

    domain = CubeDomain(cube_size, valid_actions)
    tree = SearchTree(domain, tree_depth)
    assert tree.depth() == tree_depth

    def all_scrambles():
        for idx in range(tree.size()):
            s0 = domain.solved_state()[tree.permutations()[idx]]
            a = domain.reverse(tree.paths()[idx])
            s = [s0] + domain.intermediate_states(a, s0)
            yield s, a

    print("done. running constructor passes...")
    
    start = perf_counter()
    mdb = MacroDatabase(domain, tree.size())
    mdb.add_rule(solved, (), 0, added=-1)
    for w in range(domain.state_size()): mdb.disable(0, w, tamed=-1)

    alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)
    cons = Constructor(alg, max_actions, scramble, γ, ema_threshold)
    cons.run_passes(mdb, all_scrambles)

    total_time = perf_counter() - start
    print(f"total time = {total_time}")

    print("checking...")
    for i in range(tree.size()):
        state = domain.solved_state()[tree.permutations()[i]]
        success, plan, rule_indices, triggerers = alg.run(max_actions, mdb, state)
        assert success
    print("success!")
    
