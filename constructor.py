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
            if ϕ: print(f"{self.num_incs}: ema = {ema}")
            ema = self.γ * ema + (1. - self.γ) * int(ϕ)
            self.ema_history.append(ema)

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
    
    all_states = tree.states_rooted_at(solved)
    optimal_paths = tuple(map(tuple, map(domain.reverse, tree.paths()))) # from state to solved
    all_probs = list(zip(all_states, optimal_paths))
    
    def scramble(num_actions):
        idx = np.random.choice(len(all_probs))
        a = optimal_paths[idx]
        s = [all_states[idx]] + domain.intermediate_states(a, all_states[idx])
        return s, a

    ### test persistent prototype chains
    mdb = MacroDatabase(domain, len(all_probs))
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

    # run constructor to convergence
    for reps in range(30):
        mdb = MacroDatabase(domain, len(all_probs))
        mdb.add_rule(solved, (), 0, added=-1)
        for w in range(domain.state_size()): mdb.disable(0, w, tamed=-1)

        alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)
        cons = Constructor(alg, max_actions, scramble, γ, ema_threshold)
        cons.run(mdb)
