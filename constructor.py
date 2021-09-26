import numpy as np

class Constructor:
    def __init__(self, alg, max_actions):
        self.alg = alg
        self.max_actions = max_actions
        self.num_incs = 0

    def make_ω(self, mdb, r_, s_):
        return [(n,k)
            for n in range(len(r_))
            for k in np.flatnonzero(mdb.prototypes[r_[n]] != s_[n])]

    def make_τ(self, mdb, s):
        return [(t,r)
            for t in range(len(s))
            for r in np.flatnonzero((s[t] == mdb.prototypes).all(axis=1))
            if self.alg.max_depth + t + mdb.costs[r] <= self.max_actions]

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

if __name__ == "__main__":

    import numpy as np

    max_depth = 1
    max_actions = 20
    color_neutral = False

    from cube import CubeDomain
    domain = CubeDomain(2)
    solved = domain.solved_state()

    from tree import SearchTree
    bfs_tree = SearchTree(domain, max_depth)

    from macro_database import MacroDatabase
    mdb = MacroDatabase(domain, 2)
    mdb.add_rule(solved, (), 0, added=-1)
    for w in range(domain.state_size()): mdb.disable(0, w, tamed=-1)

    from algorithm import Algorithm
    alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)

    cons = Constructor(alg, max_actions)
    ϕ = cons.incorporate(mdb, [solved], [])
    assert ϕ

    # within max depth
    s, a = [domain.perform((0,1,1), solved), solved], [(0,1,3)]
    ϕ = cons.incorporate(mdb, s, a)
    assert ϕ

    # outside max depth
    a = [(0,1,1), (1,1,1), (2,1,1)]
    s = [solved] + domain.intermediate_states(a, solved)
    ϕ = cons.incorporate(mdb, s[::-1], domain.reverse(a))
    assert not ϕ

