import numpy as np
import matplotlib.pyplot as pt
from scramblers import AllScrambler
from macro_database import MacroDatabase

class Constructor:
    def __init__(self, alg, max_actions, γ, ema_threshold):
        self.alg = alg
        self.max_actions = max_actions
        self.γ = γ
        self.ema_threshold = ema_threshold
        self.ema_history = [0.]
        self.augment_incs = [0]
        self.rule_counts = [1]
        self.num_incs = 1

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
        ϕ, maxed_out = True, False

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

        else:
            result = self.alg.rule_search(mdb, s[0])
    
            if result == False and mdb.num_rules < mdb.max_rules:
                ϕ = False
                τ = self.make_τ(mdb, s)
                (t, r) = τ[np.random.choice(len(τ))] # ~3-5 10M incs * 10**-5 | .003-.005 | .023
                # (t, r) = τ[0] # ~3.5-4.5 10M incs * 10**-5 | .003-.0045 | .023-.029
                mdb.add_rule(s[0], a[:t], t + mdb.costs[r], added=self.num_incs)
    
            elif result == False and mdb.num_rules == mdb.max_rules:
                maxed_out = True

        if not ϕ: self.augment_incs.append(self.num_incs)
        self.rule_counts.append(mdb.num_rules)
        self.num_incs += 1
        return ϕ, maxed_out

    def run(self, mdb, scramble):
        ema = self.ema_history[self.num_incs-1]
        while ema < self.ema_threshold:
            s, a = scramble(self.max_actions - self.alg.max_depth)
            ϕ, maxed_out = self.incorporate(mdb, s, a)
            ema = self.γ * ema + (1. - self.γ) * int(ϕ)
            self.ema_history.append(ema)
            if maxed_out: return False
        return True

    def run_to_augment(self, mdb, scrambler):
        ema = self.ema_history[self.num_incs-1]
        while ema < self.ema_threshold:
            _, (s, a) = scrambler.next_instance()
            ϕ, maxed_out = self.incorporate(mdb, s, a)
            ema = self.γ * ema + (1. - self.γ) * int(ϕ)
            self.ema_history.append(ema)
            if maxed_out: return False
            if not ϕ: break # stop if augmented
        return True

    def run_passes(self, mdb, scrambler, verbose=None):
        # returns True if the run finished successfully (unmaxed)
        # returns False if the run failed (maxed out)
        ema = self.ema_history[self.num_incs-1]
        done = False
        while True:

            new_pass, (s, a) = scrambler.next_instance()

            if new_pass:
                # if done stayed true for entire previous pass, finished successfully
                if done: return True
                # otherwise, reset done for the new pass
                done = True

            ϕ, maxed_out = self.incorporate(mdb, s, a)
            done = ϕ and done
            ema = self.γ * ema + (1. - self.γ) * int(ϕ)
            self.ema_history.append(ema)

            if maxed_out: return False
            if verbose is not None and not ϕ:
                print(verbose + f"{mdb.num_rules} rules, {self.num_incs} incs, done = {done}")

        return True

    def copy(self):
        con = Constructor(self.alg, self.max_actions, self.γ, self.ema_threshold)
        con.ema_history = list(self.ema_history)
        con.num_incs = self.num_incs
        con.augment_incs = list(self.augment_incs)
        return con

    def rewind(self, inc):
        # up to and including inc
        self.num_incs = inc+1
        self.ema_history = self.ema_history[:self.num_incs+1]
        self.augment_incs = list(filter(lambda i: i <= inc, self.augment_incs))
        return self

    # static methods
    def init_macro_database(domain, max_rules):
        mdb = MacroDatabase(domain, max_rules)
        mdb.add_rule(domain.solved_state(), (), 0, added=-1)
        for w in range(domain.state_size()): mdb.disable(0, w, tamed=-1)
        return mdb

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

    mdb = Constructor.init_macro_database(domain, max_rules=3)

    from algorithm import Algorithm
    alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)

    cons = Constructor(alg, max_actions, γ, ema_threshold)
    ϕ, maxed_out = cons.incorporate(mdb, [solved], [])
    assert ϕ
    assert mdb.num_rules == 1

    # within max depth
    s, a = [domain.perform((0,1,1), solved), solved], [(0,1,3)]
    ϕ, maxed_out = cons.incorporate(mdb, s, a)
    assert ϕ
    assert mdb.num_rules == 1

    # outside max depth
    a = [(0,1,1), (1,1,1), (2,1,1)]
    s = [solved] + domain.intermediate_states(a, solved)
    ϕ, maxed_out = cons.incorporate(mdb, s[::-1], domain.reverse(a))
    assert not ϕ
    assert mdb.num_rules == 2

    # outside max depth with intermediate tau endpoint
    for r in range(mdb.num_rules):
        for w in range(domain.state_size()):
            mdb.disable(r, w)
    a = [(0,1,1), (1,1,1), (2,1,1), (1,1,1), (0,1,1), (1,1,1)]
    s = [solved] + domain.intermediate_states(a, solved)
    ϕ, maxed_out = cons.incorporate(mdb, s[::-1], domain.reverse(a))
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

    scrambler = AllScrambler(domain, tree)

    ### test persistent prototype chains
    mdb = Constructor.init_macro_database(domain, max_rules=tree.size())
    alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)
    cons = Constructor(alg, max_actions, γ, ema_threshold)

    for i in range(500):
        s, a = scramble(cons.max_actions - cons.alg.max_depth)
        ϕ, maxed_out = cons.incorporate(mdb, s, a)
        for r in range(1, mdb.num_rules):
            print(i,r)
            s_ = mdb.apply_rule(r, mdb.prototypes[r])
            assert (s_ == mdb.prototypes[:r]).all(axis=1).any()

    ### test maxed_out
    mdb = Constructor.init_macro_database(domain, max_rules=2)
    alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)
    cons = Constructor(alg, max_actions, γ, ema_threshold)
    success = cons.run(mdb, scramble)
    assert not success
    assert mdb.num_rules == 2

    # run constructor to ema convergence without crashing
    for reps in range(30):
        mdb = Constructor.init_macro_database(domain, max_rules=tree.size())
        alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)
        cons = Constructor(alg, max_actions, γ, ema_threshold)
        cons.run(mdb, scramble)

    # run constructor to true convergence with all scrambler
    from time import perf_counter
    rep_times = []
    for reps in range(30):
        print(f"check {reps}")

        start = perf_counter()

        mdb = Constructor.init_macro_database(domain, max_rules=tree.size())
        alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)
        cons = Constructor(alg, max_actions, γ, ema_threshold)
        cons.run_passes(mdb, scrambler, verbose="")

        rep_times.append(perf_counter() - start)
    
        for state in tree.states_rooted_at(domain.solved_state()):
            success, plan, rule_indices, triggerers = alg.run(max_actions, mdb, state)
            assert success

    print(f"avg time = {np.mean(rep_times)}")
    
    ### Scale up domain
    print("making scaled up states...")

    cube_str = "s120"
    # cube_str = "s5040"
    # cube_str = "s29k"
    # cube_str = "pocket"

    cube_size, valid_actions, tree_depth = CubeDomain.parameters(cube_str)

    domain = CubeDomain(cube_size, valid_actions)
    tree = SearchTree(domain, tree_depth)
    assert tree.depth() == tree_depth

    scrambler = AllScrambler(domain, tree)

    print("done. running constructor passes...")
    
    start = perf_counter()

    mdb = Constructor.init_macro_database(domain, max_rules=tree.size())
    alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)
    cons = Constructor(alg, max_actions, γ, ema_threshold)
    cons.run_passes(mdb, scrambler, verbose="")

    total_time = perf_counter() - start
    print(f"total time = {total_time}")

    print("checking...")
    for i in range(tree.size()):
        state = domain.solved_state()[tree.permutations()[i]]
        success, plan, rule_indices, triggerers = alg.run(max_actions, mdb, state)
        assert success
    print("success!")
    
