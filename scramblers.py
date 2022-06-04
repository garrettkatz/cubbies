import numpy as np

class NonScrambler:

    def __init__(self, domain, tree, max_actions=None):
        self.domain = domain
        self.tree = tree
        self.inc = 0

    def next_instance(self):

        # get next instance index (wrap around with mod)
        i = self.inc % self.tree.size()

        # new pass when instance cycles back to first state
        new_pass = (i == 0)
                
        # update inc counter
        self.inc += 1

        # get current state and path
        s0 = self.domain.solved_state()[self.tree.permutations()[i]]
        a = self.domain.reverse(self.tree.paths()[i])
        s = [s0] + self.domain.intermediate_states(a, s0)

        # return scramble and pass status
        return new_pass, (s, a)

    def rewind(self, inc):
        # up to and including inc

        # reset inc
        self.inc = inc+1
        
        return self

class FolkScrambler:

    def __init__(self, domain, tree, max_actions):
        self.domain = domain
        self.tree = tree
        self.max_actions = max_actions

        # fixed scramble length might render some states unreachable
        # use random length where more scramble actions are more likely
        self.probs = np.arange(max_actions+1, dtype=float)
        self.probs /= self.probs.sum()
    
    def next_instance(self):
        valid_actions = self.domain.valid_actions()
        num_actions = np.random.choice(self.max_actions + 1, p=self.probs)
        actions = [
            valid_actions[np.random.choice(len(valid_actions))]
            for _ in range(num_actions)]

        new_pass = False
        s0 = self.domain.execute(actions, self.domain.solved_state())
        a = self.domain.reverse(actions)
        s = [s0] + self.domain.intermediate_states(a, s0)

        return new_pass, (s, a)

    def rewind(self, inc):
        return self

class AllScrambler:

    def __init__(self, domain, tree, max_actions=None):
        self.domain = domain
        self.tree = tree
        self.idx = []
        self.inc = 0

    def next_instance(self):

        # permute more indices when needed
        new_pass = False
        if self.inc == len(self.idx) * self.tree.size():
            new_pass = True
            self.idx.append(np.random.permutation(self.tree.size()))
                
        # get next scrambled index
        i = self.idx[int(self.inc / self.tree.size())][self.inc % self.tree.size()]
        self.inc += 1

        # get scrambled state and path
        s0 = self.domain.solved_state()[self.tree.permutations()[i]]
        a = self.domain.reverse(self.tree.paths()[i])
        s = [s0] + self.domain.intermediate_states(a, s0)

        # return scramble and pass status
        return new_pass, (s, a)

    def rewind(self, inc):
        # up to and including inc

        # get last pass and iteration in that pass at inc
        p = int(inc / self.tree.size())
        n = inc % self.tree.size()

        # discard subsequent passes
        self.idx = self.idx[:p+1]

        # rescramble remainder of pass
        self.idx[p][n+1:] = np.random.permutation(self.idx[p][n+1:])

        # reset inc
        self.inc = inc+1
        
        return self

if __name__ == "__main__":

    from cube import CubeDomain

    num = 120
    cube_str = f"s{num}"
    cube_size, valid_actions, tree_depth = CubeDomain.parameters(cube_str)
    domain = CubeDomain(cube_size, valid_actions)
    solved = domain.solved_state()

    from tree import SearchTree
    tree = SearchTree(domain, tree_depth)

    # test unique until second pass
    scrambler = AllScrambler(domain, tree)    
    states = set()
    for inc in range(num):
        new_pass, (s, a) = scrambler.next_instance()
        states.add(tuple(s[0]))
        assert new_pass == (inc == 0)
    assert len(states) == num
    for inc in range(num):
        _, (s, a) = scrambler.next_instance()
        states.add(tuple(s[0]))
    assert len(states) == num

    # test rewind
    scrambler = AllScrambler(domain, tree)
    states = []
    for inc in range(num):
        _, (s, a) = scrambler.next_instance()
        states.append(tuple(s[0]))

    scrambler.rewind(num//2 - 1)

    rstates = list(states[:num//2])
    for inc in range(num//2):
        _, (s, a) = scrambler.next_instance()
        rstates.append(tuple(s[0]))

    assert all([s == r for (s,r) in zip(states[:num//2], rstates[:num//2])])
    if num == 120:
        assert not all([s == r for (s,r) in zip(states[num//2:], rstates[num//2:])])

    assert set(states) == set(rstates)

