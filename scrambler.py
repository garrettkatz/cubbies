import numpy as np

class AllScrambler:

    def __init__(self, domain, tree):
        self.domain = domain
        self.tree = tree
        self.idx = [np.random.permutation(tree.size())]
        self.inc = 0

    def next_instance(self):

        # permute more indices when needed
        if self.inc == len(self.idx) * self.tree.size():
            self.idx.append(np.random.permutation(tree.size()))
                
        # get next scrambled index
        i = self.idx[int(self.inc / tree.size())][self.inc % tree.size()]
        self.inc += 1

        # return next scramble
        s0 = self.domain.solved_state()[self.tree.permutations()[i]]
        a = domain.reverse(tree.paths()[i])
        s = [s0] + domain.intermediate_states(a, s0)
        return s, a

    def rewind(self, inc):
        # up to and including inc

        # get last pass and iteration in that pass at inc
        p = int(inc / tree.size())
        n = inc % tree.size()

        # discard subsequent passes
        self.idx = self.idx[:p+1]

        # rescramble remainder of pass
        self.idx[p][n+1:] = np.random.permutation(self.idx[p][n+1:])

        # reset inc
        self.inc = inc+1

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
        s, a = scrambler.next_instance()
        states.add(tuple(s[0]))
    assert len(states) == num
    for inc in range(num):
        s, a = scrambler.next_instance()
        states.add(tuple(s[0]))
    assert len(states) == num

    # test rewind
    scrambler = AllScrambler(domain, tree)
    states = []
    for inc in range(num):
        s, a = scrambler.next_instance()
        states.append(tuple(s[0]))

    scrambler.rewind(num//2 - 1)

    rstates = list(states[:num//2])
    for inc in range(num//2):
        s, a = scrambler.next_instance()
        rstates.append(tuple(s[0]))

    assert all([s == r for (s,r) in zip(states[:num//2], rstates[:num//2])])
    if num == 120:
        assert not all([s == r for (s,r) in zip(states[num//2:], rstates[num//2:])])

    assert set(states) == set(rstates)

