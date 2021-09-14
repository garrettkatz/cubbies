import numpy as np

class LookupNode:
    def __init__(self):
        self.value = None
        self.links = [None] * 7

class MacroDatabase:
    def __init__(self):
        self.prototypes = []
        self.wildcard_masks = []
        self.macros = []
        self.costs = []

        self.root = LookupNode()
    def query(self, state):
        node = self.root
        for k in range(len(state)):
            node = node.links[state[k]]
            if node == None: return None
        return node.value
    def add_rule(self, prototype, wildcard_mask, macro, cost):

        R = len(self.prototypes)
        self.prototypes.append(prototype)
        self.wildcard_masks.append(wildcard_mask)
        self.macros.append(macro)
        self.costs.append(cost)

        node = self.root
        for k in range(len(prototype)):
            if node.links[prototype[k]] == None:
                node.links[prototype[k]] = LookupNode()
            node = node.links[prototype[k]]
        node.value = R

if __name__ == "__main__":

    md = MacroDatabase()

    from cube import CubeDomain
    domain = CubeDomain(2)
    solved = domain.solved_state()
    
    result = md.query(solved)
    assert result == None

    md.add_rule(solved, np.zeros(solved.shape, dtype=bool), (), 0)
    result = md.query(solved)
    assert result == 0

