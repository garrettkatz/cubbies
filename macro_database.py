"""
if you're adding a new rule, nobody matched prototype (early None leaf)
if you're toggling a wildcard, someone matched it (wildcard link to remove)

should never take some wild links and then fail unmatched if a different non-wild link would have matched

all prototypes should have persistent trails with no wildcard links
that way, disabling wildcards for one rule does not break another rule

invariants:
- any node with multiple child nodes must be tame (no wild links)
- all descendent rules of a tamed node must have wildcard disabled at that position

"""
import numpy as np

class PrefixTreeNode:
    def __init__(self, bound):
        self.value = None
        self.links = [None] * bound
        self.wild = True

class MacroDatabase:
    def __init__(self, max_rules, bounds):

        self.num_rules = 0
        self.bounds = tuple(bounds)
        self.root = PrefixTreeNode(bounds[0])

        self.prototypes = np.empty((max_rules, len(bounds)), dtype=int)
        self.wildcard_masks = np.empty((max_rules, len(bounds)), dtype=bool)
        self.costs = np.empty(max_rules, dtype=int)
        self.macros = [None] * max_rules

    def query(self, state):
        node = self.root
        for k in range(len(state)):
            node = node.links[state[k]]
            if node == None: return None
        return node.value

    def add_rule(self, prototype, macro, cost):
        wildcard_mask = np.empty(len(prototype), dtype=bool)

        node = self.root
        for k,v in enumerate(prototype):
            if node.links[v] == None:
                bound = 0 if k+1 == len(prototype) else self.bounds[k+1]
                node.links[v] = PrefixTreeNode(bound)
                if node.wild:
                    for u in range(self.bounds[k]):
                        node.links[u] = node.links[v]
            wildcard_mask[k] = node.wild
            node = node.links[v]
        node.value = self.num_rules

        self.prototypes[self.num_rules] = prototype
        self.wildcard_masks[self.num_rules] = wildcard_mask
        self.costs[self.num_rules] = cost
        self.macros[self.num_rules] = macro

        self.num_rules += 1

    def toggle(self, r, w):
        pass

if __name__ == "__main__":


    from cube import CubeDomain
    domain = CubeDomain(2)
    solved = domain.solved_state()

    md = MacroDatabase(max_rules=10, bounds=(7,)*len(solved))
    
    result = md.query(solved)
    assert result == None

    md.add_rule(solved, (), 0)
    result = md.query(solved)
    assert result == 0

    state = domain.perform((0,1,1), solved)
    md.add_rule(state, ((0,1,3)), 1)

    result = md.query(state)
    assert result == 1

    result = md.query(solved)
    assert result == 1
    
    assert (md.wildcard_masks[:md.num_rules] == True).all()


