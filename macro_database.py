"""
invariants:
- any node with multiple child nodes must be tame (no wild links)
- all descendent rules of a tamed node must have wildcard disabled at that position
- except for root, wild nodes never have None links, and all links point to same child node

if you're adding a new rule, nobody matched prototype (early tame None leaf)
if you're disabling a wildcard, someone matched it (wild node with non-None links

should never take some wild links and then fail unmatched if a different non-wild link would have matched
all prototypes should have persistent trails with no wildcard links
that way, disabling wildcards for one rule does not break another rule

"""
import numpy as np

class PrefixTreeNode:
    def __init__(self, bound):
        self.rule = None
        self.links = [None] * bound
        self.wild = True
    def __str__(self, prefix=""):
        if self.rule != None: return "%sr%d\n" % (prefix, self.rule)
        if self.wild:
            result = prefix + "*\n"
            if self.links[-1] != None:
                result += self.links[-1].__str__(prefix+" ")
        else:
            result = ""
            p = " " if sum([node != None for node in self.links]) == 1 else "|"
            for n,node in enumerate(self.links):
                if node != None:
                    result += prefix + "%d\n" % n
                    result += node.__str__(prefix+p)
        return result
    def rules(self):
        if self.rule != None: return [self.rule]
        result = []
        for node in self.links:
            if node != None:
                result += node.rules()
                if node.wild: break
        return result

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
        return node.rule

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
        node.rule = self.num_rules

        self.prototypes[self.num_rules] = prototype
        self.wildcard_masks[self.num_rules] = wildcard_mask
        self.costs[self.num_rules] = cost
        self.macros[self.num_rules] = macro

        self.num_rules += 1

    def disable(self, r, w):
        # get node to be tamed
        node = self.root
        for k in range(w):
            node = node.links[self.prototypes[r,k]]
        # tame it in the prefix tree
        for v in range(len(node.links)):
            if v != self.prototypes[r,w]: node.links[v] = None
        node.wild = False
        # update wildcard masks
        for rule in node.rules():
            self.wildcard_masks[rule,w] = False

if __name__ == "__main__":


    from cube import CubeDomain
    domain = CubeDomain(2)
    solved = domain.solved_state()

    md = MacroDatabase(max_rules=10, bounds=(7,)*len(solved))
    
    result = md.query(solved)
    assert result == None
    print(md.root)

    md.add_rule(solved, (), 0)
    assert md.query(solved) == 0
    assert (md.wildcard_masks[:md.num_rules] == True).all()
    print("-"*24)
    print(md.root)

    # disable all wildcards for solved
    for w in range(len(solved)): md.disable(0, w)

    assert md.query(solved) == 0
    assert (md.wildcard_masks[:md.num_rules] == False).all()
    print("-"*24)
    print(md.root)

    state = domain.perform((0,1,1), solved)
    md.add_rule(state, ((0,1,3)), 1)

    assert md.query(solved) == 0
    assert md.query(state) == 1
    assert not (md.wildcard_masks[:md.num_rules] == True).all()
    assert not (md.wildcard_masks[:md.num_rules] == False).all()
    print("-"*24)
    print(md.root)

    state2 = domain.perform((0,1,2), solved)
    md.add_rule(state2, ((0,1,2)), 1)
    print("-"*24)
    print(md.root)

    state3 = domain.perform((1,1,1), solved)
    md.add_rule(state3, ((1,1,3)), 1)
    print("-"*24)
    print(md.root)
    

    


