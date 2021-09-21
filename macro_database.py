"""
invariants:
- any node with two or more distinct child nodes must be tame (no wild links)
- all descendent rules of a tamed node must have wildcard disabled at that position
- except for root, wild nodes never have None links, and all links point to same child node
- any two distinct prototype prefixes must have distinct node paths in the prefix tree

if you're adding a new rule, nobody matched prototype (early tame None leaf)
if you're disabling a wildcard, someone matched it (wild node with non-None links)

should never take some wild links and then fail unmatched if a different non-wild link would have matched
all prototypes should have persistent trails with no wildcard links
that way, disabling wildcards for one rule does not break another rule

"""
import numpy as np

class PrefixTreeNode:
    # def __init__(self, bound, wild=True, pval=None):
    def __init__(self, bound, pval=None):
        self.bound = bound
        self.pval = pval
        self.rule = None
        self.links = {}
    
        # self.rule = None
        # self.links = [None] * bound
        # self.wild = wild
        # self.pval = pval # prototype value leading to this node

    def is_wild(self):
        child_ids = set(map(id, self.links.values()))
        return len(self.links) == self.bound and len(child_ids) == 1

    def tame(self):
        self.links = {v:c for v,c in self.links.items() if c.pval == v}

    def __str__(self, prefix=""):
        if self.rule != None: return "%sr%d\n" % (prefix, self.rule)
        if self.is_wild():
            result = prefix + "*\n"
            result += self.links[0].__str__(prefix+" ")
        else:
            result = ""
            p = " " if len(self.links) == 1 else "|"
            for v in range(self.bound):
                if v in self.links:
                    result += prefix + "%d\n" % v
                    result += self.links[v].__str__(prefix+p)
        return result
        # if self.rule != None: return "%sr%d\n" % (prefix, self.rule)
        # if self.wild:
        #     result = prefix + "*\n"
        #     if self.links[-1] != None:
        #         result += self.links[-1].__str__(prefix+" ")
        # else:
        #     result = ""
        #     p = " " if sum([node != None for node in self.links]) == 1 else "|"
        #     for n,node in enumerate(self.links):
        #         if node != None:
        #             result += prefix + "%d\n" % n
        #             result += node.__str__(prefix+p)
        # return result
    def rules(self):
        if self.rule != None: return [self.rule]
        if self.is_wild(): return self.links[0].rules()
        result = []
        for v in range(self.bound):
            if v in self.links:
                result += self.links[v].rules()
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

    def query(self, state, verbose=False):
        node = self.root
        for k,v in enumerate(state):
            if verbose: print(k, v)
            if v not in node.links: return None
            node = node.links[v]
        return node.rule

    def tame(self, node, k):
        node.tame()
        for r in node.rules(): self.wildcard_masks[r,k] = False

    def add_rule(self, prototype, macro, cost):
        r = self.num_rules
        self.prototypes[r] = prototype
        self.costs[r] = cost
        self.macros[r] = macro
        self.num_rules += 1

        node = self.root
        for k,v in enumerate(prototype):
            child_bound = self.bounds[k+1] if k+1 < len(self.bounds) else 0
            if len(node.links) == 0:
                child_node = PrefixTreeNode(child_bound, pval=v)
                node.links = {v: child_node for v in range(node.bound)}
            else:                
                if v not in node.links:
                    node.links[v] = PrefixTreeNode(child_bound, pval=v)
                elif node.links[v].pval != v:
                    self.tame(node, k)
                    node.links[v] = PrefixTreeNode(child_bound, pval=v)
            self.wildcard_masks[r,k] = node.is_wild()
            node = node.links[v]
        node.rule = r

    def disable(self, r, w):
        # get node to be tamed
        node = self.root
        for k in range(w): node = node.links[self.prototypes[r,k]]
        # tame it if still wild
        if node.is_wild(): self.tame(node, w)

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

    print("-"*24)
    print(md.root)
    assert md.query(solved) == 0
    assert md.query(state) == 1
    assert not (md.wildcard_masks[:md.num_rules] == True).all()
    assert not (md.wildcard_masks[:md.num_rules] == False).all()

    state2 = domain.perform((0,1,2), solved)
    md.add_rule(state2, ((0,1,2)), 1)
    print("-"*24)
    print(md.root)

    state3 = domain.perform((1,1,1), solved)
    md.add_rule(state3, ((1,1,3)), 1)
    print("-"*24)
    print(md.root)
    
    # simulate adding rules and check queries match after
    md = MacroDatabase(max_rules=10, bounds=(7,)*len(solved))
    md.add_rule(solved, (), 0)
    for w in range(len(solved)): md.disable(0, w)

    rng = np.random.default_rng()
    for r in range(1, 10):
        state = domain.random_state(20, rng)
        md.add_rule(state, (), 0)

    for r in range(md.num_rules):
        state = md.prototypes[r]
        result = md.query(state)
        brutes = np.flatnonzero(((md.prototypes == state) | md.wildcard_masks).all(axis=1))
        print(r, result, brutes)
        if result not in brutes:
            print("predisable")
            print("-"*24)
            print(md.root)
        assert result in brutes

    for r in range(md.num_rules):    
        for w in range(len(state)):            
            if rng.uniform() < 0.1:
                md.disable(r, w)

    for r in range(md.num_rules):
        state = md.prototypes[r]
        result = md.query(state)
        brutes = np.flatnonzero(((md.prototypes == state) | md.wildcard_masks).all(axis=1))
        print(r, result, brutes)
        if result not in brutes:
            md.query(state, verbose=True)
            print("postdisable")
            print("-"*24)
            print(md.root)
        assert result in brutes

    print("-"*24)
    print(md.root)


