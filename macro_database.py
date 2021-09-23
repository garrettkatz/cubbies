"""
invariants:
- distinct prototype prefixes have distinct node in the prefix graph, regardless of wildcard matches
- if a node has two or more distinct child nodes, all the node's links are tame

implies that any node with wild links has exactly one child node,
and all rules descending from node match without wildcards through the child prefix

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
    def __init__(self, bound, value=None):
        self.bound = bound
        self.value = value
        self.rule = None
        self.links = {}
    
    def is_wild(self):
        child_ids = set(map(id, self.links.values()))
        return len(self.links) == self.bound and len(child_ids) == 1

    def tame(self):
        self.links = {value:node for value, node in self.links.items() if node.value == value}

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
        self.wildcards = np.empty((max_rules, len(bounds)), dtype=bool)
        self.costs = np.empty(max_rules, dtype=int)
        self.macros = [None] * max_rules

    def query(self, state):
        node = self.root
        for k,v in enumerate(state):
            if v not in node.links: return None
            node = node.links[v]
        return node.rule

    def tame(self, node, k):
        node.tame()
        for r in node.rules(): self.wildcards[r,k] = False

    def add_rule(self, prototype, macro, cost):
        r = self.num_rules
        self.prototypes[r] = prototype
        self.costs[r] = cost
        self.macros[r] = macro

        node = self.root
        for k,v in enumerate(prototype):
            child_bound = self.bounds[k+1] if k+1 < len(self.bounds) else 0
            if len(node.links) == 0:
                child_node = PrefixTreeNode(child_bound, value=v)
                node.links = {v: child_node for v in range(node.bound)}
            elif v not in node.links:
                node.links[v] = PrefixTreeNode(child_bound, value=v)
            elif node.links[v].value != v:
                self.tame(node, k)
                node.links[v] = PrefixTreeNode(child_bound, value=v)
            self.wildcards[r,k] = node.is_wild()
            node = node.links[v]
        node.rule = r

        self.num_rules += 1

    def disable(self, r, w):
        node = self.root
        for k in range(w): node = node.links[self.prototypes[r,k]]
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
    assert (md.wildcards[:md.num_rules] == True).all()
    print("-"*24)
    print(md.root)

    # disable all wildcards for solved
    for w in range(len(solved)): md.disable(0, w)

    assert md.query(solved) == 0
    assert (md.wildcards[:md.num_rules] == False).all()
    print("-"*24)
    print(md.root)

    state = domain.perform((0,1,1), solved)
    md.add_rule(state, ((0,1,3)), 1)

    print("-"*24)
    print(md.root)
    assert md.query(solved) == 0
    assert md.query(state) == 1
    assert not (md.wildcards[:md.num_rules] == True).all()
    assert not (md.wildcards[:md.num_rules] == False).all()

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
        brutes = np.flatnonzero(((md.prototypes == state) | md.wildcards).all(axis=1))
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
        brutes = np.flatnonzero(((md.prototypes == state) | md.wildcards).all(axis=1))
        print(r, result, brutes)
        if result not in brutes:
            print("postdisable")
            print("-"*24)
            print(md.root)
        assert result in brutes

    print("-"*24)
    print(md.root)

    rule_count = 5000

    md = MacroDatabase(max_rules=rule_count, bounds=(7,)*len(solved))
    md.add_rule(solved, (), 0)
    for w in range(len(solved)): md.disable(0, w)

    rng = np.random.default_rng()
    for r in range(1, rule_count):
        state = domain.random_state(20, rng)
        md.add_rule(state, (), 0)

    from time import perf_counter
    prefix_time = 0
    brute_time = 0

    for state in md.prototypes:

        start = perf_counter()
        result = md.query(state)
        prefix_time += perf_counter() - start

        start = perf_counter()
        result = np.flatnonzero(((state == md.prototypes) | md.wildcards).all(axis=1))
        brute_time += perf_counter() - start

    print("prototype queries:")
    print("prefix time", prefix_time)
    print("brute time", brute_time)

    prefix_time = 0
    brute_time = 0

    for s in range(rule_count):
        state = domain.random_state(20, rng)

        start = perf_counter()
        result = md.query(state)
        prefix_time += perf_counter() - start

        start = perf_counter()
        result = np.flatnonzero(((state == md.prototypes) | md.wildcards).all(axis=1))
        brute_time += perf_counter() - start

    print("random state queries:")
    print("prefix time", prefix_time)
    print("brute time", brute_time)

