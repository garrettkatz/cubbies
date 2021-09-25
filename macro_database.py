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
    def __init__(self, bound, value=None, added=0):
        self.bound = bound # maximum number of outgoing links
        self.value = value # value of incoming link
        self.added = added # integer timestamp when node is added
        self.tamed = np.iinfo(int).max # integer timestamp when node is tamed
        self.rule = None # for leaf nodes, the associated rule
        self.links = {} # links[v] = child node for value v
    
    def is_wild(self):
        child_ids = set(map(id, self.links.values()))
        return len(self.links) == self.bound and len(child_ids) == 1

    def tame(self, tamed=0):
        self.links = {value:node for value, node in self.links.items() if node.value == value}
        self.tamed = tamed

    def __str__(self, prefix=""):
        tamed = "" if self.tamed == np.iinfo(int).max else str(self.tamed)
        if self.rule != None: return "%sr%d [%d~>%s]\n" % (prefix, self.rule, self.added, tamed)
        if self.is_wild():
            result = prefix + "* [%d~>%s]\n" % (self.added, tamed)
            result += self.links[0].__str__(prefix+" ")
        else:
            result = ""
            p = " " if len(self.links) == 1 else "|"
            for v in range(self.bound):
                if v in self.links:
                    result += prefix + "%d [%d~>%s]\n" % (v, self.added, tamed)
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

    def copy(self):
        node = PrefixTreeNode(self.bound, self.value, self.added)
        node.tamed = self.tamed
        node.rule = self.rule
        if self.is_wild():
            child_node = self.links[0].copy()
            node.links = {v: child_node for v in range(self.bound)}
        else:
            node.links = {v: child_node.copy() for v, child_node in self.links.items()}
        return node

    def rewind(self, i):

        # Rewind current node and links
        self.links = {v: node for v, node in self.links.items() if node.added <= i}
        if self.tamed > i and len(self.links) > 0: # should imply len == 1
            _, node = self.links.popitem()
            for v in range(self.bound): self.links[v] = node
            self.tamed = np.iinfo(int).max

        # Recurse on children
        if self.is_wild():
            self.links[0].rewind(i)
        else:
            for child_node in self.links.values(): child_node.rewind(i)

class MacroDatabase:
    def __init__(self, domain, max_rules):

        self.domain = domain
        self.max_rules = max_rules
        self.num_rules = 0
        self.bounds = (7,) * domain.state_size()
        self.root = PrefixTreeNode(self.bounds[0])

        self.prototypes = np.empty((max_rules, domain.state_size()), dtype=int)
        self.wildcards = np.empty((max_rules, domain.state_size()), dtype=bool)
        self.costs = np.empty(max_rules, dtype=int)
        self.macros = [None] * max_rules
        self.permutations = np.empty((max_rules, domain.state_size()), dtype=int)

        self.added = np.ones(max_rules, dtype=int) * np.iinfo(int).max
        self.tamed = np.ones((max_rules, domain.state_size()), dtype=int) * np.iinfo(int).max

    def query(self, state):
        node = self.root
        for k,v in enumerate(state):
            if v not in node.links: return None
            node = node.links[v]
        return node.rule

    def tame(self, node, w, tamed=0):
        node.tame(tamed)
        for r in node.rules():
            self.wildcards[r,w] = False
            self.tamed[r,w] = tamed

    def add_rule(self, prototype, macro, cost, added=0):
        r = self.num_rules
        self.prototypes[r] = prototype
        self.costs[r] = cost
        self.macros[r] = macro
        self.permutations[r] = self.domain.execute(macro, np.arange(self.domain.state_size()))
        self.added[r] = added

        node = self.root
        for k, value in enumerate(prototype):
            child_bound = self.bounds[k+1] if k+1 < len(self.bounds) else 0
            if len(node.links) == 0:
                child_node = PrefixTreeNode(child_bound, value, added)
                node.links = {v: child_node for v in range(node.bound)}
            elif value not in node.links:
                node.links[value] = PrefixTreeNode(child_bound, value, added)
            elif node.links[value].value != value:
                self.tame(node, k, tamed=added)
                node.links[value] = PrefixTreeNode(child_bound, value, added)
            self.wildcards[r,k] = node.is_wild()
            node = node.links[value]
        node.rule = r

        self.num_rules += 1

    def disable(self, r, w, tamed=0):
        node = self.root
        for k in range(w): node = node.links[self.prototypes[r,k]]
        if node.is_wild(): self.tame(node, w, tamed)

    def apply_rule(self, r, state):
        return state[self.permutations[r]].copy()

    def copy(self):
        db = MacroDatabase(self.domain, self.max_rules)
        db.num_rules = self.num_rules
        db.root = self.root.copy()

        db.prototypes = self.prototypes.copy()
        db.wildcards = self.wildcards.copy()
        db.costs = self.costs.copy()
        db.macros = list(self.macros)
        db.permutations = self.permutations.copy()

        db.added = self.added.copy()
        db.tamed = self.tamed.copy()
        
        return db

    def rewind(self, i):
        self.root.rewind(i)
        self.num_rules = np.argmax(self.added > i)
        self.added[self.num_rules:] = np.iinfo(int).max        
        self.wildcards = self.tamed > i
        self.tamed[self.wildcards] = np.iinfo(int).max

if __name__ == "__main__":


    from cube import CubeDomain
    domain = CubeDomain(2)
    solved = domain.solved_state()

    md = MacroDatabase(domain, max_rules=10)
    
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
    md.add_rule(state, ((0,1,3),), 1)

    print("-"*24)
    print(md.root)
    assert md.query(solved) == 0
    assert md.query(state) == 1
    assert not (md.wildcards[:md.num_rules] == True).all()
    assert not (md.wildcards[:md.num_rules] == False).all()

    state2 = domain.perform((0,1,2), solved)
    md.add_rule(state2, ((0,1,2),), 1)
    print("-"*24)
    print(md.root)

    state3 = domain.perform((1,1,1), solved)
    md.add_rule(state3, ((1,1,3),), 1)
    print("-"*24)
    print(md.root)
    
    # simulate adding rules and check queries match after
    md = MacroDatabase(domain, max_rules=10)
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

    # test rewinding
    md = MacroDatabase(domain, max_rules=10)
    md.add_rule(prototype=solved, macro=(), cost=0, added=0)
    for w in range(len(solved)): md.disable(0, w, tamed=0)

    rng = np.random.default_rng()
    for r in range(1, 5):
        state = domain.random_state(20, rng)
        md.add_rule(prototype=state, macro=(), cost=0, added=r)
        md.disable(rng.integers(r, endpoint=True), rng.integers(len(state)), tamed=r)

    for r in range(md.num_rules):
        state = md.prototypes[r]
        result = md.query(state)
        brutes = np.flatnonzero(((md.prototypes == state) | md.wildcards).all(axis=1))
        if result not in brutes:
            print("prerewind")
            print("-"*24)
            print(md.root)
        assert result in brutes

    print("prerewind")
    print("-"*24)
    print(md.root)

    md.rewind(3)
    md.rewind(2)
    for r in range(md.num_rules):
        state = md.prototypes[r]
        result = md.query(state)
        brutes = np.flatnonzero(((md.prototypes == state) | md.wildcards).all(axis=1))
        if result not in brutes:
            print("postrewind")
            print("-"*24)
            print(md.root)
        assert result in brutes

    print("postrewind")
    print("-"*24)
    print(md.root)

    # test copy
    md = MacroDatabase(domain, max_rules=10)
    md.add_rule(prototype=solved, macro=(), cost=0, added=0)
    for w in range(len(solved)): md.disable(0, w, tamed=0)

    rng = np.random.default_rng()
    num_rules = 4
    states = []
    for r in range(1, num_rules-1):
        state = domain.random_state(20, rng)
        md.add_rule(prototype=state, macro=(), cost=0, added=r)
        md.disable(rng.integers(r, endpoint=True), rng.integers(len(state)), tamed=r)
        states.append(state)

    md2 = md.copy()
    state = domain.random_state(20, rng)
    md2.add_rule(prototype=state, macro=(), cost=0, added=num_rules-1)
    for r in range(md.num_rules):
        for w in range(len(state)):
            md.disable(r, w, tamed=num_rules)

    print("orig")
    print("-"*24)
    print(md.root)
    print("copy")
    print("-"*24)
    print(md2.root)

    assert md.num_rules + 1 == md2.num_rules
    assert md.wildcards[:md.num_rules].sum() == 0
    assert md2.wildcards[:md.num_rules].sum() > 0
    assert md.query(state) == None
    assert md2.query(state) != None
    for state in states:
        assert md.query(state) != None
        assert md2.query(state) != None

    # test macro permutations
    md = MacroDatabase(domain, max_rules=1)
    state = domain.perform((0, 1, 1), solved)
    actions = ((1, 1, 1), (2, 1, 1))
    md.add_rule(prototype=solved, macro=actions, cost=0, added=0)
    for w in range(len(solved)):
        if state[w] == solved[w]: md.disable(0, w, tamed=0)

    print("one rule perm")
    print("-"*24)
    print(md.root)

    r = md.query(state)
    assert r == 0
    new_state = md.apply_rule(r, state)
    for action in actions: state = domain.perform(action, state)
    assert (state == new_state).all()

    # compare timing of prefix and brute queries
    rule_count = 1000

    md = MacroDatabase(domain, max_rules=rule_count)
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

