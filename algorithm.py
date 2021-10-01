import numpy as np

class Algorithm:
    def __init__(self, domain, bfs_tree, max_depth, color_neutral=True):
        self.domain = domain
        self.bfs_tree = bfs_tree
        self.max_depth = max_depth
        self.color_neutral = color_neutral

    def rule_search(self, macro_database, state):
        # returns result = (recolor index, actions, rule index, macro, triggering state, new_state)
        # or result = False if there is no path to a macro or solved state

        paths = self.bfs_tree.paths(up_to_depth=self.max_depth)
        solved_state = self.domain.solved_state()
    
        if self.color_neutral:
            recolorings = self.domain.color_neutral_to(state)
        else:
            recolorings = state.reshape(1, self.domain.state_size())
    
        for sym, recoloring in enumerate(recolorings):
    
            # Return macro if any descendent triggers a rule, including solve state rule
            descendents = self.bfs_tree.states_rooted_at(recoloring, up_to_depth=self.max_depth)    
            for k in range(len(paths)):
                actions, descendent = paths[k], descendents[k] 
                rule_index = macro_database.query(descendent)
                if rule_index != None:
                    macro = macro_database.macros[rule_index]
                    new_state = macro_database.apply_rule(rule_index, descendent)
                    return (sym, actions, rule_index, macro, descendent, new_state)
    
        # Failure if no path to macro found
        return False
    
    def run(self, max_actions, macro_database, state):
        # returns solved, plan, rule_indices, triggerers
        # solved: True if path to solved state was found, False otherwise
        # plan: [...,(actions, sym index, macro),...] a sequence of rule_search results
        
        # Form plan one macro at a time
        plan = []
        rules = []
        triggerers = []
        num_actions = 0
        while True:
    
            # Search for next macro
            result = self.rule_search(macro_database, state)
            
            # Return failure if none found
            if result is False: return False, plan, rules, triggerers
    
            # Execute search result
            sym, actions, rule_index, macro, triggerer, state = result
            plan.append((sym, actions, macro))
            rules.append(rule_index)
            triggerers.append(triggerer)
    
            # Fail if max actions exceeded
            num_actions += len(actions) + len(macro)
            if num_actions > max_actions: return False, plan, rules, triggerers
            
            # Terminate once solved
            if self.domain.is_solved_in(state): return True, plan, rules, triggerers

if __name__ == "__main__":

    import numpy as np

    max_depth = 2
    max_actions = 20
    color_neutral = False

    #### test rule_search
    from cube import CubeDomain
    domain = CubeDomain(3)

    from tree import SearchTree
    bfs_tree = SearchTree(domain, max_depth)

    from macro_database import MacroDatabase
    patterns = domain.perform((0,1,1), domain.solved_state()).reshape(1,-1)
    wildcards = np.zeros(patterns.shape, dtype=bool)
    macros = [((0,1,3),)]
    costs = [0]
    mdb = MacroDatabase(domain, len(macros))
    for r in range(len(macros)):
        mdb.add_rule(patterns[r], macros[r], costs[r])
        for w in range(len(wildcards[r])): mdb.disable(r, w)

    sym = 4 if color_neutral else 0
    actions = ((1,1,1),)

    solved = domain.solved_state()
    state = domain.execute(domain.reverse(macros[0]), solved)
    new_state = domain.color_neutral_to(state)[sym,:]
    invsym = domain.inverse_symmetry_of(sym)
    state = domain.execute(domain.reverse(actions), new_state)

    alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)
    result = alg.rule_search(mdb, state)

    print(result)
    assert result != False

    recolorer, path, rule, macro, triggerer, new_state = result
    recolored = domain.color_neutral_to(state)[recolorer]

    import matplotlib.pyplot as pt
    domain.render_subplot(1, 6, 1, state)
    pt.title("state")
    domain.render_subplot(1, 6, 2, recolored)
    pt.title("recolored")
    domain.render_subplot(1, 6, 3, triggerer)
    pt.title("triggerer")
    domain.render_subplot(1, 6, 4, patterns[0])
    pt.title("patterns[0]")
    domain.render_subplot(1, 6, 5, new_state)
    pt.title("new_state")
    domain.render_subplot(1, 6, 6, solved)
    pt.title("solved")
    pt.show()

    print(path)
    print(actions)
    assert path == actions
    print(recolorer, invsym)
    assert recolorer == invsym
    print(macro)
    assert macro == macros[0]
    print(new_state)
    assert (new_state == domain.solved_state()).all()

    #### test run
    from cube import CubeDomain
    domain = CubeDomain(3)

    from tree import SearchTree
    bfs_tree = SearchTree(domain, max_depth)

    import numpy as np
    from macro_database import MacroDatabase
    state = domain.solved_state()
    patterns = np.stack((
        domain.execute([(0,1,1),(1,1,1)], state),
        domain.execute([(0,1,1),(1,1,1),(2,1,1),(1,1,1),(0,1,1)], state),
    ))
    macros = (
        ((1,1,3),(0,1,3)),
        ((0,1,3),(1,1,3),(2,1,3)),
    )
    costs = [0, 0]
    mdb = MacroDatabase(domain, len(macros))
    for r in range(len(macros)):
        mdb.add_rule(patterns[r], macros[r], costs[r])
        for w in range(patterns.shape[1]): mdb.disable(r, w)

    rule_index = mdb.query(patterns[1])
    assert rule_index == 1

    assert None == mdb.query(domain.solved_state())

    actions = ((1,1,1),)
    sym = 4 if color_neutral else 0
    state = domain.execute(domain.reverse(actions), patterns[1])
    state = domain.color_neutral_to(state)[sym]
    invsym = domain.inverse_symmetry_of(sym)

    alg = Algorithm(domain, bfs_tree, max_depth, color_neutral)
    solved, plan, rules, triggerers = alg.run(max_actions, mdb, state)
    
    assert solved
    s, path, macro = plan[0]
    assert path == actions
    assert s == invsym
    assert macro == macros[1]
    assert rules[0] == 1
    assert domain.is_solved_in(domain.execute(macros[rules[-1]], triggerers[-1]))

    import matplotlib.pyplot as pt
    def draw(st, title, i):
        ax = pt.subplot(4, 6, i)
        domain.render(st, ax, 0, 0)
        ax.axis("equal")
        ax.axis('off')
        ax.set_title(title)

    i = 1
    draw(state, "initial", i)
    i += 1
    for (sym, actions, macro) in plan:
        print(sym)
        print(actions)
        print(macro)

        state = domain.color_neutral_to(state)[sym]
        draw(state, str(sym), i)
        i += 1

        for action in actions:
            state = domain.perform(action, state)
            draw(state, str(action), i)
            i += 1

        # state = domain.color_neutral_to(state)[sym]
        # draw(state, str(sym), i)
        # i += 1

        for action in macro:
            state = domain.perform(action, state)
            draw(state, str(action), i)
            i += 1

    pt.show()

