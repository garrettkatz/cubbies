"""
Memoize a depth-limited, BFS search tree for the cube domain
Paths and resulting permutations to each node are precomputed
"""
import numpy as np

class SearchTree:

    def __init__(self, domain, max_depth, orientation_neutral=False):

        permutation = np.arange(domain.state_size())
        actions = tuple()
        
        explored = set([permutation.tobytes()])
        layers = {0: [(actions, permutation)]}

        all_permutations = [permutation]
        all_paths = [actions]
        
        for depth in range(max_depth):
            layers[depth+1] = []

            for path, permutation in layers[depth]:
                for action in domain.valid_actions(permutation):

                    # get child state permutation
                    new_path = path + (action,)
                    new_permutation = domain.perform(action, permutation)
                    
                    # skip if already explored
                    if orientation_neutral:
                        orientations = domain.orientations_of(new_permutation)
                        if any([perm.tobytes() in explored for perm in orientations]): continue
                    else:
                        if new_permutation.tobytes() in explored: continue

                    # otherwise add the new state to the frontier and explored set
                    explored.add(new_permutation.tobytes())
                    layers[depth+1].append((new_path, new_permutation))

                    all_permutations.append(new_permutation)
                    all_paths.append(new_path)

        self._permutations = np.stack(all_permutations)
        self._paths = tuple(all_paths)
        self._count = np.cumsum([len(layers[depth]) for depth in range(max_depth+1)])
        
        # self._layers = layers

    def depth(self):
        return len(self._count) - 1

    def size(self):
        return self._count[-1]

    def __iter__(self):
        return zip(self._paths, self._permutations)

    def rooted_at(self, state, up_to_depth=None):
        if up_to_depth is None: up_to_depth = self.depth()
        states = state.take(self._permutations[:self._count[up_to_depth]])
        paths = self._paths[:self._count[up_to_depth]]
        return zip(paths, states)
    
    def paths(self, up_to_depth=None):
        if up_to_depth is None: up_to_depth = self.depth()
        return self._paths[:self._count[up_to_depth]]

    def permutations(self, up_to_depth=None):
        if up_to_depth is None: up_to_depth = self.depth()
        return self._permutations[:self._count[up_to_depth]]

    def states_rooted_at(self, state, up_to_depth=None):
        if up_to_depth is None: up_to_depth = self.depth()
        states = state.take(self._permutations[:self._count[up_to_depth]])
        return states

if __name__ == "__main__":

    import itertools as it
    # cube_size, num_twist_axes, quarter_turns = 2, 3, True # maxdep 11, 3.6M states
    # # cube_size, num_twist_axes, quarter_turns = 2, 3, False # maxdep 3, 24 states
    # # cube_size, num_twist_axes, quarter_turns = 2, 2, True # maxdep 13, 29160 states
    # # cube_size, num_twist_axes, quarter_turns = 2, 2, False # maxdep 2, 6 states
    # valid_actions = tuple(it.product(range(num_twist_axes), range(1,cube_size), range(2-quarter_turns, 4, 2-quarter_turns)))

    # pocket cube: one axis with quarter twists, one with half twists
    # 120 states, reached in max_depth=10
    cube_size = 2
    valid_actions = (
        (0,1,1), (0,1,2), (0,1,3),
        (1,1,2), 
    )

    # # pocket cube: one axis with quarter twists, two with half twists
    # # 5040 states, reached in max_depth=13
    # cube_size = 2
    # valid_actions = (
    #     (0,1,1), (0,1,2), (0,1,3),
    #     (1,1,2),
    #     (2,1,2),
    # )

    from cube import CubeDomain
    domain = CubeDomain(cube_size, valid_actions)
    A = len(list(domain.valid_actions(domain.solved_state())))

    # tree = SearchTree(domain, max_depth=2)
    # print(tree.layers)

    tree = SearchTree(domain, max_depth=10) # 5 uses up 4+ GB memory for 3cube
    print("depth, total nodes, nodes at depth, A**depth")
    for depth in range(len(tree._count)-1):
        print(depth+1, tree._count[depth+1], tree._count[depth+1] - tree._count[depth], A**(depth+1))

    # #### profile different tree iteration methods
    # # slow
    # for rep in range(10):
    #     x = 0
    #     for action, permutation in tree:
    #         x += (permutation > 10).sum()
    # print(x)

    # # fast
    # for rep in range(10):
    #     x = (tree.permutations() > 10).sum()
    # print(x)

    # for rep in range(20):    
    #     states_slow = np.stack([state for path, state in tree.rooted_at(domain.solved_state())])
    # print("slow", states_slow.shape)
    # for rep in range(20):
    #     states_fast = tree.states_rooted_at(domain.solved_state())
    # print("fast", states_fast.shape)
    # assert (states_slow == states_fast).all()

    # # measure color neutrality savings
    # for depth in range(tree.depth()+1):
    #     explored_neutral = set()
    #     states = tree.states_rooted_at(domain.solved_state(), up_to_depth=depth)
    #     for state in states:
    #         if not any([neut.tobytes() in explored_neutral for neut in domain.color_neutral_to(state)]):
    #             explored_neutral.add(state.tobytes())
    #     print(depth, len(explored_neutral), len(states))

    # tree = paths(domain, new_actions)

    # for actions, permutation in tree:
    #     print(actions, permutation)

    # # iterate over search tree object and visually inspect states and their order, esp those more than 1 action away
    # tree = SearchTree(domain, max_depth=2)
    # import matplotlib.pyplot as pt
    # for n, (actions, permutation) in enumerate(tree):
    #     if n == 70: break
    #     state = domain.solved_state()[permutation]
    #     ax = pt.subplot(7, 10, n+1)
    #     domain.render(state, ax, 0, 0)
    #     ax.axis("equal")
    #     ax.axis('off')
    #     ax.set_title(str(actions))
    # pt.show()

    # #### check whether color-neutral trees are isomorphic for any starting state
    # # the answer is NO.  also, not much smaller than non-neutral at shallow trees
    # # results (number of depth 4 tree nodes):
    # # solved_state, not neutral: 174604
    # # solved_state, yes neutral: 174491
    # # random_state(20), not neutral: 174604
    # # random_state(20), yes neutral: 174604 almost always, once was 174601
    # # random_state(1 or 2), yes neutral: around 174390
    # tree = SearchTree(domain, max_depth=4)
    # init = domain.random_state(2, np.random.default_rng())
    # # init = domain.solved_state()
    # color_neutral = True
    # # color_neutral = False

    # for rep in range(100):
    #     explored = set()
    #     cn = []
    #     for n, (actions, state) in enumerate(tree.rooted_at(init)):
    #         if color_neutral:
    #             recolorings = domain.recolorings_of(state)
    #             if any([state.tobytes() in explored for state in recolorings]): continue
    #         explored.add(state.tobytes())
    #         cn.append((actions, state))
    #     print(len(cn))

    # #### count distinct action sequences in tree
    # tree = SearchTree(domain, max_depth=3)
    # action_sequences = [a[1:] for a, _ in tree if len(a) > 1]
    # print(len(action_sequences))
    # print(len(set(action_sequences)))

    # #### count distinct action subsequences in tree
    # # this is also the number of distinct states in the tree, which makes sense in hindsight
    # import itertools as it
    # tree = SearchTree(domain, max_depth=4)
    # distinct = set()
    # repeated = 0
    # for actions, _ in tree:
    #     for lo,hi in it.combinations(range(len(actions)+1), 2):
    #         distinct.add(actions[lo:hi])
    #         repeated += 1
    # print(len(distinct))
    # print(repeated)

    # #### check frequencies and wildcards of each action subsequences in tree
    # # only lo (no hi) since hi < len(actions) is in a different branch to state reached at hi
    # # depth 4 results:
    # # distinct len-1 macros range from 5711 to 7125 occurences each,
    # # len-2 from 153 to 390, len 3 from 3 to 21, len 4 from 1 to 1 each
    # # number of invariant faces after distinct macros:
    # # almost constant for a fixed macro length, with a few outliers at larger depth
    # # len-1, len-2 have 0 invariant faces
    # # some len-3 macros have 27/54 invariant facies in their set of terminal states
    # # len-4 macros have 54 invariants but only because their sets of terminal states are singletons
    # import itertools as it
    # max_depth = 4
    # init = domain.random_state(20, np.random.default_rng())
    # # init = domain.solved_state()
    # tree = SearchTree(domain, max_depth)
    # distinct = {}
    # for actions, state in tree.rooted_at(init):
    #     for lo in range(len(actions)):
    #         if actions[lo:] not in distinct: distinct[actions[lo:]] = []
    #         distinct[actions[lo:]].append(state)

    # import matplotlib.pyplot as pt
    # data = [
    #     np.array([len(states) for macro, states in distinct.items() if len(macro) == k])
    #     for k in range(max_depth+1)]
    # for k in range(1,max_depth+1):
    #     print(k, data[k].min(), data[k].max(), data[k].mean())
    # pt.subplot(1, 2, 1)
    # pt.hist(data)
    # pt.xlabel("Occurrences of distinct macro")
    # pt.ylabel("Frequency")
    # pt.legend([str(k) for k in range(max_depth+1)])

    # data = [list() for _ in range(max_depth+1)]
    # for macro, states in distinct.items():
    #     state_array = np.array(states)
    #     invariants = (state_array == state_array[0]).all(axis=0)
    #     data[len(macro)].append(invariants.sum())
    # print("\nstate size = %d" % domain.state_size())
    # for k in range(1,max_depth+1):
    #     data[k] = np.array(data[k])
    #     print(k, data[k].min(), data[k].max(), data[k].mean())

    # pt.subplot(1, 2, 2)
    # pt.hist(data)
    # pt.xlabel("Number of invariants after distinct macro")
    # pt.ylabel("Frequency")
    # pt.legend([str(k) for k in range(max_depth+1)])

    # pt.show()

    # #### profile
    # import itertools as it
    # valid_actions = tuple(it.product((0,1,2), (0,), (0, 1, 2, 3))) # only spinning one plane on each axis for 2cube

    # from cube import CubeDomain
    # domain = CubeDomain(2, valid_actions)
    # init = domain.solved_state()

    # tree = SearchTree(domain, 5)
    # paths, states = zip(*tree.rooted_at(init))

    # def prof(s):
    #     for _, neighbor in tree.rooted_at(states[s], up_to_depth=1):
    #         dumb = (np.arange(24) == np.arange(24*500).reshape(500, 24)).all(axis=1)

    # for s in range(1000):
    #     # print(s)
    #     prof(s)

