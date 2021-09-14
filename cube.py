"""
NxNxN rubiks cube
state is a NxNxNx3 array
first 3 dimensions are positions on the cube
last dimension is the colors in each spatial direction
spatial directions are 0:x, 1:y, 2:z
"""
import itertools as it
import numpy as np
import matplotlib.pyplot as pt
from matplotlib.patches import Polygon

# Set up color enum and rgb tuples
_R, _G, _B, _W, _Y, _O = range(1,7)
_colors = {
    _R: (1.0, 0.0, 0.0), # red
    _G: (0.0, 1.0, 0.0), # green
    _B: (0.0, 0.0, 1.0), # blue
    _W: (1.0, 1.0, 1.0), # white
    _Y: (1.0, 1.0, 0.0), # yellow
    _O: (1.0, 0.6, 0.0), # orange
}

class CubeDomain:

    def __init__(self, N, valid_actions=None):
        # N is side-length of cube
        
        # Count cubies and facies
        num_facies = 6*N**2
        num_cubies = N**3 * 3

        # Build solved_cube
        # solved_cube[i,j,k,d] is color of facie at cubie position (i,j,k) normal to d^th rotation axis
        solved_cube = np.zeros((N,N,N,3), dtype=int)
        solved_cube[ 0, :, :,0] = _R
        solved_cube[-1, :, :,0] = _O
        solved_cube[ :, 0, :,1] = _W
        solved_cube[ :,-1, :,1] = _Y
        solved_cube[ :, :, 0,2] = _B
        solved_cube[ :, :,-1,2] = _G

        # state representation is flat array of external facie colors only
        # cube_index[i,j,k,d] is index in state array of facie at position (i,j,k) normal to d^th axis
        # for internal facies, cube_index[i,j,k,d] == -1
        # face_index is the partial inverse of cube_index.  For the i^th facie in the state representation:
        # cube_index.flat[face_index[i]] == i
        
        # Make external facies non-zero in cube_index
        cube_index = np.zeros((N,N,N,3), dtype=int)
        cube_index[ 0, :, :,0] = 1
        cube_index[-1, :, :,0] = 1
        cube_index[ :, 0, :,1] = 1
        cube_index[ :,-1, :,1] = 1
        cube_index[ :, :, 0,2] = 1
        cube_index[ :, :,-1,2] = 1
        
        # Finalize index data from non-zero facies
        face_index = np.flatnonzero(cube_index)
        cube_index[:] = -1 # for all internal elements
        cube_index.flat[face_index] = np.arange(num_facies) # overwrite external facie indices

        # flatten the solved cube to state representation using face_index
        solved_state = solved_cube.flat[face_index]
        
        # twists are performed by permuting indices in the state array
        # new_state = old_state[twist_permutation[a, p, n]]
        # twist_permutation[a, p, n] is the permuted index array for n quarter twists of plane p around axis a
        twist_permutation = np.empty((3, N, 4, num_facies), dtype=int)

        # compute quarter twists for each axis
        for p in range(N):
            # rotx
            permuted = cube_index.copy()
            permuted[p,:,:,:] = np.rot90(permuted[p,:,:,:], axes=(0,1)) # rotate cubie positions
            permuted[p,:,:,(1,2)] = permuted[p,:,:,(2,1)] # rotate cubies
            twist_permutation[0, p, 1] = permuted.flat[face_index]
            # roty
            permuted = cube_index.copy()
            permuted[:,p,:,:] = np.rot90(permuted[:,p,:,:], axes=(1,0))
            permuted[:,p,:,(2,0)] = permuted[:,p,:,(0,2)]
            twist_permutation[1, p, 1] = permuted.flat[face_index]
            # rotz
            permuted = cube_index.copy()
            permuted[:,:,p,:] = np.rot90(permuted[:,:,p,:], axes=(0,1))
            permuted[:,:,p,(0,1)] = permuted[:,:,p,(1,0)]
            twist_permutation[2, p, 1] = permuted.flat[face_index]
        # compute non-quarter twists
        for a, p, n in it.product((0,1,2), range(N), (2, 3, 4)):
            twist_permutation[a, p, n % 4] = twist_permutation[a, p, n-1][twist_permutation[a, p, 1]]

        # precompute valid action list if not provided
        # action format: (rotation_axis, plane_index, num_twists)
        if valid_actions is None:
            valid_actions = tuple(it.product((0,1,2), range(1,N), (1,2,3))) # leave [0,0,0,:] corner cubie invariant
        else:
            valid_actions = tuple(valid_actions)

        # orientations of the full cube are also computed via state permutations
        orientation_permutation = np.empty((24, num_facies), dtype=int)

        # helper function to rotate all planes around a given axis
        def rotate_all_planes(state, axis, num_twists):
            state = state.copy()
            for plane in range(N):
                state = state[twist_permutation[axis, plane, num_twists % 4]]
            return state

        # reorientation also permutes valid actions
        action_permutation = []

        # helper function to rotate all actions around a given axis
        def rotate_all_actions(actions, axis, num_twists):
            actions = list(actions)
            mod2 = num_twists % 2
            mod4 = num_twists % 4
            for k, (a, p, t) in enumerate(actions):
                if (axis, a) == (0, 1): actions[k] = ([1, 2][mod2], [p, p, N-1-p, N-1-p][mod4], [+t, +t, -t, -t][mod4] % 4)
                if (axis, a) == (0, 2): actions[k] = ([2, 1][mod2], [p, N-1-p, N-1-p, p][mod4], [+t, -t, -t, +t][mod4] % 4)
                if (axis, a) == (1, 2): actions[k] = ([2, 0][mod2], [p, p, N-1-p, N-1-p][mod4], [+t, +t, -t, -t][mod4] % 4)
                if (axis, a) == (1, 0): actions[k] = ([0, 2][mod2], [p, N-1-p, N-1-p, p][mod4], [+t, -t, -t, +t][mod4] % 4)
                if (axis, a) == (2, 0): actions[k] = ([0, 1][mod2], [p, p, N-1-p, N-1-p][mod4], [+t, +t, -t, -t][mod4] % 4)
                if (axis, a) == (2, 1): actions[k] = ([1, 0][mod2], [p, N-1-p, N-1-p, p][mod4], [+t, -t, -t, +t][mod4] % 4)
            return actions

        # compute reorientation permutations
        for s, (axis, direction, num_twists) in enumerate(it.product((2,0,1), (1,-1), (0,1,2,3))):
            # align top face with one of six directed axes
            permuted_facies = np.arange(num_facies)
            permuted_actions = list(valid_actions)
            if axis != 2:
                permuted_facies = rotate_all_planes(permuted_facies, 1-axis, direction)
                permuted_actions = rotate_all_actions(permuted_actions, 1-axis, direction)
            elif direction != 1:
                permuted_facies = rotate_all_planes(permuted_facies, 0, 2)
                permuted_actions = rotate_all_actions(permuted_actions, 0, 2)
            # rotate cube around directed axis
            permuted_facies = rotate_all_planes(permuted_facies, axis, num_twists)
            orientation_permutation[s] = permuted_facies
            permuted_actions = rotate_all_actions(permuted_actions, axis, num_twists)
            action_permutation.append({valid_actions[k]: permuted_actions[k] for k in range(len(valid_actions))})

        # physically possible permutations of the colors correspond to possible orientations of the full cube
        color_permutation = np.zeros((24, 7), dtype=int) # 7 since color enum starts at 1
        
        # get one index in solved state for each color
        color_index = np.array([(solved_state == c).argmax() for c in range(1,7)])
        
        # extract color permutation from each orientation
        for sym in range(24):
            color_permutation[sym,1:] = solved_state[orientation_permutation[sym]][color_index]

        # determine symmetry inverses
        inverse_symmetry = {}
        for s, s_inv in it.product(range(24), repeat=2):
            if (color_permutation[s_inv][color_permutation[s]] == np.arange(7)).all():
                inverse_symmetry[s] = s_inv

        # look-up table for inverse recoloring of fixed corner cubie at cube[0,0,0,:]
        # fixed facies are first three elements of state arrays
        fixed_corner_restorer = np.zeros((7,7,7), dtype=int)
        for sym in range(24):
            key = tuple(solved_state[orientation_permutation[sym]][:3])
            fixed_corner_restorer[key] = inverse_symmetry[sym]

        # precompute symmetries of solved state
        solved_states = solved_state[orientation_permutation].copy()

        # memoize results
        self.N = N
        self._face_index = face_index
        self._solved_state = solved_state
        self._solved_states = solved_states
        self._twist_permutation = twist_permutation
        self._orientation_permutation = orientation_permutation
        self._action_permutation = action_permutation
        self._inverse_symmetry = inverse_symmetry
        self._color_permutation = color_permutation
        self._fixed_corner_restorer = fixed_corner_restorer
        self._valid_actions = valid_actions
    
    def god_number(self):
        if self.N == 2:
            if len(self._valid_actions) == 3: return 4
            if len(self._valid_actions) == 6: return 13
            return 11
        return 20

    def state_size(self):
        return self._solved_state.size

    def solved_state(self):
        return self._solved_state.copy()

    def valid_actions(self, state=None):
        # action format: (rotation_axis, plane_index, num_twists)
        return self._valid_actions

    def perform(self, action, state):
        axis, plane, num_twists = action
        return state[self._twist_permutation[axis, plane, num_twists % 4]].copy()

    def execute(self, actions, state):
        for action in actions: state = self.perform(action, state)
        return state

    def intermediate_states(self, actions, state):
        states = []
        for action in actions:
            state = self.perform(action, state)
            states.append(state)
        return states

    def is_solved_in_orientation_of(self, state):
        # matches = (self._solved_states == state).all(axis=1)
        # ori = matches.argmax()
        # return matches[ori], ori
        return (self._solved_states == state).all(axis=1).any()

    def is_solved_in(self, state):
        return (self._solved_state == state).all()

    def orientations_of(self, state):
        return state[self._orientation_permutation].copy()

    def reoriented_actions(self, sym):
        return dict(self._action_permutation[sym])

    def reorient_path(self, path, sym):
        return tuple(self._action_permutation[sym][action] for action in path)

    def inverse_symmetry_of(self, s):
        return self._inverse_symmetry[s]

    def recolorings_of(self, state):
        return self._color_permutation.take(state, axis=1)

    def color_neutral_to(self, state):
        orientations = self.orientations_of(state)
        fixed_corners = orientations[:,:3]
        restorers = self._fixed_corner_restorer[tuple(fixed_corners.T)]
        neutrals = np.take_along_axis(self._color_permutation[restorers], orientations, axis=1)
        return neutrals

    def reverse(self, actions):
        return [(axis, plane, -twists % 4) for (axis, plane, twists) in reversed(actions)]

    def superflip_path(self):
        # from https://www.cube20.org
        path = "R L U2 F U' D F2 R2 B2 L U2 F' B' U R2 D F2 U R2 U"
        action_map = {
            "U": (1, 0, 1),
            "D": (1, self.N-1, 3),
            "L": (2, self.N-1, 3),
            "R": (2, 0, 1),
            "F": (0, 0, 1),
            "B": (0, self.N-1, 3),
            "U2": (1, 0, 2),
            "D2": (1, self.N-1, 2),
            "L2": (2, self.N-1, 2),
            "R2": (2, 0, 2),
            "F2": (0, 0, 2),
            "B2": (0, self.N-1, 2),
            "U'": (1, 0, 3),
            "D'": (1, self.N-1, 1),
            "L'": (2, self.N-1, 1),
            "R'": (2, 0, 3),
            "F'": (0, 0, 3),
            "B'": (0, self.N-1, 1),
        }
        return [action_map[a] for a in path.split(" ")]

    def random_state(self, scramble_length, rng):
        state = self.solved_state()
        valid_actions = self.valid_actions()
        for s in range(scramble_length):
            state = self.perform(rng.choice(valid_actions), state)
        return state

    def render(self, state, ax, x0=0, y0=0):
        # ax is matplotlib Axes object
        # unflatten state into cube for easier indexing
        N = self.N
        cube = np.empty((N,N,N,3), dtype=int)
        cube.flat[self._face_index] = state
        # render orthogonal projection
        angles = -np.arange(3) * np.pi * 2 / 3
        axes = np.array([np.cos(angles), np.sin(angles)])
        for d in range(3):
            for a, b in it.product(range(N), repeat=2):
                xy = [ a   *axes[:,d] +  b   *axes[:,(d+1) % 3],
                      (a+1)*axes[:,d] +  b   *axes[:,(d+1) % 3],
                      (a+1)*axes[:,d] + (b+1)*axes[:,(d+1) % 3],
                       a   *axes[:,d] + (b+1)*axes[:,(d+1) % 3]]
                xy = [(x+x0, y+y0) for (x,y) in xy]
                c = _colors.get(cube[tuple(np.roll((a,b,0),d))+((d+2) % 3,)], (.7,)*3)
                ax.add_patch(Polygon(xy, facecolor=c, edgecolor='k'))
            ax.text((N+.1)*axes[0,d], (N+.1)*axes[1,d], str(d))

    def render_subplot(self, numrows, numcols, sp, state):
        ax = pt.subplot(numrows, numcols, sp)
        self.render(state, ax, 0, 0)
        ax.axis("equal")
        ax.axis('off')
        return ax

if __name__ == "__main__":

    # #### test performing actions
    # domain = CubeDomain(4)
    # actions = [(1, 0, 1), (0, 1, 1), (2, 2, 1), (1, 0, 1)]
    # # actions = [(0,0,1)]
    # state = domain.solved_state()

    # domain.render_subplot(1, len(actions)+1, 1, state)
    # for a, (axis, depth, num) in enumerate(actions):
    #     state = domain.perform((axis, depth, num), state)
    #     domain.render_subplot(1, len(actions)+1, a+2, state)
    
    # pt.show()

    # #### test valid_action input (only twist plane 1 in each axis)
    # domain = CubeDomain(2, it.product((0,1,2), (1,), (0, 1, 2, 3)))
    # for a, (axis, plane, num) in enumerate(domain.valid_actions()):
    #     state = domain.perform((axis, plane, num), domain.solved_state())
    #     domain.render_subplot(3, 4, a+1, state)
    
    # pt.show()

    # #### test right-handedness of each action (visual inspect)
    # pt.figure(figsize=(20,10))
    # domain = CubeDomain(3)
    # solved = domain.solved_state()
    # domain.render_subplot(7, 7, 1, solved)
    # for a, (axis, plane, num) in enumerate(domain.valid_actions()):
    #     state = domain.perform((axis, plane, num), solved)
    #     ax = domain.render_subplot(7, 7, a+2, state)
    #     ax.set_title(str((axis, plane, num)))
    # pt.tight_layout()
    # pt.show()

    # #### test symmetries
    # domain = CubeDomain(3)
    # state = domain.solved_state()
    # # state = domain.perform((0, 0, 1), state)
    # for s, sym_state in enumerate(domain.orientations_of(state)):
    #     ax = domain.render_subplot(4, 6, s+1, sym_state)
    #     ax.set_title("%s: %s" % (s, domain._color_permutation[s]))
    # pt.show()

    # #### test color permutations
    # domain = CubeDomain(2)
    # print(domain._color_permutation)
    # state = domain.solved_state()
    # # state = domain.perform((0, 0, 1), state)
    # for s, sym_state in enumerate(domain.recolorings_of(state)):
    # # for s in range(24):
    # #     sym_state = domain._color_permutation[s][state]
    #     ax = domain.render_subplot(4, 6, s+1, sym_state)
    #     ax.set_title(str(s))
    # pt.show()

    # #### test hardest state
    # domain = CubeDomain(3)
    # path = domain.superflip_path() # from unsolved to solved
    # # inverted = [a[:2]+(-a[2] % 4,) for a in path[::-1]] # from solved to unsolved
    # hardest_state = domain.execute(domain.reverse(path), domain.solved_state())
    # states = [hardest_state]
    # for action in path: states.append(domain.perform(action, states[-1]))
    # assert domain.is_solved_in(states[-1])
    # for s, state in enumerate(states): domain.render_subplot(4, 6, s+1, state)
    # pt.show()

    # # #### test to confirm that macros are invariant under recoloring conjugation
    # # i.e. macro(state) = (colsym * macro * inv colsym)(state)
    # # NOT true for full cube reorientations
    # # this is relevant to color neutrality of pdb patterns
    # domain = CubeDomain(3)
    # valid_actions = tuple(domain.valid_actions())

    # import numpy as np
    # rng = np.random.default_rng()
    # macro = rng.choice(valid_actions, size=5)
    # print(macro)

    # solved = domain.solved_state()
    # # solved = domain.random_state(10, rng)

    # init = solved
    # states = domain.intermediate_states(macro, solved)
    # states = [solved] + states + states[-1:]
    # for s, state in enumerate(states):
    #     ax = pt.subplot(4, len(states), s+1)
    #     domain.render(state, ax, 0, 0)
    #     ax.axis("equal")
    #     ax.axis('off')

    # sym = 11
    # init = domain.orientations_of(solved)[sym]
    # states = domain.intermediate_states(macro, init)
    # states = [init] + states + [domain.orientations_of(states[-1])[domain.inverse_symmetry_of(sym)]]
    # for s, state in enumerate(states):
    #     ax = domain.render_subplot(4, len(states), len(states) + s+1, state)

    # init = domain.recolorings_of(solved)[sym]
    # states = domain.intermediate_states(macro, init)
    # states = [init] + states + [domain.recolorings_of(states[-1])[domain.inverse_symmetry_of(sym)]]
    # for s, state in enumerate(states):
    #     ax = domain.render_subplot(4, len(states), 2*len(states) + s+1, state)

    # init = domain.orientations_of(solved)[sym]
    # states = domain.intermediate_states(macro, init)
    # states = [init] + states + [domain.recolorings_of(states[-1])[domain.inverse_symmetry_of(sym)]]
    # for s, state in enumerate(states):
    #     ax = domain.render_subplot(4, len(states), 3*len(states) + s+1, state)

    # pt.show()

    # # #### test to check whether (orisym, colsym, macro) sequence solves regardless of orisym and colsym
    # # it does solve (up to orientation) for no intermediate orisyms, but intermediate colsyms allowed
    # domain = CubeDomain(3)
    # valid_actions = tuple(domain.valid_actions())

    # import numpy as np

    # solved = domain.solved_state()

    # rng = np.random.default_rng()
    # scramble = rng.choice(valid_actions, size=9)
    # scrambled = domain.execute(scramble, solved)

    # scramble = domain.reverse(scramble)
    # macros = scramble[:3], scramble[3:6], scramble[6:]

    # pt.figure(figsize=(20, 5))

    # states = [scrambled]
    # for macro in macros:
    #     for action in macro:
    #         states.append(domain.perform(action, states[-1]))
    # for s, state in enumerate(states):
    #     ax = domain.render_subplot(2, len(states), s+1, state)

    # states = [scrambled]
    # for macro in macros:
    #     orisym, colsym = rng.choice(24, size=2)
    #     # states[-1] = domain.orientations_of(states[-1])[orisym]
    #     states[-1] = domain.recolorings_of(states[-1])[colsym]
    #     for action in macro:
    #         states.append(domain.perform(action, states[-1]))
    # for s, state in enumerate(states):
    #     ax = domain.render_subplot(2, len(states), len(states) + s+1, state)

    # pt.show()

    # #### test to confirm that (orisym, colsym) commute
    # domain = CubeDomain(3)
    # valid_actions = tuple(domain.valid_actions())

    # import numpy as np
    # solved = domain.solved_state()

    # rng = np.random.default_rng()
    # scramble = rng.choice(valid_actions, size=9)
    # scrambled = domain.execute(scramble, solved)

    # pt.figure(figsize=(20, 5))

    # orisym, colsym = rng.choice(24, size=2)
    # states = [scrambled]
    # states.append(domain.orientations_of(states[-1])[orisym])
    # states.append(domain.recolorings_of(states[-1])[colsym])
    # for s, state in enumerate(states):
    #     ax = domain.render_subplot(2, len(states), s+1, state)
    # final = states[-1]

    # states = [scrambled]
    # states.append(domain.recolorings_of(states[-1])[colsym])
    # states.append(domain.orientations_of(states[-1])[orisym])
    # for s, state in enumerate(states):
    #     ax = domain.render_subplot(2, len(states), len(states) + s+1, state)

    # assert (final == states[-1]).all()

    # pt.show()

    #### test reoriented actions
    domain = CubeDomain(3)
    valid_actions = tuple(domain.valid_actions())

    import numpy as np
    solved = domain.solved_state()

    rng = np.random.default_rng()
    scramble = [tuple(a) for a in rng.choice(valid_actions, size=1)]
    # scramble = [(2, 2, 2)]

    for orisym in range(24):
    # for orisym in [8]:    

        states = [solved] + domain.intermediate_states(scramble, solved)    

        # action_map = domain.reoriented_actions(orisym)    
        # ori_solved = domain.orientations_of(solved)[orisym]
        # ori_scramble = [action_map[a] for a in scramble]
        ori_solved = domain.orientations_of(solved)[orisym]
        ori_scramble = domain.reorient_path(scramble, orisym)
        ori_states = [ori_solved] + domain.intermediate_states(ori_scramble, ori_solved)

        if not (states[-1] == domain.orientations_of(ori_states[-1])[domain.inverse_symmetry_of(orisym)]).all():

            pt.figure(figsize=(20, 5))

            for s, state in enumerate(states):
                ax = domain.render_subplot(2, len(states), s+1, state)
                if s > 0: ax.set_title(str(scramble[s-1]))

            for s, state in enumerate(ori_states):
                ax = domain.render_subplot(2, len(states), len(states) + s+1, state)
                if s > 0: ax.set_title(str(ori_scramble[s-1]))
                else: ax.set_title(str(orisym))

            pt.show()

        assert (states[-1] == domain.orientations_of(ori_states[-1])[domain.inverse_symmetry_of(orisym)]).all()

    # #### test reorient+recolor neutralization approach on 2cube
    # # confirms that for every reorientation of a random state, there is exactly one recoloring that restores the invariant rbw cubie
    # # furthermore, orientation->recoloring is bijective (every possible recoloring occurs once).
    # # however, the bijection is different for different random states.
    # # fortunately, a state-independent lookup (restorer variable below) maps colors of invariant cubie position to the correct recoloring.

    # # domain = CubeDomain(2)
    # # valid_actions = tuple(domain.valid_actions())
    # valid_actions = tuple(it.product((0,1,2), (1,), (1, 2, 3))) # only spinning one plane on each axis for 2cube
    # domain = CubeDomain(2, valid_actions)

    # rng = np.random.default_rng()
    # solved = domain.solved_state()
    # invariant_facies = [0,1,2] # facies of fixed corner at coordinate [0,0,0,:] in 2cube

    # invariant_state = np.zeros(domain.state_size())
    # invariant_state[invariant_facies] = solved[invariant_facies]
    # domain.render_subplot(1,1,1,invariant_state)
    # pt.show()

    # restorer = {}
    # for ori in range(24):
    #     ori_solved = domain.orientations_of(solved)[ori]
    #     key = tuple(ori_solved[invariant_facies])
    #     restorer[key] = domain.inverse_symmetry_of(ori)
    #     col_solved = domain.recolorings_of(ori_solved)[restorer[key]]
    #     assert (col_solved[invariant_facies] == solved[invariant_facies]).all()

    # for rep in range(5):
    #     state = domain.random_state(scramble_length=20, rng=rng)
    #     invariant = {}
    #     for ori in range(24):
    #         ori_state = domain.orientations_of(state)[ori]
    #         ori_solved = domain.orientations_of(solved)[ori]
    #         invariant[ori] = []
    #         for col in range(24):
    #             col_state = domain.recolorings_of(ori_state)[col]
    #             col_solved = domain.recolorings_of(ori_solved)[col]
    #             if (col_state == state)[invariant_facies].all(): invariant[ori].append(col)
    #         assert len(invariant[ori]) == 1
    #         invariant[ori] = invariant[ori][0]
    #         assert invariant[ori] == restorer[tuple(ori_state[invariant_facies])]

    #     cols = list(invariant.values())
    #     print(cols)
    #     assert len(set(cols)) == 24

    # for ori in range(24):
    #     ori_state = domain.orientations_of(state)[ori]
    #     ori_solved = domain.orientations_of(solved)[ori]
    #     # col = invariant[ori]
    #     col = restorer[tuple(ori_state[invariant_facies])]
    #     col_state = domain.recolorings_of(ori_state)[col]
    #     col_solved = domain.recolorings_of(ori_solved)[col]
    #     domain.render_subplot(2,3,1,state)
    #     domain.render_subplot(2,3,2,ori_state)
    #     domain.render_subplot(2,3,3,col_state)
    #     domain.render_subplot(2,3,4,solved)
    #     domain.render_subplot(2,3,5,ori_solved)
    #     domain.render_subplot(2,3,6,col_solved)
    #     pt.show()

    # #### test color_neutral_to
    # domain = CubeDomain(2)
    # rng = np.random.default_rng()
    # solved = domain.solved_state()
    # ori_solved = domain.orientations_of(solved)
    # neu_solved = domain.color_neutral_to(solved)

    # for rep in range(5):
    #     state = domain.random_state(scramble_length=20, rng=rng)
    #     ori_state = domain.orientations_of(state)
    #     neu_state = domain.color_neutral_to(state)
    #     for sym in range(24):
    #         assert (neu_state[sym][:3] == state[:3]).all()
    #         if rep == 4:
    #             domain.render_subplot(3, 24, sym + 1 + 0*24, ori_solved[sym])
    #             domain.render_subplot(3, 24, sym + 1 + 1*24, ori_state[sym])
    #             domain.render_subplot(3, 24, sym + 1 + 2*24, neu_state[sym])
    # pt.show()
