import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
from DesignTool import *

class RediCell:
    def __init__(self, sides=None, spacing=None, t_step=None, molecule_types=None, reaction_set=None, wall=True, design=None):
        # sides in number of voxels
        # spacing in m
        # "wall" adds one extra cell each direction, set as barrier
        if design is not None:
            self.spacing = design.spacing
            self.wall = design.wall
            self.sides = design.sides.copy()
            self.ndim = design.ndim
        else:
            assert len(sides) > 0 and len(sides) < 4
            self.spacing = spacing
            self.wall = wall
            self.sides = np.array(sides).astype(int) # Should be [32, 32] or [32, 32, 32]
            self.ndim = len(self.sides)
        self.one_per_voxel_equal_um = 1.0 / self.spacing**3 / 6.023e23 / 1000 * 1e6
        # Should be a list of Molecule instances
        if isinstance(molecule_types, MoleculeSet):
            self.molecule_types = molecule_types.molecule_types
            self.molecule_names = molecule_types.molecule_names
            self.molecule_observed_barrier_types = molecule_types.molecule_observed_barrier_types
            self.molecule_special_diffusion_coefficients = molecule_types.molecule_special_diffusion_coefficients
        elif isinstance(molecule_types, list):
            raise
        else:
            raise

        self.num_types = len(self.molecule_types)
        self.mol_to_id = {mol.molecule_name: idx for idx, mol in enumerate(self.molecule_types)}
        self.id_to_mol = {idx: mol.molecule_name for idx, mol in enumerate(self.molecule_types)}
        
        self.initialized = False
        self.voxel_matrix = []
        
        
        if self.wall:
            self.true_sides = self.sides + 2
        else:
            self.true_sides = self.sides

        self.t_step = t_step
        self.t_trace = []
        self.conc_trace = []
        self.diffusion_vector = None
        self.reagent_vector_list = []
        self.reaction_vector_list = []
        self.reaction_coefficients = []
        self.fig = None
        self.cumulative_t = 0
        self.reaction_set = None
        self.num_reaction = 0
    
        self.side_coord = [np.linspace(0, side * self.spacing, side) for side in self.true_sides]
        self.mesh = np.meshgrid(*self.side_coord, indexing='ij')

        if design is not None:
            # Barrier types: -1 = no barrier, 0 = surrounding wall, 1 -> N = custom
            self.barrier_type = design.barrier_type.copy()
            # Barrier types: 0 = default (diffusion the same everywhere), 1 -> N = custom defined locations
            self.special_space_type = design.special_space_type.copy()
        else:
            # Barrier types: -1 = no barrier, 0 = surrounding wall, 1 -> N = custom
            self.barrier_type = np.zeros(self.true_sides).astype(int) - 1
            # Barrier types: 0 = default (diffusion the same everywhere), 1 -> N = custom defined locations
            self.special_space_type = np.zeros(self.true_sides).astype(int)

        self.reaction_set = reaction_set
        if self.reaction_set is not None:
            self.num_reaction = len(self.reaction_set.reaction)
        else:
            self.num_reaction = 0

        self.external_conditions = []
    
    def partition(self):
        # m, x, y matrix
        self.voxel_matrix = np.zeros((self.num_types, *self.true_sides))
        self.voxel_matrix_shape = self.voxel_matrix.shape
        
        self.construct_possible_actions()
        
        
        if self.reaction_set is not None:
        
            self.reaction_matrix_list = [np.tile(np.expand_dims(x, tuple(range(1, self.ndim+1))), 
                                                       (1, *self.voxel_matrix.shape[1:])).astype(np.float32)
                                         for x in self.reaction_vector_list]

            self.reaction_voxel_shape = (self.num_reaction+1, *self.true_sides)
    
    def set_border_wall(self): 
        if self.ndim >= 1:
            self.barrier_type[0] = 0
            self.barrier_type[-1] = 0
        if self.ndim >= 2:
            self.barrier_type[:, 0] = 0
            self.barrier_type[:, -1] = 0
        if self.ndim >= 3:
            self.barrier_type[:, :, 0] = 0
            self.barrier_type[:, :, -1] = 0
        

    def add_barrier(self, voxel_list=None, barrier_type_index=None):
        if voxel_list is None or barrier_type_index is None:
            print('Either voxel_list or barrier_type_index is not set. (Nothing done)')
            return
        if barrier_type_index == 0:
            print('Barrier type index 0 is reserved for full-system border wall. Use an integer >= 1. (Nothing done)')
            return
        if isinstance(voxel_list, tuple):
            self.barrier_type[voxel_list] = barrier_type_index
        if isinstance(voxel_list, list):
            for v in voxel_list:
                self.barrier_type[v] = barrier_type_index

    def configure_barrier(self):

        # Build border wall
        if self.wall:
            self.set_border_wall()

        self.not_barrier_matrix = np.ones(self.voxel_matrix_shape).astype(bool)

        for idx, mobt in enumerate(self.molecule_observed_barrier_types):
            # mobt is molecule observed barrier types
            for obt in mobt:
                self.not_barrier_matrix[idx, self.barrier_type == obt] = 0

        if self.ndim >= 1:
            self.not_barrier_matrix_up = self.not_barrier_matrix[:, :-1]
            self.not_barrier_matrix_down = self.not_barrier_matrix[:, 1:]
        if self.ndim >= 2:
            self.not_barrier_matrix_left = self.not_barrier_matrix[:, :, :-1]
            self.not_barrier_matrix_right = self.not_barrier_matrix[:, :, 1:]
        if self.ndim >= 3:
            self.not_barrier_matrix_front = self.not_barrier_matrix[:, :, :, :-1]
            self.not_barrier_matrix_back = self.not_barrier_matrix[:, :, :, 1:]

    def plot_wall(self):
        assert self.ndim == 2
        my_cmap = cm.get_cmap('tab10')
        my_cmap.set_under('w')
        plt.imshow(self.barrier_type, cmap=my_cmap, vmin=0)
        
    def initialize(self):
        assert self.initialized == False
        # Set condition based on initial conditions
        # Set external conditions
        # Compile all possible reactions (diffuse to 4 neighbors, reactions, and do nothing)
        # Calculate largest possible timestep
        self.initialized = True

    def replace_reaction_set(self, reaction_set=None):
        self.reaction_set = reaction_set
        self.num_reaction = len(self.reaction_set.reaction)
    
    def list_reaction_set(self):
        pass
    
    def construct_possible_actions(self):
        # up, down, left, right for each kind of molecule
        # then reactions
        self.diffusion_vector = []
        self.reagent_vector_list = []
        self.reaction_vector_list = []
        self.reaction_coefficients = []
        for mol in self.molecule_types:
            if mol.special_diffusion_coefficients is None:
                self.diffusion_vector.append(self.ndim * 2 * mol.diffusion_coefficient / self.spacing**2) 
            else: # there is special diffusion coefficients, process it
                # Base diffusion
                diffusion_matrix = np.ones(self.true_sides) * self.ndim * 2 * mol.diffusion_coefficient / self.spacing**2
                # Special diffusions
                for key in mol.special_diffusion_coefficients.keys():
                    print("Do special diffusion")
                    diffusion_matrix[self.special_space_type == key] = self.ndim * 2 * mol.special_diffusion_coefficients[key] / self.spacing**2
                self.diffusion_vector.append(diffusion_matrix)
        
        if self.reaction_set is not None:
            for reaction in self.reaction_set.reaction:
                reagent_vector = []
                reaction_vector = np.zeros(self.num_types)
                for reagent in reaction[0]: # Should be names of the reagents
                    reagent_vector.append(self.mol_to_id[reagent])
                    reaction_vector[self.mol_to_id[reagent]] = -1
                for product in reaction[1]:
                    reaction_vector[self.mol_to_id[product]] = 1
                self.reagent_vector_list.append(reagent_vector)
                self.reaction_vector_list.append(reaction_vector.astype(int))
                self.reaction_coefficients.append(reaction[2])
            print('Reagent list:', self.reagent_vector_list)
        
        print('Action list:')
        for mol in self.molecule_types:
            print(f'Diffusion of {mol.molecule_name} ({self.ndim*2} directions)')
        if self.reaction_set is not None:
            for reaction in self.reaction_set.reaction:
                print(f'Reaction: reagent {reaction[0]} -> product {reaction[1]}')
        else:
            print('No reactions')
    
    def determine_maximum_timestep(self):
        print(f'Max time step is {1 / np.max(np.array(self.diffusion_vector)/2/self.ndim) / 4 / self.voxel_matrix.max() :.2e} s (max {self.voxel_matrix.max()} particles in voxel)')
        return 1 / np.max(np.array(self.diffusion_vector)/2/self.ndim) / 4 / self.voxel_matrix.max()
        
    def add_molecules(self, molecule_type, molecule_count):
        self.molecule_count[molecule_type] = molecule_count
        # distribute molecules randomly
        pass

    def add_external_conditions(self, region, molecule, concentration):
        # region is a space type index or a region specified by ones in a matrix same shape as self.true_sides
        # molecule is a Molecule object
        # concentration is a float in micromolar
        assert molecule in self.molecule_types
        
        if isinstance(region, int):
            assert region in self.special_space_type
            real_region = self.special_space_type == region
        elif np.all(region.shape == self.true_sides):
            real_region = region > 0
        else:
            raise("region has to be either an existing space type, or a matrix with same shape as RediCell.true_sides")
        num_molecule = int(np.round(concentration / self.one_per_voxel_equal_um * real_region.sum()))
        region_voxel = np.where(real_region)
        self.external_conditions.append([real_region, self.mol_to_id[molecule.molecule_name], concentration, num_molecule])
    
    def show_external_conditions(self):
        for row in self.external_conditions:
            print(f'Maintain {row[2]} micromolar of {self.id_to_mol[row[1]]} within a space of {row[0].sum()} voxels ({row[3]} molecules)')
    
    def maintain_external_conditions(self):
        for row in self.external_conditions:
            if row[2] == 0:
                # Just remove everything in it
                self.voxel_matrix[row[1], row[0]] = 0
                # print('emptied region')
            else:
                current = int(self.voxel_matrix[row[1], row[0]].sum())
                change = row[3] - current
                if change > 0:
                    candidates = np.where(row[0])
                    choices = np.random.choice(len(candidates[0]), change, replace=False)
                    selections = [x[choices] for x in candidates]
                    if self.ndim == 2:
                        self.voxel_matrix[row[1], selections[0], selections[1]] += 1
                    if self.ndim == 3:
                        self.voxel_matrix[row[1], selections[0], selections[1], selections[2]] += 1
                    # print(f'Added {change} molecules')
                elif change < 0:
                    candidates = np.where((self.voxel_matrix[row[1]] * row[0]) == 1)
                    choices = np.random.choice(len(candidates[0]), -change, replace=False)
                    selections = [x[choices] for x in candidates]
                    if self.ndim == 2:
                        self.voxel_matrix[row[1], selections[0], selections[1]] -= 1
                    if self.ndim == 3:
                        self.voxel_matrix[row[1], selections[0], selections[1], selections[2]] -= 1
                    # print(f'Deleted {change} molecules')

    # @profile
    def react_diffuse(self, t_step):
        diffuse_voxel = np.zeros(self.voxel_matrix_shape, dtype=np.float32)
        random_choice = np.random.random(self.voxel_matrix_shape) * 2 * self.ndim + 1
        random_sampling = np.random.random(self.voxel_matrix_shape)
        for idx in range(self.num_types):
            # Diffuse part
            diffuse_voxel[idx] = self.voxel_matrix[idx] * self.diffusion_vector[idx] * t_step
            random_choice[idx] += idx * 2 * self.ndim

        diffusion_choice = random_choice.astype(int) * (random_sampling < diffuse_voxel)
            
        
        for choicep1 in range(1, 2 * self.ndim + 1):
            choice = choicep1 - 1
            if self.ndim >= 1:
            
                if choice % (2*self.ndim) == 0: 
                    move_action = (diffusion_choice[:, 1:] % (2*self.ndim) == choicep1 % (2 * self.ndim)) * self.not_barrier_matrix_up
                    self.voxel_matrix[:, 1:] -= move_action
                    self.voxel_matrix[:, :-1] += move_action
                    continue
                    
                if choice % (2*self.ndim) == 1:
                    if self.ndim == 1: # Does not really happen
                        move_action = (diffusion_choice[:, :-1] % (2*self.ndim) == choicep1 % (2 * self.ndim)) * self.not_barrier_matrix_down * (diffusion_choice[:, :-1] > 0)
                    else:
                        move_action = (diffusion_choice[:, :-1] % (2*self.ndim) == choicep1 % (2 * self.ndim)) * self.not_barrier_matrix_down
                    self.voxel_matrix[:, :-1] -= move_action
                    self.voxel_matrix[:, 1:] += move_action
                    continue
                    
            if self.ndim >= 2:
                if choice % (2*self.ndim) == 2:
                    move_action = (diffusion_choice[:, :, 1:] % (2*self.ndim) == choicep1 % (2 * self.ndim)) * self.not_barrier_matrix_left
                    self.voxel_matrix[:, :, 1:] -= move_action
                    self.voxel_matrix[:, :, :-1] += move_action
                    continue
                
                if choice % (2*self.ndim) == 3:
                    if self.ndim == 2:
                        move_action = (diffusion_choice[:, :, :-1] % (2*self.ndim) == choicep1 % (2 * self.ndim)) * self.not_barrier_matrix_right * (diffusion_choice[:, :, :-1] > 0)
                    else:
                        move_action = (diffusion_choice[:, :, :-1] % (2*self.ndim) == choicep1 % (2 * self.ndim)) * self.not_barrier_matrix_right
                    self.voxel_matrix[:, :, :-1] -= move_action
                    self.voxel_matrix[:, :, 1:] += move_action
                    continue
                    
            if self.ndim >= 3:
                if choice % (2*self.ndim) == 4:
                    move_action = (diffusion_choice[:, :, :, 1:] % (2*self.ndim) == choicep1 % (2 * self.ndim)) * self.not_barrier_matrix_front
                    self.voxel_matrix[:, :, :, 1:] -= move_action
                    self.voxel_matrix[:, :, :, :-1] += move_action
                    continue
                
                if choice % (2*self.ndim) == 5:
                    if self.ndim == 3:
                        move_action = (diffusion_choice[:, :, :, :-1] % (2*self.ndim) == choicep1 % (2 * self.ndim)) * self.not_barrier_matrix_back * (diffusion_choice[:, :, :, :-1] > 0)
                    else: # does not really happen
                        move_action = (diffusion_choice[:, :, :, :-1] % (2*self.ndim) == choicep1 % (2 * self.ndim)) * self.not_barrier_matrix_back
                    self.voxel_matrix[:, :, :, :-1] -= move_action
                    self.voxel_matrix[:, :, :, 1:] += move_action
                    continue
                                
    
        # React part
        if self.reaction_set is not None:
            
            for idx, (reagent, coeff) in enumerate(zip(self.reagent_vector_list, self.reaction_coefficients)):
                
                if len(reagent) == 2:
                    reaction_voxel = 1 - np.exp(-np.prod(self.voxel_matrix[reagent], axis=0) * coeff / 6.023e23 / self.spacing**3 / 1000 * t_step)
                if len(reagent) == 1:
                    reaction_voxel = 1 - np.exp(-np.prod(self.voxel_matrix[reagent], axis=0) * coeff * t_step)
                
                random_sampling = np.random.random(self.true_sides)
                self.voxel_matrix += self.reaction_matrix_list[idx] * (random_sampling < reaction_voxel)
                    
        self.cumulative_t += t_step
        self.t_trace.append(self.cumulative_t)
        self.conc_trace.append(self.voxel_matrix.sum(tuple(range(1, self.ndim+1))))
                
    def simulate(self, steps, t_step=None, plot_every=None, timing=False):
        if not self.initialized:
            self.initialize()
        if t_step is not None:
            self.t_step = t_step
        for step in range(steps):
            if step == 0: 
                t0 = time.time()
                print(f'Simulate {steps} steps')
            if steps > 100 and step % (steps // 100) == 0 and step > 0:
                print(step, end=' ')
            if timing and steps > 100 and step % (steps // 10) == 0 and step > 0:
                t1 = time.time()
                print(f'{(t1 - t0):.2f} s - {(t1-t0)*1000 / step:.2f} ms / step')
                
            self.maintain_external_conditions()
            self.react_diffuse(self.t_step)
            if plot_every is not None:
                if step % plot_every == 0:
                    self.plot(self.molecule_names)
        
    def plot(self, mol_type, wall=True):
        if self.ndim == 2:
            self.plot2D(mol_type, wall)
        if self.ndim == 3:
            self.plot3D(mol_type, wall)
            
    def plot2D(self, mol_type, wall=True):
        plt.figure(figsize=(6,6), dpi=100)
        if type(mol_type) == str:
            mol_type = [mol_type]
        if wall:
            # plot wall 
            barrier_present = self.barrier_type >= 0
            barrier_location = [self.mesh[x][barrier_present] for x in range(self.ndim)]
            plt.scatter(barrier_location[0], barrier_location[1], s = 25, c = 'gray', marker='s')
        for mol in mol_type:
            idx = self.mol_to_id[mol]
            particles_present = np.where(self.voxel_matrix[idx] > 0)
            particle_number = self.voxel_matrix[idx][particles_present].astype(int)
            particle_location = np.array([np.repeat(self.mesh[x][particles_present], particle_number) 
                                 for x in range(self.ndim)])
            # randomize location a bit
            particle_location += (np.random.random(size=particle_location.shape) - 0.5) * self.spacing * 0.5
            if idx < 2:
                plt.scatter(particle_location[0], particle_location[1], s=3)
            else:
                plt.scatter(particle_location[0], particle_location[1], s=9, marker='x')
        plt.xlim([-0.5 * self.spacing, (self.true_sides[0]+0.5)*self.spacing])
        plt.ylim([-0.5 * self.spacing, (self.true_sides[1]+0.5)*self.spacing])
        plt.gca().invert_yaxis()
        plt.title(f't = {self.cumulative_t:.3e} s')
        plt.grid(alpha=0.3)
        plt.show()

    def plot3D(self, mol_type, wall=True):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        if type(mol_type) == str:
            mol_type = [mol_type]
        if wall:
            # plot wall 
            barrier_present = self.barrier_type >= 0
            barrier_location = [self.mesh[x][barrier_present] for x in range(self.ndim)]
            ax.scatter(barrier_location[0], barrier_location[1], barrier_location[2], s = 25, c = 'gray', marker='s')
        for mol in mol_type:
            idx = self.mol_to_id[mol]
            particles_present = np.where(self.voxel_matrix[idx] > 0)
            particle_number = self.voxel_matrix[idx][particles_present].astype(int)
            particle_location = np.array([np.repeat(self.mesh[x][particles_present], particle_number) 
                                 for x in range(self.ndim)])
            # randomize location a bit
            particle_location += (np.random.random(size=particle_location.shape) - 0.5) * self.spacing * 0.5
            # if idx < 2:
            ax.scatter(particle_location[0], particle_location[1], particle_location[2], s=2)
            # else:
            #     ax.scatter(particle_location[0], particle_location[1], particle_location[2], s=9, marker='x')    
        xs, ys, zs = [self.mesh[x] for x in range(self.ndim)]
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.set_xlim([-0.5 * self.spacing, (self.true_sides[0]-0.5)*self.spacing])
        ax.set_ylim([-0.5 * self.spacing, (self.true_sides[1]-0.5)*self.spacing])
        ax.set_zlim([-0.5 * self.spacing, (self.true_sides[2]-0.5)*self.spacing])
        plt.title(f't = {self.cumulative_t:.3e} s')
        plt.grid(alpha=0.3)
        plt.show()


class MoleculeSet:
    def __init__(self, molecule=[]):
        self.molecule_types = list(molecule)
        self.molecule_names = [mol.molecule_name for mol in self.molecule_types]
        self.molecule_observed_barrier_types = [mol.observed_barrier_types for mol in self.molecule_types]
        self.molecule_special_diffusion_coefficients = [mol.special_diffusion_coefficients for mol in self.molecule_types]
        
    def add_molecule(self, molecule):
        assert molecule.molecule_name not in self.molecule_names
        self.molecule_types.append(molecule)
        
class Molecule:
    def __init__(self, molecule_name, diffusion_coefficient, observed_barrier_types=None, special_diffusion_coefficients=None):
        # diffusion_coefficient in m^2 / s, so a value like 1e-13 m^2/s is likely
        self.molecule_name = molecule_name

        # The default diffusion coefficient for everywhere
        self.diffusion_coefficient = diffusion_coefficient
        
        # Should be an integer or a list of positive integers. If None then observes only full-system barriers (type 0)
        if isinstance(observed_barrier_types, int):
            self.observed_barrier_types = [0, observed_barrier_types]
        elif isinstance(observed_barrier_types, list):
            self.observed_barrier_types = [0] + observed_barrier_types
        elif observed_barrier_types is None:
            self.observed_barrier_types = [0]
        else:
            raise

        # Extra diffusion coefficients
        # A dictionary with special space types
        if special_diffusion_coefficients is not None and not isinstance(special_diffusion_coefficients, dict):
            raise
        self.special_diffusion_coefficients = special_diffusion_coefficients
        
            
class ReactionSet:
    def __init__(self):
        self.reaction = []
    def add_reaction(self, reagent, product, reaction_coefficient):
        # reagent can be [typeA, typeB] for bimolecular reaction
        # or [typeA] or typeA for unimolecular reaction
        self.reaction.append([reagent, product, reaction_coefficient])


