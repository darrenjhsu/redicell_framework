import cupy as cp, numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
import nvtx
from tqdm import tqdm
import pickle
import h5py


class RediCell_CuPy:
    
    def __init__(self, sides=None, spacing=None, t_step=None, molecule_types=None, reaction_set=None, wall=True, design=None, project_name='redicell', out_freq=0.1):
        # sides in number of voxels
        # spacing in m
        # "wall" adds one extra cell each direction, set as barrier
        
        if design is not None:
            self.spacing = design.spacing
            self.wall = design.wall
            self.sides = np.array(design.sides).astype(int)
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
        self.diffusion_vector = None
        self.reagent_vector_list = []
        self.reaction_vector_list = []
        self.reaction_coefficients = []
        self.fig = None
        self.cumulative_t = 0.
        self.reaction_set = None
        self.num_reaction = 0

        self.side_coord = [cp.linspace(0, int(side) * self.spacing, int(side)) for side in self.true_sides]
        self.mesh = cp.meshgrid(*self.side_coord, indexing='ij')

        if design is not None:
            # Barrier types: -1 = no barrier, 0 = surrounding wall, 1 -> N = custom
            self.barrier_type = cp.array(design.barrier_type)
            # Barrier types: 0 = default (diffusion the same everywhere), 1 -> N = custom defined locations
            self.special_space_type = cp.array(design.special_space_type)
        else:
            # Barrier types: -1 = no barrier, 0 = surrounding wall, 1 -> N = custom
            self.barrier_type = cp.zeros(self.true_sides).astype(int) - 1
            # Barrier types: 0 = default (diffusion the same everywhere), 1 -> N = custom defined locations
            self.special_space_type = cp.zeros(self.true_sides).astype(int)

        self.reaction_set = reaction_set
        if self.reaction_set is not None:
            self.num_reaction = len(self.reaction_set.reaction)
        else:
            self.num_reaction = 0

        self.external_conditions = []
        
        self.timestamp = time.time()
        self.timestring = self.convert_timestamp(self.timestamp) # Used for suffixing file
        self.project_name = project_name + "-" + self.timestring
        self.checkpoint_filename = self.project_name + '.ckpt'
        self.traj_filename = self.project_name + '.hdf5'
        self.step = 0
        self.frame = 0
        self.start_from_scratch=True
        
    def convert_timestamp(self, timestamp):
        # Convert timestamp to datetime object
        dt_object = datetime.fromtimestamp(timestamp)

        # Format the datetime object as yyyy-mm-dd-hh-mm-ss
        formatted_time = dt_object.strftime('%Y-%m-%d-%H-%M-%S')

        return formatted_time

    def partition(self):
        # m, x, y matrix
        self.voxel_matrix = cp.zeros((self.num_types, *self.true_sides)).astype(cp.float32)
        self.voxel_matrix_shape = self.voxel_matrix.shape

        self.construct_possible_actions()
        
        if self.reaction_set is not None:
        
            self.reaction_matrix_list = [cp.tile(np.expand_dims(x, tuple(range(1, self.ndim+1))), 
                                                       (self.true_sides)).astype(cp.float32)
                                         for x in self.reaction_vector_list]

            self.reaction_voxel_shape = (self.num_reaction, *self.true_sides)
            
            for idx, reaction in enumerate(self.reaction_set.reaction):
                if reaction[3] is not None:
                    self.reaction_matrix_list[idx] *= cp.array(reaction[3], dtype=bool)[None, :]
            
        

    def set_border_wall(self, propagate=False): 
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

        self.not_barrier_matrix = cp.ones(self.voxel_matrix_shape).astype(bool)

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
        plt.imshow(self.barrier_type.get(), cmap=my_cmap, vmin=0)
        
    def initialize(self):
        assert self.initialized == False
        # Set condition based on initial conditions
        # Set external conditions
        # Compile all possible reactions (diffuse to 4 neighbors, reactions, and do nothing)
        # Calculate largest possible timestep
        with h5py.File(self.traj_filename, 'a') as f:
            mols = f.create_dataset('metadata/mols', data=np.array(self.molecule_names, dtype='S'))            
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
                diffusion_matrix = cp.ones(self.true_sides, dtype=cp.float32) * self.ndim * 2 * mol.diffusion_coefficient / self.spacing**2
                # Special diffusions
                for key in mol.special_diffusion_coefficients.keys():
                    print("Do special diffusion")
                    diffusion_matrix[self.special_space_type == key] = self.ndim * 2 * mol.special_diffusion_coefficients[key] / self.spacing**2
                self.diffusion_vector.append(diffusion_matrix)
        
        if self.reaction_set is not None:
            for reaction in self.reaction_set.reaction:
                reagent_vector = []
                reaction_vector = cp.zeros(self.num_types)
                for reagent in reaction[0]: # Should be names of the reagents
                    reagent_vector.append(self.mol_to_id[reagent])
                    reaction_vector[self.mol_to_id[reagent]] -= 1
                for product in reaction[1]:
                    reaction_vector[self.mol_to_id[product]] += 1
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
        
    def add_molecules(self, location, name, count=None, uM=None):
        if count is None and uM is None:
            print(f'Neither count nor uM specified - do nothing and return')
            return
        elif isinstance(count, (int, float)) and isinstance(uM, (int, float)):
            print(f'Both count and uM specified - I do not know what to do - do nothing and return')
            return
        
        avail = np.where(location)
        n_vox = len(avail[0])
        if count is not None:
            num_to_add = int(count)
            print(f'Add {num_to_add} {name} molecules')
        else: #molecule_concentration is not None
            num_to_add = int(np.round(n_vox / self.one_per_voxel_equal_um * uM))
            print(f'Add {num_to_add} {name} molecules ({num_to_add * self.one_per_voxel_equal_um / n_vox:.2f} uM)')
        choice = np.random.choice(n_vox, num_to_add, replace=False)
        sel = [x[choice] for x in avail]
        if self.ndim == 2:
            self.voxel_matrix[self.mol_to_id[name], sel[0], sel[1]] = 1
        elif self.ndim == 3:
            self.voxel_matrix[self.mol_to_id[name], sel[0], sel[1], sel[2]] = 1


    def add_external_conditions(self, region, molecule, concentration):
        # region is a space type index or a region specified by ones in a matrix same shape as self.true_sides
        # molecule is a Molecule object
        # concentration is a float in micromolar
        assert molecule in self.molecule_types
        
        if isinstance(region, int):
            assert region in self.special_space_type
            real_region = self.special_space_type == region
        elif np.all(region.shape == self.true_sides):
            real_region = cp.array(region > 0)
        else:
            raise("region has to be either an existing space type, or a matrix with same shape as RediCell.true_sides")
        num_molecule = int(np.round(concentration / self.one_per_voxel_equal_um * real_region.sum()))
        region_voxel = cp.where(real_region)
        self.external_conditions.append([real_region, self.mol_to_id[molecule.molecule_name], concentration, num_molecule, cp.where(real_region)])
    
    def show_external_conditions(self):
        for row in self.external_conditions:
            print(f'Maintain {row[2]} micromolar of {self.id_to_mol[row[1]]} within a space of {row[0].sum()} voxels ({row[3]} molecules)')
            
    @nvtx.annotate("maintain_external_conditions()", color="purple")
    def maintain_external_conditions(self):
        for row in self.external_conditions:
            if row[2] == 0:
                # Just remove everything in it
                self.voxel_matrix[row[1], row[0]] = 0
                # print('emptied region')
            else:
                with nvtx.annotate("sum", color="green"):
                    current = int(self.voxel_matrix[row[1], row[0]].sum())
                    change = row[3] - current
#                     print(f'Current: {current}, ideally {row[3]}, change: {change}')
                if change > 0:
                    with nvtx.annotate("where", color="green"):
                        candidates = row[4]
                    with nvtx.annotate("choice", color="green"):
                        choices = cp.random.random(change, dtype=cp.float32) * len(candidates[0])
                    with nvtx.annotate("select", color="green"):    
                        # Use int32! Because choices may be a big number
                        selections = [x[choices.astype(cp.int32)] for x in candidates]
                    with nvtx.annotate("apply", color="green"):
                        if self.ndim == 2:
                            self.voxel_matrix[row[1], selections[0], selections[1]] = 1
#                                 self.voxel_matrix[row[1], selections[0], selections[1]] += 1
                        if self.ndim == 3:
                            self.voxel_matrix[row[1], selections[0], selections[1], selections[2]] = 1
#                                 self.voxel_matrix[row[1], selections[0], selections[1], selections[2]] += 1
                        # print(f'Added {change} molecules')
                elif change < 0:
                    with nvtx.annotate("neg change", color="green"):
                        # Hacky way to save time, doesn't work for low conc
                        candidates = cp.where((self.voxel_matrix[row[1]] * row[0]) == 1)
#                         candidates = row[4]
                        choices = cp.random.random(-change, dtype=cp.float32) * len(candidates[0])
                        selections = [x[choices.astype(cp.int32)] for x in candidates]
                        if self.ndim == 2:
#                                 self.voxel_matrix[row[1], selections[0], selections[1]] -= 1
                            self.voxel_matrix[row[1], selections[0], selections[1]] = 0
                        if self.ndim == 3:
#                                 self.voxel_matrix[row[1], selections[0], selections[1], selections[2]] -= 1
                            self.voxel_matrix[row[1], selections[0], selections[1], selections[2]] = 0
                        # print(f'Deleted {change} molecules')

    @nvtx.annotate("react_diffuse()", color="purple")
    def react_diffuse(self, t_step, warning=True):
        with nvtx.annotate("diffuse", color="orange"):
            with nvtx.annotate("rand setup", color="orange"):
                self.diffuse_voxel = cp.zeros(self.voxel_matrix_shape, dtype=cp.float16)
                random_choice = (cp.random.random(self.voxel_matrix_shape, dtype=cp.float32) * 2 * self.ndim + 1).astype(cp.int16)
                random_sampling = cp.random.random(self.voxel_matrix_shape, dtype=cp.float32) / t_step
            for idx in range(self.num_types):
                if self.current_conc[idx] == 0:
                        continue
                if isinstance(self.diffusion_vector[idx], float) and self.diffusion_vector[idx] == 0:
                    continue
                # Diffuse part
                with nvtx.annotate("vox1", color="green"):    
                    self.diffuse_voxel[idx] = self.voxel_matrix[idx] * self.diffusion_vector[idx] 
                    self.active_mol += 1
                    self.active_mol_list += ' ' + self.id_to_mol[idx]
            with nvtx.annotate("choice", color="green"):
                diffusion_choice = random_choice * (random_sampling < self.diffuse_voxel)

            with nvtx.annotate("move_diffuse", color="green"):
                with nvtx.annotate("dir1", color="purple"):
                    with nvtx.annotate("move_action", color="brown"):
                        move_action = (diffusion_choice[:, 1:] == 1) * self.not_barrier_matrix_up
                    with nvtx.annotate("plus_move", color="brown"):
                        self.voxel_matrix[:, 1:] -= move_action
                    with nvtx.annotate("minus_move", color="brown"):
                        self.voxel_matrix[:, :-1] += move_action
                with nvtx.annotate("dir2", color="purple"):
                    move_action = (diffusion_choice[:, :-1] == 2) * self.not_barrier_matrix_down
                    self.voxel_matrix[:, :-1] -= move_action
                    self.voxel_matrix[:, 1:] += move_action
                with nvtx.annotate("dir3", color="purple"):
                    move_action = (diffusion_choice[:, :, 1:] == 3) * self.not_barrier_matrix_left
                    self.voxel_matrix[:, :, 1:] -= move_action
                    self.voxel_matrix[:, :, :-1] += move_action
                with nvtx.annotate("dir4", color="purple"):
                    move_action = (diffusion_choice[:, :, :-1] == 4) * self.not_barrier_matrix_right
                    self.voxel_matrix[:, :, :-1] -= move_action
                    self.voxel_matrix[:, :, 1:] += move_action
                if self.ndim >= 3:
                    with nvtx.annotate("dir5", color="purple"):
                        move_action = (diffusion_choice[:, :, :, 1:] == 5) * self.not_barrier_matrix_front
                        self.voxel_matrix[:, :, :, 1:] -= move_action
                        self.voxel_matrix[:, :, :, :-1] += move_action
                    with nvtx.annotate("dir6", color="purple"):
                        move_action = (diffusion_choice[:, :, :, :-1] == 6) * self.not_barrier_matrix_back
                        self.voxel_matrix[:, :, :, :-1] -= move_action
                        self.voxel_matrix[:, :, :, 1:] += move_action
    
       
        with nvtx.annotate("react", color="orange"):
            if self.reaction_set is not None:
                # React part
                with nvtx.annotate("random", color="green"):
                    # Filter by possible locations of reactions
                    random_sampling = cp.random.random(self.reaction_voxel_shape, dtype=cp.float32) #* self.reaction_location
            
                for idx, (reagent, coeff) in enumerate(zip(self.reagent_vector_list, self.reaction_coefficients)):
                    # Only if no diffusion happened there - guarantees no negative mol count
                    if self.current_conc[reagent[0]] == 0:
                        continue
                    if len(reagent) == 2:
                        with nvtx.annotate("react 2 exp", color="green"):
                            if self.current_conc[reagent[1]] == 0:
                                continue
                            scaling = (-coeff / 6.023e23 / self.spacing**3 / 1000 * t_step)
                            exponent = self.voxel_matrix[reagent[0]] * self.voxel_matrix[reagent[1]] * scaling
                        with nvtx.annotate("react 2 voxel", color="green"):
                            reaction_voxel = 1 - cp.exp(exponent)
                    elif len(reagent) == 1:
                        with nvtx.annotate("react 1 exp", color="green"):
                            scaling = (-coeff * t_step)
                            exponent = self.voxel_matrix[reagent[0]] * scaling
                        with nvtx.annotate("react 1 voxel", color="green"):
                            reaction_voxel = 1 - cp.exp(exponent)
                    self.active_rxn += 1
                    with nvtx.annotate("move_react", color="green"):   
                        self.voxel_matrix += self.reaction_matrix_list[idx] * (random_sampling[idx] < reaction_voxel)
                        
        self.cumulative_t += t_step
        self.current_conc = self.voxel_matrix.astype(cp.int32).sum(tuple(range(1, self.ndim+1))).get()

                
    def simulate(self, steps, t_step=None, 
                 maintain_every=100, log_every=500, 
                 traj_every=10000, checkpoint_every=10000, 
                 warning=True, out_freq=0.1):
        
        self.current_conc = self.voxel_matrix.astype(cp.float32).sum(tuple(range(1, self.ndim+1))).get()
        
        if not self.initialized:
            self.initialize()
        
        if t_step is not None:
            self.t_step = t_step
        
        step_trace = []
        t_trace = []
        conc_trace = []
        self.active_mol = 0
        self.active_mol_list = ''
        self.active_rxn = 0

        with tqdm(total=steps+1, bar_format="{percentage:3.1f}% |{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, {postfix[0]}={postfix[1][value]} ({postfix[1][list]} ), {postfix[2]}={postfix[3][value]:2d}]", mininterval=out_freq,
          postfix=["Act Mol", {"value": 0, 'list': 0}, "Act Rxn", {"value": 0}]) as t:
            for step in range(0, steps+1):
                if step == 0: 
                    print(f'Simulate {steps + 1} steps')

                if checkpoint_every is not None:
                    if self.step % checkpoint_every == 0:
                        pickle.dump(self, open(self.checkpoint_filename, 'wb'))

                if log_every is not None:
                    if (step == 0 and self.start_from_scratch) or (step > 0 and self.step % log_every == 0):
                        step_trace.append(self.step)
                        t_trace.append(self.cumulative_t)
                        conc_trace.append(self.current_conc)

                if traj_every is not None:        
                    if self.step % traj_every == 0:
                        # save step, sim time, trajectory, also logs
                        self.save_traj(self.step, step_trace, t_trace, conc_trace)
                        self.frame += 1
                        step_trace = []
                        t_trace = []
                        conc_trace = []

                with nvtx.annotate("simulate", color="orange"):
                    if step % maintain_every == 0:
                        self.maintain_external_conditions()
                    self.react_diffuse(self.t_step, warning=warning)

                self.step += 1
                t.postfix[1]["value"] = self.active_mol
                t.postfix[1]["list"] = self.active_mol_list
                t.postfix[3]["value"] = self.active_rxn
                t.update()
                self.active_mol = 0
                self.active_mol_list = ''
                self.active_rxn = 0
                self.start_from_scratch = False # So that further restarts won't have duplicate frames
    
    def save_traj(self, step, step_trace, t_trace, conc_trace):
        # Open HDF5 as append
        # Append dataset with name {self.frame} to data group
        # dataset attributes: steps (step), cum_t
        # Also dump step_trace, t_trace and conc_trace into log group
        with h5py.File(self.traj_filename, 'a') as f:
            if 'traj' not in f:
                traj_data = f.create_dataset(f'traj/data', (1, *self.voxel_matrix_shape), maxshape=(None, *self.voxel_matrix_shape), compression='gzip')
                traj_steps = f.create_dataset(f'traj/steps', (1), maxshape=(None,))
                traj_cum_t = f.create_dataset(f'traj/cumulative_t', (1), maxshape=(None,))
                traj_time_step = f.create_dataset(f'traj/time_step', (1), maxshape=(None,))   
            else:
                traj_data = f['traj/data']
                traj_steps = f['traj/steps']
                traj_cum_t = f['traj/cumulative_t']
                traj_time_step = f['traj/time_step']
            if self.frame >= len(traj_data):
                traj_data.resize(len(traj_data) + 1, axis=0)
                traj_steps.resize(len(traj_steps) + 1, axis=0)
                traj_cum_t.resize(len(traj_cum_t) + 1, axis=0)
                traj_time_step.resize(len(traj_time_step) + 1, axis=0)
            traj_data[self.frame] = self.voxel_matrix.get()
            traj_steps[self.frame] = self.step
            traj_cum_t[self.frame] = self.cumulative_t
            traj_time_step[self.frame] = self.t_step
            if len(step_trace) > 0:
                if 'log' not in f:
                    log_step = f.create_dataset(f'log/step', len(step_trace), maxshape=(None,), data=step_trace)
                    log_t = f.create_dataset(f'log/t', len(t_trace), maxshape=(None,), data=t_trace)
                    log_conc = f.create_dataset(f'log/count', (len(conc_trace), self.num_types), maxshape=(None, self.num_types), data=conc_trace)
                else:
                    log_step = f['log/step']
                    log_t = f['log/t']
                    log_conc = f['log/count']
                    log_step.resize(len(log_step)+len(step_trace), axis=0)
                    log_t.resize(len(log_t)+len(t_trace), axis=0)
                    log_conc.resize(len(log_conc)+len(conc_trace), axis=0)
                    log_step[-len(step_trace):] = step_trace
                    log_t[-len(t_trace):] = t_trace
                    log_conc[-len(conc_trace):] = conc_trace
    
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
            barrier_location = [self.mesh[x][barrier_present.get()].get() for x in range(self.ndim)]
            plt.scatter(barrier_location[0], barrier_location[1], s = 25, c = 'gray', marker='s')
        for mol in mol_type:
            idx = self.mol_to_id[mol]
            particles_present = np.where(self.voxel_matrix[idx].get() > 0)
            particle_number = self.voxel_matrix[idx][particles_present].get().astype(int)
            particle_location = np.array([np.repeat(self.mesh[x].get()[particles_present], particle_number) 
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
            barrier_location = [self.mesh[x][barrier_present.get()].get() for x in range(self.ndim)]
            ax.scatter(barrier_location[0], barrier_location[1], barrier_location[2], s = 25, c = 'gray', marker='s')
        for mol in mol_type:
            idx = self.mol_to_id[mol]
            particles_present = np.where(self.voxel_matrix[idx].get() > 0)
            particle_number = self.voxel_matrix[idx][particles_present].get().astype(int)
            particle_location = np.array([np.repeat(self.mesh[x][particles_present].get(), particle_number) 
                                 for x in range(self.ndim)])
            # randomize location a bit
            particle_location += (np.random.random(size=particle_location.shape) - 0.5) * self.spacing * 0.5
            # if idx < 2:
            ax.scatter(particle_location[0], particle_location[1], particle_location[2], s=2)
            # else:
            #     ax.scatter(particle_location[0], particle_location[1], particle_location[2], s=9, marker='x')    
        xs, ys, zs = [self.mesh[x].get() for x in range(self.ndim)]
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
    def add_reaction(self, reagent, product, reaction_coefficient, location=None):
        # reagent can be [typeA, typeB] for bimolecular reaction
        # or [typeA] or typeA for unimolecular reaction
        self.reaction.append([reagent, product, reaction_coefficient, location])
        