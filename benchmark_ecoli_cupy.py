import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
from RediCell_cupy import *
from DesignTool import *
import argparse
import nvtx
# from cupyx.profiler import benchmark
# import cupy

def main(steps):
    d = DesignTool()
    d.get_blanket_space([128, 64, 64], spacing=16e-9)
    d.add_ecoli_rod(l=2e-6, r=0.4e-6, barrier_type=1, space_type=1, offsety=0)#, offsety=1.6e-7)
    d.set_border_wall()
    # mol set
    mol_Iex = Molecule('Iex', diffusion_coefficient=1.28e-12, observed_barrier_types=None)
    mol_Iin = Molecule('Iin', diffusion_coefficient=1.28e-12, observed_barrier_types=None)
    mol_Y = Molecule('Y', diffusion_coefficient=0, observed_barrier_types=None)
    mol_YI = Molecule('YI', diffusion_coefficient=0, observed_barrier_types=None)
    molset = MoleculeSet([mol_Iex, mol_Iin, mol_Y, mol_YI])
    # rx set
    rxset = ReactionSet()
    rxset.add_reaction(['Iex'], ['Iin'], 2.33e-3)
    rxset.add_reaction(['Iin'], ['Iex'], 2.33e-3)
    rxset.add_reaction(['Y', 'Iex'], ['YI'], 3.03e+4)
    rxset.add_reaction(['YI'], ['Y', 'Iex'], 1.2e-1)
    rxset.add_reaction(['YI'], ['Y', 'Iin'], 1.2e+1)

    a = RediCell_CuPy(design=d, molecule_types=molset, reaction_set=rxset, t_step=2e-3)
    
    a_supply_matrix = cp.zeros(a.true_sides)
    a_supply_matrix[(a.special_space_type == 0) * (a.barrier_type == -1)] = 1
    a_supply_matrix[1:-1, 1:-1, [1, -2]] = 1
    a_supply_matrix[1:-1, [1, -2], 1:-1] = 1
    a_supply_matrix[[1, -2], 1:-1, 1:-1] = 1

    a.add_external_conditions(a_supply_matrix, mol_Iex, 5)
    a.show_external_conditions()
    
    a.partition()
    a.configure_barrier()
    a.maintain_external_conditions()
    
    # place 1000 Y where E coli barrier is
    barriers = np.where(a.barrier_type.get() == 1)
    num_voxel = len(barriers[0])
    print(num_voxel)
    choice = np.random.choice(num_voxel, 1000, replace=False)
    selection = [x[choice] for x in barriers]
    a.voxel_matrix[a.mol_to_id['Y'], selection[0], selection[1], selection[2]] = 1
    
    a.simulate(steps, t_step=5e-5, plot_every=None, timing=False, traj_every=None, checkpoint_every=None,
               traj_filename='ecoli_test.npy', checkpoint_filename='ecoli_test.pkl')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5000)
    args = parser.parse_args()
    main(steps=args.steps)
