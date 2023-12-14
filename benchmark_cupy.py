import numpy as np
import time
from RediCell_cupy import *
import argparse


def main(dim, side, steps):
    mol_A = Molecule('A', diffusion_coefficient=8.15e-14)
    mol_B = Molecule('B', diffusion_coefficient=8.15e-14)
    mol_C = Molecule('C', diffusion_coefficient=8.15e-14)
    molset = MoleculeSet([mol_A, mol_B, mol_C])
    
    rxset = ReactionSet()
    rxset.add_reaction(['A', 'B'], ['C'], 1.07e5)
    rxset.add_reaction(['C'], ['A', 'B'], 0.351)

    a = RediCell_CuPy(sides=[side]*dim, spacing=31.25e-9, molecule_types=molset, t_step=None)
    a.add_reaction_set(rxset)
    a.partition()
    
    num_mol = int(0.03 * side **dim)
    print(f'Put {num_mol} molecules')
    for i in range(a.num_types-1):
        while a.voxel_matrix[i].sum() < num_mol:
            if dim == 2:
                a.voxel_matrix[i, np.random.randint(1, side, num_mol - int(a.voxel_matrix[i].sum())), 
                               np.random.randint(1, side, num_mol - int(a.voxel_matrix[i].sum()))] = 1
            if dim == 3:
                a.voxel_matrix[i, np.random.randint(1, side, num_mol - int(a.voxel_matrix[i].sum())), 
                               np.random.randint(1, side, num_mol - int(a.voxel_matrix[i].sum())),
                               np.random.randint(1, side, num_mol - int(a.voxel_matrix[i].sum()))] = 1
    
    a.simulate(steps, t_step=1e-4, plot_every=None, timing=True)

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dim', type=int, default=2)
    parser.add_argument('-s', '--side', type=int, default=64)
    parser.add_argument('--steps', type=int, default=5000)
    args = parser.parse_args()
    main(dim=args.dim, side=args.side, steps=args.steps)