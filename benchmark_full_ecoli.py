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

def main(steps, name, conc, out_freq):
    d = DesignTool()
    d.get_blanket_space([119, 57, 57], spacing=16e-9, wall=False)
    d.add_ecoli_rod(l=1.8e-6, r=0.4e-6, barrier_type=1, space_type=1, offsety=0, thickness=2, method='const')#, offsety=1.6e-7)
    d.special_space_type[d.barrier_type==1] = 2
    d.barrier_type[d.special_space_type==1] = 2
    d.barrier_type[d.special_space_type==0] = 3
    cyt = np.where(d.special_space_type == 1)
    v_cyt = len(cyt[0])
    print(v_cyt)
    v_cyt_b = np.random.choice(v_cyt, v_cyt//2, replace=False)
    b_sel = [x[v_cyt_b] for x in cyt]
    d.barrier_type[b_sel[0], b_sel[1], b_sel[2]] = 4

    # Type   Space_type    Barrier_type
    #    0   External      Border wall
    #    1   Cytoplasm     Membrane
    #    2   Membrane      Cytoplasm
    #    3   None          External
    #    4   None          Cytoplasm_blockers


    
    # mol set
    mol_Iex =   Molecule('Iex',   diffusion_coefficient=1.28e-12, observed_barrier_types=None)
    mol_I =     Molecule('I',   diffusion_coefficient=1.28e-12, observed_barrier_types=None)
    mol_Y =     Molecule('Y',     diffusion_coefficient=0.1e-12,  observed_barrier_types=[2, 3, 4])
    mol_YI =    Molecule('YI',    diffusion_coefficient=0.1e-12,  observed_barrier_types=[2, 3, 4])
    mol_R2 =    Molecule('R2',    diffusion_coefficient=1e-12,    observed_barrier_types=[1, 4])
    mol_R2O =   Molecule('R2O',   diffusion_coefficient=1.28e-12, observed_barrier_types=[1, 4])
    mol_IR2 =   Molecule('IR2',   diffusion_coefficient=1.28e-12, observed_barrier_types=[1, 4])
    mol_O =     Molecule('O',     diffusion_coefficient=1.28e-12, observed_barrier_types=[1, 4])
    mol_IR2O =  Molecule('IR2O',  diffusion_coefficient=1.28e-12, observed_barrier_types=[1, 4])
    mol_I2R2O = Molecule('I2R2O', diffusion_coefficient=1.28e-12, observed_barrier_types=[1, 4])
    mol_mY =    Molecule('mY',    diffusion_coefficient=0.1e-12,  observed_barrier_types=[3, 4])
    mol_I2R2 =  Molecule('I2R2',  diffusion_coefficient=1.28e-12, observed_barrier_types=[1, 4])
    molset = MoleculeSet([mol_Iex, mol_I, mol_Y, mol_YI, mol_R2, mol_R2O, mol_IR2, mol_O, mol_IR2O, mol_I2R2O, mol_mY, mol_I2R2])
    
    # rx set
    rxset = ReactionSet()
    rxset.add_reaction(['O', 'R2'], ['R2O'], 2.43e6)
    rxset.add_reaction(['O', 'IR2'], ['IR2O'], 1.21e6)
    rxset.add_reaction(['O', 'I2R2'], ['I2R2O'], 2.43e4)
    rxset.add_reaction(['R2O'], ['R2', 'O'], 6.3e-4)
    rxset.add_reaction(['IR2O'], ['IR2', 'O'], 6.3e-4)
    rxset.add_reaction(['I2R2O'], ['I2R2', 'O'], 3.15e-1)
    rxset.add_reaction(['O'], ['mY', 'O'], 1.26e-1)
    rxset.add_reaction(['mY'], ['mY', 'Y'], 4.44e-2, location=d.special_space_type==2)
    rxset.add_reaction(['mY'], [], 1.11e-2, location=d.special_space_type==2)
    rxset.add_reaction(['Y'], [], 2.1e-4)
    rxset.add_reaction(['R2', 'I'], ['IR2'], 9.71e4)
    rxset.add_reaction(['IR2', 'I'], ['I2R2'], 4.85e4)
    rxset.add_reaction(['R2O', 'I'], ['IR2O'], 2.24e4)
    rxset.add_reaction(['IR2O', 'I'], ['I2R2O'], 1.12e4)
    rxset.add_reaction(['IR2'], ['I', 'R2'], 2e-1)
    rxset.add_reaction(['I2R2'], ['I', 'IR2'], 4e-1)
    rxset.add_reaction(['IR2O'], ['I', 'R2O'], 1.0)
    rxset.add_reaction(['I2R2O'], ['I', 'IR2O'], 2.0)
    rxset.add_reaction(['Iex'], ['I'], 2.33e-3)
    rxset.add_reaction(['I'], ['Iex'], 2.33e-3)
    rxset.add_reaction(['Y', 'Iex'], ['YI'], 3.03e+4)
    rxset.add_reaction(['YI'], ['Y', 'Iex'], 1.2e-1)
    rxset.add_reaction(['YI'], ['Y', 'I'], 1.2e+1)

    a = RediCell_CuPy(design=d, molecule_types=molset, reaction_set=rxset, t_step=50e-6, project_name=name)
    a_supply_matrix = cp.zeros(a.true_sides)
    a_supply_matrix[:, :, [0, -1]] = 1
    a_supply_matrix[:, [0, -1], :] = 1
    a_supply_matrix[[0, -1], :, :] = 1

    if conc > 0:
        a.add_external_conditions(a_supply_matrix, mol_Iex, conc)
        a.show_external_conditions()
    
    a.partition()
    a.configure_barrier()
    
    # Initial conditions
    # Type   Space_type    Barrier_type
    #    0   External      Border wall
    #    1   Cytoplasm     Membrane
    #    2   Membrane      Cytoplasm
    #    3   None          External
    #    4   None          Cytoplasm_blockers
    a.add_molecules((a.special_space_type == 1) & (a.barrier_type != 4), 'O', count=1)
    a.add_molecules((a.special_space_type == 1) & (a.barrier_type != 4), 'R2', count=10)
    a.add_molecules(a.special_space_type == 2, 'Y', count=30)
    if conc > 0:
        a.add_molecules(a.special_space_type > -1, 'Iex', uM=conc)
        a.add_molecules(a.special_space_type > -1, 'I', uM=conc)

    
    a.maintain_external_conditions()
    
    a.determine_maximum_timestep()
    
    a.simulate(steps, t_step=5e-5, traj_every=50000, log_every=100, checkpoint_every=50000, out_freq=out_freq)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--name', type=str, default='redicell')
    parser.add_argument('--conc', type=float, default=5.0)
    parser.add_argument('--out_freq', type=float, default=0.1) # Change to 5 if too much output
    args = parser.parse_args()
    main(steps=args.steps, name=args.name, conc=args.conc, out_freq=args.out_freq)
