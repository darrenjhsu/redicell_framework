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


    a = RediCell_CuPy(design=d, molecule_types=molset, reaction_set=rxset, t_step=50e-6)
    a_supply_matrix = cp.zeros(a.true_sides)
    a_supply_matrix[:, :, [0, -1]] = 1
    a_supply_matrix[:, [0, -1], :] = 1
    a_supply_matrix[[0, -1], :, :] = 1

    a.add_external_conditions(a_supply_matrix, mol_Iex, 5)
    a.show_external_conditions()
    
    a.partition()
    a.configure_barrier()
    
    # Initial conditions

    # 1 O and 9 R2
    cyto_avail = np.where((a.special_space_type == 1) & (a.barrier_type != 4))
    n_ca = len(cyto_avail[0])
    OR2 = np.random.choice(n_ca, 10, replace=False)
    ca_sel = [x[OR2] for x in cyto_avail]

    a.voxel_matrix[a.mol_to_id['O'], ca_sel[0][0], ca_sel[1][0], ca_sel[2][0]] = 1
    a.voxel_matrix[a.mol_to_id['R2'], ca_sel[0][1:], ca_sel[1][1:], ca_sel[2][1:]] = 1

    # 30 Y
    mem_avail = np.where(a.special_space_type == 2)
    n_ma = len(mem_avail[0])
    OR2 = np.random.choice(n_ma, 30, replace=False)
    ma_sel = [x[OR2] for x in mem_avail]
    a.voxel_matrix[a.mol_to_id['Y'], ma_sel[0], ma_sel[1], ma_sel[2]] = 1

    # 5 uM of Iex and I
    ext_avail = np.where(a.special_space_type > -1)
    n_ea = len(ext_avail[0])
    n_Iex = int(n_ea / a.one_per_voxel_equal_um * 5.0)
    print(n_ea, n_Iex)
    Iex = np.random.choice(n_ea, n_Iex, replace=False)
    ea_sel = [x[Iex] for x in ext_avail]
    a.voxel_matrix[a.mol_to_id['Iex'], ea_sel[0], ea_sel[1], ea_sel[2]] = 1

    cyto_all = np.where(a.special_space_type > -1)
    n_call = len(cyto_all[0])
    n_Iin = int(n_call / a.one_per_voxel_equal_um * 5.0)
    Iin = np.random.choice(n_ea, n_Iin, replace=False)
    call_sel = [x[Iin] for x in cyto_all]
    a.voxel_matrix[a.mol_to_id['I'], call_sel[0], call_sel[1], call_sel[2]] = 1

    
    a.maintain_external_conditions()
    
    a.determine_maximum_timestep()
    
    a.simulate(steps, t_step=5e-5, traj_every=5000, log_every=100, checkpoint_every=50000)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=5000)
    args = parser.parse_args()
    main(steps=args.steps)
