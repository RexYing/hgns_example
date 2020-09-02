import argparse
import pickle as pkl

import taichi as ti
import math
import numpy as np
import sklearn.cluster as cluster
from engine.mpm_solver import MPMSolver

import sys, os
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
from hgns.util import cluster_scene

def parse_mpm():
    parser = argparse.ArgumentParser(description='MPM sim arguments')
    parser.add_argument('--dataset_name', type=str,
                        help='Path under data/ to save the simulation.')
    parser.add_argument('--num_sims', type=int,
                        help='Number of groundtruth simulations to generate.')
    parser.add_argument('--num_frames', type=int,
                        help='Number of groundtruth simulation frames to generate per trajectory.')
    parser.add_argument('--step_size', type=float,
                        help='Simulation step size')
    parser.add_argument('--young', type=int,
                        help='Young\'s coefficients')
    parser.add_argument('--ncubes', type=int,
                        help='Number of boxes to be dropped')
    parser.add_argument('--material', type=str,
                        help='Material of boxes to be dropped')
    parser.add_argument('--gui_particle_radius', type=float,
                        help='particle radius for GUI visualization')
    parser.add_argument('--gui_write_to_disk', action='store_true', default=False,
                        help='particle radius for GUI visualization')

    parser.set_defaults(dataset_name='water',
                        num_sims=1000,
                        num_frames=500,
                        step_size=8e-3,
                        young=1e5,
                        ncubes=3,
                        material='water',
                        gui_particle_radius=1.5,
                        gui_write_to_disk=False)
    return parser

def cube_overlap(lower_corners, cube_sizes, new_lower_corner, new_cube_size):
    eps = 0.03
    new_upper_corner = new_lower_corner + new_cube_size
    for lc, cs in zip(lower_corners, cube_sizes):
        overlap = True
        uc = lc + cs
        if lc[0] > new_upper_corner[0] + eps or new_lower_corner[0] > uc[0] + eps:
            overlap = False
                          
        if lc[1] > new_upper_corner[1] + eps or new_lower_corner[1] > uc[1] + eps:
            overlap = False

        if overlap:
            return True
                                          
    return False

def process_particle_features(particles):
    material = np.expand_dims(particles['material'], axis=1)
    return np.concatenate((particles['position'], 
                           particles['velocity'],
                           material), axis=1)

def simulate(mpm, gui, args):
    particle_states = []
    for frame in range(args.num_frames):
        mpm.step(args.step_size)
        
        colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00], dtype=np.uint32)
        particles = mpm.particle_info()
    
        particle_states.append(process_particle_features(particles))
    
        gui.circles(particles['position'], radius=args.gui_particle_radius, 
                    color=colors[particles['material']])
        gui.show(f'{frame:06d}.png' if args.gui_write_to_disk else None)
    
    particle_states = np.stack(particle_states, axis=0)
    return particle_states

def rollout(trajectory, gui, radius=1.5):
    colors = np.array([0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00], dtype=np.uint32)
    nframes = trajectory.shape[0]
    for i in range(nframes):
        curr_frame = trajectory[i]
        gui.circles(curr_frame[:, :2], radius=radius, 
                    color=colors[curr_frame[:, -1].astype(int)])
        gui.show(None)

def main():
    parser = parse_mpm()
    args = parser.parse_args()
    simulation_trajectories = []
    
    if args.material == 'water':
        material = MPMSolver.material_water
    elif args.material == 'elastic':
        material = MPMSolver.material_elastic
    elif args.material == 'snow':
        material = MPMSolver.material_snow
    elif args.material == 'sand':
        material = MPMSolver.material_sand

    for sim_i in range(args.num_sims):
        ti.init(arch=ti.cuda)  # Try to run on GPU
        gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
    
        mpm = MPMSolver(res=(128, 128))
        mpm.E = args.young
        # init condition
        n_samples = args.ncubes * 10

        lower_corner_candidates = np.random.uniform(0.2, 0.8, size=(n_samples, 2))
        cube_size_candidates = np.random.uniform(0.06, 0.15, size=(n_samples, 2))
        
        lower_corners = []
        cube_sizes = []
        num_cubes = args.ncubes
        for s in range(1, n_samples):
            if not cube_overlap(lower_corner_candidates[:s], cube_size_candidates[:s], 
                                lower_corner_candidates[s], cube_size_candidates[s]):
                lower_corners.append(lower_corner_candidates[s])
                cube_sizes.append(cube_size_candidates[s])
            if len(lower_corners) == num_cubes:
                break
        
        for i in range(len(lower_corners)):
            mpm.add_cube(lower_corner=lower_corners[i],
                         cube_size=cube_sizes[i],
                         material=material)
        
        simulation_trajectories.append(simulate(mpm, gui, args))
        #rollout(simulation_trajectories[0], gui, args.gui_particle_radius)
        #cluster_scene(simulation_trajectories[-1][0])
    
        if (sim_i+1) % 1 == 0:
            print('Simulated {} trajectories.'.format(sim_i+1))

        data_path = os.path.join('../data/', args.dataset_name)
        if not os.path.exists(data_path):
            os.makedirs(data_path)
    
        if (sim_i+1) % 100 == 0 or sim_i+1 == args.num_sims:
            print(len(simulation_trajectories))
            with open('../data/{}/{}_{}.pkl'.format(args.dataset_name, args.material, sim_i // 100), 'wb') as f:
                pkl.dump(simulation_trajectories, f)
            simulation_trajectories = []
    
if __name__ == '__main__':
    main()
    

