import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from scipy import ndimage

class DesignTool:
    def __init__(self):
        pass

    def get_blanket_space(self, sides, spacing, wall=True):
        self.sides = np.array(sides)
        self.ndim = len(self.sides)
        self.spacing = float(spacing)
        self.wall = wall
        print(f'Sides are {self.sides} with spacing of {self.spacing} m ({self.ndim} dimensions).')
        print(f'Overall system sizes are {self.sides * self.spacing} m')
        if self.wall:
            self.true_sides = self.sides + 2
        else:
            self.true_sides = self.sides
        self.voxels = np.zeros(self.true_sides)
        self.center_v = (np.array(self.true_sides) - 1) // 2
        # This is so center is at (0, 0, 0)
        self.side_coord = [np.linspace(-(int((side-1) / 2)) * self.spacing, int((side) / 2) * self.spacing, side) for side in self.true_sides]
        self.center_x = np.array([self.side_coord[x][self.center_v[x]] for x in range(len(self.center_v))])
        
        self.mesh = np.meshgrid(*self.side_coord, indexing='ij')
        # Barrier types: -1 = no barrier, 0 = surrounding wall, 1 -> N = custom
        self.barrier_type = np.zeros(self.true_sides).astype(int) - 1
        # Barrier types: 0 = default (diffusion the same everywhere), 1 -> N = custom defined locations
        self.special_space_type = np.zeros(self.true_sides).astype(int)

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

    
    def draw2D(self, include_barrier_type=None, exclude_barrier_type=None,
               include_space_type=None, exclude_space_type=None):
        if self.ndim == 3:
            print('Use draw3D()')
            return
        my_barrier_cmap = matplotlib.colormaps['tab10']
        my_barrier_cmap.set_under('w')
        my_space_cmap = matplotlib.colormaps['Set1']
        my_space_cmap.set_under('w')
        barrier_types = set(np.unique(self.barrier_type))
        space_types = set(np.unique(self.special_space_type))
        
        if exclude_barrier_type == 'all':
            pass
        else:
            if include_barrier_type is not None: # Include takes precedence
                barrier_show = np.zeros_like(self.barrier_type) - 1
                for idx in include_barrier_type:
                    barrier_show[self.barrier_type == idx] = idx
            elif exclude_barrier_type is not None:
                barrier_show = self.barrier_type.copy()
                for idx in exclude_barrier_type:
                    barrier_show[barrier_show == idx] = -1
            else:
                barrier_show = self.barrier_type
            barrier_show = np.ma.masked_array(barrier_show, barrier_show < 0)
            plt.imshow(barrier_show, cmap=my_barrier_cmap, vmin=0, alpha=0.9)
        if exclude_space_type == 'all':
            pass
        else:
            if include_space_type is not None: # Include takes precedence
                space_show = np.zeros_like(self.special_space_type) - 1
                for idx in include_space_type:
                    space_show[self.special_space_type == idx] = idx
            elif exclude_space_type is not None:
                space_show = self.special_space_type.copy()
                for idx in exclude_space_type:
                    space_show[space_show == idx] = -1
            else:
                space_show = self.special_space_type
            space_show = np.ma.masked_array(space_show)
            plt.imshow(space_show, cmap=my_space_cmap, vmin=0, alpha=0.2)
        plt.figure(figsize=(6,6))
    
    def draw3D(self, include_barrier_type=None, exclude_barrier_type=None,
               include_space_type=None, exclude_space_type=None):
        if self.ndim == 2:
            print('Use draw2D()')
            return
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection='3d')
        barrier_types = set(np.unique(self.barrier_type))
        space_types = set(np.unique(self.special_space_type))
        if exclude_barrier_type == 'all':
            pass
        else:
            if include_barrier_type is not None: # Include takes precedence
                plot_types = barrier_types.intersection(set(include_barrier_type))
            elif exclude_barrier_type is not None:
                plot_types = barrier_types - set(exclude_barrier_type)
            else:
                plot_types = barrier_types
            
            for barrier_idx in plot_types:
                if barrier_idx >= 0:
                    xs, ys, zs = [self.mesh[x][self.barrier_type == barrier_idx] for x in range(self.ndim)]
                    ax.scatter(xs, ys, zs, s = 25, marker='s', alpha=0.02)
        if exclude_space_type == 'all':
            pass
        else:
            if include_space_type is not None: # Include takes precedence
                plot_types = space_types.intersection(set(include_space_type))
            elif exclude_space_type is not None:
                plot_types = space_types - set(exclude_space_type)
            else:
                plot_types = space_types
            
            for space_idx in plot_types:
                if space_idx >= 0:
                    xs, ys, zs = [self.mesh[x][self.special_space_type == space_idx] for x in range(self.ndim)]
                    ax.scatter(xs, ys, zs, s = 25, marker='s', alpha=0.005)
        
        xs, ys, zs = [self.mesh[x] for x in range(self.ndim)]
        ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs)))
        ax.set_xlim([self.side_coord[0][0], self.side_coord[0][-1]])
        ax.set_ylim([self.side_coord[1][0], self.side_coord[1][-1]])
        ax.set_zlim([self.side_coord[2][0], self.side_coord[2][-1]])
        # plt.title(f't = {self.cumulative_t:.3e} s')
        plt.grid(alpha=0.3)
        plt.show()

    def shift_center(self, center_v, offset):
        return center_v + (np.array(offset[:len(center_v)])/self.spacing).astype(int)

    def distance_to_center(self, mesh, center):
        if self.ndim == 2:
            return np.sqrt(((np.concatenate([x[None, ...] for x in mesh]) - np.array(center)[:, None, None])**2).sum(0))
        if self.ndim == 3:
            return np.sqrt(((np.concatenate([x[None, ...] for x in mesh]) - np.array(center)[:, None, None, None])**2).sum(0))

    def determine_coord(self, v):
        print(f'Voxel {v} has a coordinate of {np.array([self.side_coord[x][v[x]] for x in range(len(v))])}')
        return np.array([self.side_coord[x][v[x]] for x in range(len(v))])

    def find_seperating_boundary(self, distance, r, return_factor=False):
        # Adjust factor until a full separating boundary can be drawn
        factor = 5.0
        while factor > 0:
            factor -= 0.1
            object = (distance > (r - self.spacing / factor)) * (distance < (r + self.spacing / factor))
            label = ndimage.label(1-object)
            if label[1] >= 2:
                # print(f'Determined factor for a separating boundary as {factor:.1f}')
                return object, label, factor
            if factor == 0.0:
                raise("Could not determine a good factor to generate a separating boundary")
                
    def add_box(self, x, y, barrier_type, space_type=None, z=None, offsetx=0, offsety=0, offsetz=0):
        if self.ndim == 3:
            assert z is not None
        # Put a centered box
        center_v = self.shift_center(self.center_v, [offsetx, offsety, offsetz])
        print(center_v)
        
        vxp = center_v[0] + int(x/2/self.spacing)
        vxm = center_v[0] - int(x/2/self.spacing)
        vyp = center_v[1] + int(y/2/self.spacing)
        vym = center_v[1] - int(y/2/self.spacing)

        if self.ndim == 2:
            print(vxm, vxp, vym, vyp)
            self.barrier_type[[vxp, vxm], vym:vyp+1] = barrier_type
            # self.barrier_type[xm, ym:yp+1] = barrier_type
            self.barrier_type[vxm:vxp+1, [vym, vyp]] = barrier_type
            # self.barrier_type[xm:xp+1, yp] = barrier_type
            if space_type is not None:
                self.special_space_type[vxm+1:vxp, vym+1:vyp] = space_type
        if self.ndim == 3:
            vzp = center_v[2] + int(z/2/self.spacing)
            vzm = center_v[2] - int(z/2/self.spacing)
            self.barrier_type[[vxm, vxp], vym:vyp+1, vzm:vzp+1] = barrier_type
            self.barrier_type[vxm:vxp+1, [vym, vyp], vzm:vzp+1] = barrier_type
            self.barrier_type[vxm:vxp+1, vym:vyp+1, [vzm, vzp]] = barrier_type
            if space_type is not None:
                self.special_space_type[vxm+1:vxp, vym+1:vyp, vzm+1:vzp] = space_type
            
    def add_circle(self, r, barrier_type, space_type=None, offsetx=0, offsety=0, offsetz=0):
        center_v = self.shift_center(self.center_v, [offsetx, offsety, offsetz])
        center_x = self.determine_coord(center_v)
        # Draw a circle
        distance = self.distance_to_center(self.mesh, center_x)
        circle, label, factor = self.find_seperating_boundary(distance, r)
        self.barrier_type[circle == 1] = barrier_type
        print(center_v)
        print(label[0].shape)
        label_in = label[0][tuple(center_v)]
        
        if space_type is not None:
            self.special_space_type[label[0] == label_in] = space_type

    def add_sphere(self, r, barrier_type, offsetx=0, offsety=0, offsetz=0):
        # Same as adding circle
        self.add_circle(r, barrier_type, offsetx=0, offsety=0, offsetz=0)

    def add_rod(self, l, r, barrier_type, space_type=None, direction='x', offsetx=0, offsety=0, offsetz=0):
        assert self.ndim == 3
        temp_barrier = np.zeros_like(self.barrier_type)
        center_v = self.shift_center(self.center_v, [offsetx, offsety, offsetz])
        center_x = self.determine_coord(center_v)
        vxp = center_v[0] + int(l/2/self.spacing)
        vxm = center_v[0] - int(l/2/self.spacing)
        xp = self.determine_coord([vxp, *center_v[1:]])
        xm = self.determine_coord([vxm, *center_v[1:]])
        xm_dist = self.distance_to_center(self.mesh, [xm[0], center_x[1], center_x[2]])
        print(xm_dist.shape)
        tube, label, factor = self.find_seperating_boundary(xm_dist[vxm], r, return_factor=True)
        temp_barrier[vxm:vxp, tube] = barrier_type
        temp_barrier[vxm, xm_dist[vxm] < (r + self.spacing / factor)] = barrier_type
        temp_barrier[vxp, xm_dist[vxm] < (r + self.spacing / factor)] = barrier_type
        label = ndimage.label(1 - temp_barrier>0)
        if label[1] > 1:
            print("Successfully created a rod")
            self.barrier_type[temp_barrier > 0] = barrier_type
        else:
            raise("Failed creating rod")
        label_in = label[0][tuple(center_v)]
        if space_type is not None:
            self.special_space_type[label[0] == label_in] = space_type
            
    def add_ecoli_rod(self, l, r, barrier_type, space_type=None, direction='x', offsetx=0, offsety=0, offsetz=0):
        assert self.ndim == 3
        temp_barrier = np.zeros_like(self.barrier_type)
        # Rod part
        center_v = self.shift_center(self.center_v, [offsetx, offsety, offsetz])
        center_x = self.determine_coord(center_v)
        vxp = center_v[0] + int((l/2-r)/self.spacing)
        vxm = center_v[0] - int((l/2-r)/self.spacing)
        xp = self.determine_coord([vxp, *center_v[1:]])
        xm = self.determine_coord([vxm, *center_v[1:]])
        xm_dist = self.distance_to_center(self.mesh, [xm[0], center_x[1], center_x[2]])
        xp_dist = self.distance_to_center(self.mesh, [xp[0], center_x[1], center_x[2]])
        factor = 5.0
        tube, label, factor = self.find_seperating_boundary(xm_dist[vxm], r)
        temp_barrier[vxm:vxp, tube] = 1
        # Cap part
        cap, label, factor = self.find_seperating_boundary(xm_dist, r)
        temp_barrier[:vxm] = cap[:vxm]
        cap, label, factor = self.find_seperating_boundary(xp_dist, r)
        temp_barrier[vxp:] = cap[vxp:]
        label = ndimage.label(1 - temp_barrier>0)
        if label[1] > 1:
            print("Successfully created a E. coli rod")
            self.barrier_type[temp_barrier > 0] = barrier_type
        label_in = label[0][tuple(center_v)]
        if space_type is not None:
            self.special_space_type[label[0] == label_in] = space_type
        