import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Circle
from environment import draw_map, WALLS
from pynput import keyboard
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

DT = 0.1
SIM_TIME = 1000.0
KEYBOARD = {"x": 0, "y": 0}

MIN_X = 0.0
MAX_X = 40.0
MIN_Y = 0.0
MAX_Y = 19.0
GRID_RESOLUTION = 1.5
MOTION_STD = 1.0  

LIDAR_MAX_RANGE = 10.0   # Maximum detection range
LIDAR_NUM_RAYS = 10      # Number of rays (every 10 degrees)
NOISE_RANGE = 0.3        # Sensor noise
RANGE_STD = 2.0          # Standard deviation for observation likelihood

def line_intersection(ray_origin, ray_dir, wall):
    x1, y1, x2, y2 = wall
    rx, ry = ray_origin
    dx, dy = ray_dir
    
    wx = x2 - x1
    wy = y2 - y1
    
    denom = dx * wy - dy * wx
    if abs(denom) < 1e-10:  
        return None
    
    t = ((x1 - rx) * wy - (y1 - ry) * wx) / denom  
    s = ((x1 - rx) * dy - (y1 - ry) * dx) / denom  
    if t > 0 and 0 <= s <= 1:
        return t
    return None


def cast_ray(robot_x, robot_y, angle):
    dx = np.cos(angle)
    dy = np.sin(angle)
    
    min_dist = LIDAR_MAX_RANGE
    
    for wall in WALLS:
        dist = line_intersection((robot_x, robot_y), (dx, dy), wall)
        if dist is not None and dist < min_dist:
            min_dist = dist
    hit_x = robot_x + min_dist * dx
    hit_y = robot_y + min_dist * dy
    
    return min_dist, hit_x, hit_y


class Histogram:
    def __init__(self):
        self.robot_x = 1
        self.robot_y = 10
        self.robot_theta = 0

        self.data = []
        self.x_w = None
        self.y_w = None
        self.dx = 0.0 
        self.dy = 0.0

    def init_grid_map(self):
        self.x_w = int(round((MAX_X - MIN_X) / GRID_RESOLUTION))
        self.y_w = int(round((MAX_Y - MIN_Y) / GRID_RESOLUTION))
        self.data = np.zeros((self.x_w, self.y_w))
        
        # Convert world coordinates to grid cell index
        start_ix = int((self.robot_x - MIN_X) / GRID_RESOLUTION)
        start_iy = int((self.robot_y - MIN_Y) / GRID_RESOLUTION)
        
        # 100% probability at starting position
        self.data[start_ix, start_iy] = 1.0
        
    def normalize_probability(self):
        self.data = np.array(self.data)
        sump = np.sum(self.data)
        self.data = self.data / sump

    def calc_grid_index(self):
        mx, my = np.mgrid[slice(MIN_X - GRID_RESOLUTION / 2.0,
                                MAX_X + GRID_RESOLUTION / 2.0,
                                GRID_RESOLUTION),
                          slice(MIN_Y- GRID_RESOLUTION / 2.0,
                                MAX_Y + GRID_RESOLUTION / 2.0,
                                GRID_RESOLUTION)]
        return mx, my

    def draw_heat_map(self, ax, mx, my):
        max_value = np.max(self.data)
        ax.pcolor(mx, my, self.data, vmax=max_value, cmap=mpl.colormaps["Blues"])
        ax.set_xlim(MIN_X, MAX_X)
        ax.set_ylim(MIN_Y, MAX_Y)
        ax.set_aspect("equal")

    def lidar_scan(self):
        observations = []
        for i in range(LIDAR_NUM_RAYS):
            angle = self.robot_theta + (2 * np.pi * i / LIDAR_NUM_RAYS)
            dist, hit_x, hit_y = cast_ray(self.robot_x, self.robot_y, angle)
            
            noisy_dist = dist + np.random.randn() * NOISE_RANGE
            noisy_dist = max(0.1, noisy_dist)  
            
            observations.append([noisy_dist, hit_x, hit_y])
        
        return np.array(observations)

    def draw_lidar(self, ax, observations):
        for obs in observations:
            dist, hit_x, hit_y = obs
            ax.plot([self.robot_x, hit_x], [self.robot_y, hit_y], 'g-', alpha=0.3, linewidth=0.5)
            ax.plot(hit_x, hit_y, 'go', markersize=2)

    def map_shift(self, x_shift, y_shift):
        tmp_data = copy.deepcopy(self.data)
        for ix in range(self.x_w):
            for iy in range(self.y_w):
                self.data[ix][iy] = 0.0001
        for ix in range(self.x_w):
            for iy in range(self.y_w):
                nix = ix + x_shift
                niy = iy + y_shift
                if 0 <= nix < self.x_w and 0 <= niy < self.y_w:
                    self.data[nix][niy] = tmp_data[ix][iy]

    def histogram_motion_update(self, u):
        self.dx += DT * np.cos(self.robot_theta) * u[0, 0]
        self.dy += DT * np.sin(self.robot_theta) * u[0, 0]
        
        x_shift = int(self.dx // GRID_RESOLUTION)
        y_shift = int(self.dy // GRID_RESOLUTION)
        if abs(x_shift) >= 1 or abs(y_shift) >= 1:
            self.map_shift(x_shift, y_shift)
            self.dx -= x_shift * GRID_RESOLUTION
            self.dy -= y_shift * GRID_RESOLUTION
        self.data = gaussian_filter(self.data, sigma=MOTION_STD)
        self.normalize_probability()

    def observation_update(self, z):
        for iz in range(0, len(z)):
            measured_dist = z[iz, 0]  
            
            if measured_dist >= LIDAR_MAX_RANGE:
                continue
            
            ray_angle = self.robot_theta + (2 * np.pi * iz / LIDAR_NUM_RAYS)
            for ix in range(self.x_w):
                for iy in range(self.y_w):
                    cell_x = ix * GRID_RESOLUTION + MIN_X
                    cell_y = iy * GRID_RESOLUTION + MIN_Y
                    expected_dist, _, _ = cast_ray(cell_x, cell_y, ray_angle)
                    diff = measured_dist - expected_dist
                    likelihood = norm.pdf(diff, 0.0, RANGE_STD)
                    
                    self.data[ix, iy] *= likelihood
        
        self.normalize_probability()

    def calc_control_input(self):
        v = 0.5 * KEYBOARD["y"]
        yaw_rate = KEYBOARD["x"]
        u = np.array([v, yaw_rate]).reshape(2, 1)
        return u
    
    def calc_orientation(self):
        dx = 1.0 * np.cos(self.robot_theta) 
        dy = 1.0 * np.sin(self.robot_theta)
        y = np.array([dx, dy]).reshape(2, 1)
        return y
    
    def motion_update(self, u):
        self.robot_x += DT * np.cos(self.robot_theta) * u[0, 0]  
        self.robot_y += DT * np.sin(self.robot_theta) * u[0, 0] 
        self.robot_theta = self.robot_theta + u[1, 0] 
    
    def keyboard_listener(self, key):
        global KEYBOARD
        try:
            if key == keyboard.Key.up:
                KEYBOARD["y"] += 0.5
            elif key == keyboard.Key.down:
                KEYBOARD["y"] -= 0.5   
            elif key == keyboard.Key.left:
                KEYBOARD["x"] += 0.01
            elif key == keyboard.Key.right:
                KEYBOARD["x"] -= 0.01
        except AttributeError:
            pass

    def main(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        self.init_grid_map()
        mx, my = self.calc_grid_index()
        
        listener = keyboard.Listener(on_press=self.keyboard_listener)
        listener.start()

        time = 0.1
        while SIM_TIME >= time:
            time += DT 
            u = self.calc_control_input()
            y = self.calc_orientation()
            self.motion_update(u)
            
            lidar_observations = self.lidar_scan()
            
            self.histogram_motion_update(u)
            
            self.observation_update(lidar_observations)
            
            ax1.cla()
            ax2.cla()
            
            ax1.set_xlim(0, MAX_X)
            ax1.set_ylim(0, MAX_Y)
            ax1.set_aspect("equal")
            draw_map(ax1)
            
            self.draw_lidar(ax1, lidar_observations)
            
            robot_body = Circle((self.robot_x, self.robot_y), radius=0.5, facecolor="brown")
            ax1.arrow(self.robot_x, self.robot_y, y[0, 0], y[1, 0], head_width=0.2, head_length=0.1, fc="orange", ec="pink")
            ax1.add_patch(robot_body)
            ax1.set_title("Real Map (Ground Truth) + LiDAR")
            
            self.draw_heat_map(ax2, mx, my)
            ax2.set_title("Probability Grid Map")
            
            plt.pause(0.01)
        
        plt.show()
    

if __name__ == "__main__":
    ekf = Histogram()
    ekf.main()