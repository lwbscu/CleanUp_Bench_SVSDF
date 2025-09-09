#!/usr/bin/env python3
"""
Dilation Radius Visualization Tool
Real-time display of different dilation radius effects on path planning
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
import cv2
from scipy import ndimage
import sys
import os
import warnings

# Configure matplotlib to avoid font warnings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

# Add MapEx path
sys.path.append('/home/lwb/MapEx/scripts/')
try:
    import sim_utils
    import pyastar2d
    MAPEX_AVAILABLE = True
except:
    MAPEX_AVAILABLE = False
    # Suppress print output for missing modules

class DilationVisualizer:
    def __init__(self, map_data=None, pixel_per_meter=20):
        self.pixel_per_meter = pixel_per_meter
        self.current_dilation = 15
        
        # Create sample map if no map data provided
        if map_data is None:
            self.original_map = self._create_sample_map()
        else:
            self.original_map = map_data.copy()
        
        self.dilated_map = None
        self.planning_map = None
        
        # Create GUI interface
        self.setup_visualization()
        
    def _create_sample_map(self):
        """Create sample map for testing"""
        map_size = 200
        obs_map = np.full((map_size, map_size), 0.0, dtype=np.float32)  # Default free space
        
        # Add some obstacles
        # Walls
        obs_map[50:60, 20:180] = 1.0
        obs_map[140:150, 20:180] = 1.0
        obs_map[20:180, 50:60] = 1.0
        obs_map[20:180, 140:150] = 1.0
        
        # Internal obstacles
        obs_map[80:120, 80:120] = 1.0  # Center square
        obs_map[30:40, 100:160] = 1.0  # Horizontal bar
        obs_map[100:160, 30:40] = 1.0  # Vertical bar
        
        # Add some unknown regions
        obs_map[70:90, 160:180] = 0.5
        obs_map[160:180, 70:90] = 0.5
        
        return obs_map
        
    def setup_visualization(self):
        """Setup visualization interface"""
        # Create main window
        self.fig = plt.figure(figsize=(16, 12))
        self.fig.suptitle('Dilation Radius Visualization Tool - Path Planning Parameter Debugging', fontsize=16)
        
        # Subplot layout
        gs = self.fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.3], hspace=0.3, wspace=0.3)
        
        # Original map
        self.ax_original = self.fig.add_subplot(gs[0, 0])
        self.ax_original.set_title('Original Map')
        
        # Dilated map
        self.ax_dilated = self.fig.add_subplot(gs[0, 1])
        self.ax_dilated.set_title('Dilated Map')
        
        # Comparison plot
        self.ax_comparison = self.fig.add_subplot(gs[0, 2])
        self.ax_comparison.set_title('Dilation Comparison')
        
        # Path planning results
        self.ax_path = self.fig.add_subplot(gs[1, :])
        self.ax_path.set_title('Path Planning Results (Green=Feasible, Red=Infeasible)')
        
        # Control panel
        self.ax_controls = self.fig.add_subplot(gs[2, :])
        self.ax_controls.axis('off')
        
        # Slider controls
        self.setup_controls()
        
        # Initial drawing
        self.update_visualization()
        
    def setup_controls(self):
        """Setup control sliders"""
        # Dilation radius slider
        ax_dilation = plt.axes([0.15, 0.15, 0.3, 0.03])
        self.slider_dilation = Slider(ax_dilation, 'Dilation Radius (pixels)', 1, 50, valinit=self.current_dilation, valfmt='%d')
        self.slider_dilation.on_changed(self.on_dilation_change)
        
        # Pixel density slider
        ax_pixel_meter = plt.axes([0.55, 0.15, 0.3, 0.03])
        self.slider_pixel_meter = Slider(ax_pixel_meter, 'Pixels/Meter', 5, 50, valinit=self.pixel_per_meter, valfmt='%d')
        self.slider_pixel_meter.on_changed(self.on_pixel_meter_change)
        
        # Reset button
        ax_reset = plt.axes([0.15, 0.05, 0.1, 0.04])
        self.button_reset = Button(ax_reset, 'Reset')
        self.button_reset.on_clicked(self.reset_params)
        
        # Save button
        ax_save = plt.axes([0.3, 0.05, 0.1, 0.04])
        self.button_save = Button(ax_save, 'Save Image')
        self.button_save.on_clicked(self.save_visualization)
        
        # Test path button
        ax_test = plt.axes([0.45, 0.05, 0.1, 0.04])
        self.button_test = Button(ax_test, 'Test Path')
        self.button_test.on_clicked(self.test_path_planning)
        
    def on_dilation_change(self, val):
        """Dilation radius change callback"""
        self.current_dilation = int(val)
        self.update_visualization()
        
    def on_pixel_meter_change(self, val):
        """Pixel density change callback"""
        self.pixel_per_meter = int(val)
        self.update_visualization()
        
    def reset_params(self, event):
        """Reset parameters"""
        self.slider_dilation.reset()
        self.slider_pixel_meter.reset()
        self.current_dilation = 15
        self.pixel_per_meter = 20
        self.update_visualization()
        
    def save_visualization(self, event):
        """Save visualization image"""
        timestamp = int(time.time())
        filename = f'dilation_visualization_{timestamp}.png'
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')
        
    def create_dilated_map(self, dilation_radius):
        """Create dilated map"""
        # Find obstacles
        obstacle_mask = (self.original_map >= 0.8)
        
        if dilation_radius <= 0:
            return self.original_map.copy()
        
        # Create dilation kernel
        kernel_size = int(dilation_radius * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        
        # Perform dilation
        dilated_obstacles = cv2.dilate(obstacle_mask.astype(np.uint8), kernel, iterations=1)
        
        # Create dilated map
        dilated_map = self.original_map.copy()
        dilated_map[dilated_obstacles.astype(bool)] = 1.0
        
        return dilated_map
    
    def create_planning_map(self, dilated_map):
        """Create map for path planning (A* format)"""
        planning_map = np.zeros_like(dilated_map)
        
        # Set obstacles to infinity
        planning_map[dilated_map >= 0.8] = np.inf
        # Set unknown regions to infinity (safer)
        planning_map[dilated_map == 0.5] = np.inf
        # Set free space to 1
        planning_map[dilated_map < 0.8] = 1.0
        planning_map[dilated_map == 0.5] = np.inf  # Reset unknown regions
        
        return planning_map
        
    def update_visualization(self):
        """Update visualization"""
        # Create dilated map
        self.dilated_map = self.create_dilated_map(self.current_dilation)
        self.planning_map = self.create_planning_map(self.dilated_map)
        
        # Clear all subplots
        for ax in [self.ax_original, self.ax_dilated, self.ax_comparison, self.ax_path]:
            ax.clear()
        
        # Draw original map
        self.ax_original.imshow(self.original_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        self.ax_original.set_title('Original Map')
        self._add_colorbar_text(self.ax_original)
        
        # Draw dilated map
        self.ax_dilated.imshow(self.dilated_map, cmap='RdYlBu_r', vmin=0, vmax=1)
        self.ax_dilated.set_title(f'Dilated Map (Radius={self.current_dilation}px)')
        self._add_colorbar_text(self.ax_dilated)
        
        # Draw comparison plot
        self._draw_comparison()
        
        # Draw path planning plot
        self._draw_path_planning()
        
        # Update info text
        self._update_info_text()
        
        plt.draw()
        
    def _add_colorbar_text(self, ax):
        """Add color legend"""
        ax.text(0.02, 0.98, 'Blue=Free\nYellow=Unknown\nRed=Obstacle', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    def _draw_comparison(self):
        """Draw dilation comparison plot"""
        # Create RGB image to show differences
        h, w = self.original_map.shape
        comparison_img = np.zeros((h, w, 3))
        
        # Original obstacles - red
        original_obstacles = (self.original_map >= 0.8)
        comparison_img[original_obstacles] = [1, 0, 0]
        
        # Dilated added regions - orange
        dilated_obstacles = (self.dilated_map >= 0.8)
        new_obstacles = dilated_obstacles & ~original_obstacles
        comparison_img[new_obstacles] = [1, 0.5, 0]
        
        # Free space - light blue
        free_space = (self.original_map < 0.3)
        comparison_img[free_space] = [0.7, 0.9, 1]
        
        # Unknown space - gray
        unknown_space = (self.original_map == 0.5)
        comparison_img[unknown_space] = [0.5, 0.5, 0.5]
        
        self.ax_comparison.imshow(comparison_img)
        self.ax_comparison.set_title(f'Dilation Comparison (Orange=New Obstacles)')
        
        # Add legend
        self.ax_comparison.text(0.02, 0.98, 
                               'Red=Original Obstacles\nOrange=Dilated Area\nBlue=Free Space\nGray=Unknown Area', 
                               transform=self.ax_comparison.transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
    def _draw_path_planning(self):
        """Draw path planning results"""
        # Display planning map
        planning_display = np.copy(self.planning_map)
        planning_display[planning_display == np.inf] = -1  # Use -1 for impassable areas
        
        self.ax_path.imshow(planning_display, cmap='RdYlGn', vmin=-1, vmax=1)
        
        # Test several paths
        self._test_sample_paths()
        
        self.ax_path.set_title(f'Path Planning Map (Dilation Radius={self.current_dilation}px, {self.current_dilation/self.pixel_per_meter:.2f}m)')
        
    def _test_sample_paths(self):
        """Test several sample paths"""
        h, w = self.planning_map.shape
        
        # Define several test paths
        test_cases = [
            ((10, 10), (h-10, w-10)),  # Diagonal
            ((10, w//2), (h-10, w//2)),  # Vertical
            ((h//2, 10), (h//2, w-10)),  # Horizontal
            ((20, 20), (h//3, w//3)),   # Short distance
        ]
        
        for i, (start, goal) in enumerate(test_cases):
            try:
                if MAPEX_AVAILABLE:
                    path = pyastar2d.astar_path(self.planning_map, start, goal, allow_diagonal=False)
                else:
                    path = self._simple_astar(start, goal)
                
                if path is not None and len(path) > 1:
                    # Draw successful path - green
                    self.ax_path.plot(path[:, 1], path[:, 0], 'g-', linewidth=2, alpha=0.7)
                    self.ax_path.plot(start[1], start[0], 'go', markersize=8)
                    self.ax_path.plot(goal[1], goal[0], 'g^', markersize=8)
                else:
                    # Draw failed path - red
                    self.ax_path.plot([start[1], goal[1]], [start[0], goal[0]], 'r--', linewidth=2, alpha=0.7)
                    self.ax_path.plot(start[1], start[0], 'ro', markersize=8)
                    self.ax_path.plot(goal[1], goal[0], 'r^', markersize=8)
                    
            except Exception:
                # Suppress error output for missing modules
                pass
                
    def _simple_astar(self, start, goal):
        """Simplified A* algorithm (when pyastar2d is unavailable)"""
        # Simple reachability check
        h, w = self.planning_map.shape
        if (self.planning_map[start] == np.inf or 
            self.planning_map[goal] == np.inf or
            start[0] < 0 or start[0] >= h or start[1] < 0 or start[1] >= w or
            goal[0] < 0 or goal[0] >= h or goal[1] < 0 or goal[1] >= w):
            return None
        
        # Simple straight line path check
        from scipy.interpolate import interp1d
        
        num_points = max(abs(goal[0] - start[0]), abs(goal[1] - start[1])) * 2
        if num_points < 2:
            return np.array([start, goal])
            
        x_coords = np.linspace(start[0], goal[0], num_points).astype(int)
        y_coords = np.linspace(start[1], goal[1], num_points).astype(int)
        
        # Check for obstacles along the path
        for x, y in zip(x_coords, y_coords):
            if 0 <= x < h and 0 <= y < w:
                if self.planning_map[x, y] == np.inf:
                    return None
        
        return np.column_stack([x_coords, y_coords])
        
    def test_path_planning(self, event):
        """Test path planning button callback"""
        # Redraw path planning plot
        self.ax_path.clear()
        self._draw_path_planning()
        plt.draw()
        
    def _update_info_text(self):
        """Update info text"""
        info_text = (f"Current Settings:\n"
                    f"Dilation Radius: {self.current_dilation} pixels ({self.current_dilation/self.pixel_per_meter:.2f} meters)\n"
                    f"Pixel Density: {self.pixel_per_meter} pixels/meter\n"
                    f"Map Size: {self.original_map.shape}")
        
        # Display info on control panel
        self.ax_controls.text(0.02, 0.5, info_text, transform=self.ax_controls.transAxes,
                             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))

def load_slam_map(file_path):
    """Load SLAM map from file"""
    try:
        if file_path.endswith('.npy'):
            return np.load(file_path)
        elif file_path.endswith('.pgm'):
            return cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) / 255.0
        else:
            return None
    except Exception:
        return None

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Dilation Radius Visualization Tool')
    parser.add_argument('--map_file', type=str, help='SLAM map file path (.npy or .pgm)')
    parser.add_argument('--pixel_per_meter', type=int, default=20, help='Pixel density (pixels/meter)')
    parser.add_argument('--initial_dilation', type=int, default=15, help='Initial dilation radius')
    
    args = parser.parse_args()
    
    # Load map data
    map_data = None
    if args.map_file:
        map_data = load_slam_map(args.map_file)
    
    # Create visualizer
    visualizer = DilationVisualizer(map_data, args.pixel_per_meter)
    visualizer.current_dilation = args.initial_dilation
    
    plt.show()

if __name__ == '__main__':
    import time
    main()