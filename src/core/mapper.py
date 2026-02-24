import numpy as np

class CoordinateMapper:
    def __init__(self, screen_w, screen_h, margin=0.2):
        """
        Maps normalized camera coordinates (0.0 - 1.0) to screen pixel coordinates.
        
        Args:
            screen_w (int): Total width of the target monitor.
            screen_h (int): Total height of the target monitor.
            margin (float): The 'Virtual Box' margin (e.g., 0.2 creates a box from 20% to 80%).
        """
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.margin = margin

    def map_to_screen(self, x_norm, y_norm):
        """
        Applies the Virtual Box logic to convert normalized camera points to screen pixels.
        
        Args:
            x_norm (float): X from MediaPipe (0.0 to 1.0).
            y_norm (float): Y from MediaPipe (0.0 to 1.0).
            
        Returns:
            tuple: (screen_x, screen_y) as integers.
        """
        # Define the Virtual Box range
        # E.g., if margin is 0.2, range is [0.2, 0.8]
        v_min = self.margin
        v_max = 1.0 - self.margin

        # Linear interpolation: maps [v_min, v_max] -> [0, screen_dimension]
        # np.interp handles values outside the range by clamping to the boundaries
        screen_x = np.interp(x_norm, [v_min, v_max], [0, self.screen_w])
        screen_y = np.interp(y_norm, [v_min, v_max], [0, self.screen_h])

        return int(screen_x), int(screen_y)

    def get_box_corners(self, frame_w, frame_h):
        """
        Calculates pixel coordinates for drawing the Virtual Box on the camera feed.
        """
        x1 = int(self.margin * frame_w)
        y1 = int(self.margin * frame_h)
        x2 = int((1.0 - self.margin) * frame_w)
        y2 = int((1.0 - self.margin) * frame_h)
        return (x1, y1), (x2, y2)
