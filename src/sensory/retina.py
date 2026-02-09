"""Event-based foveated retina encoder"""

import numpy as np
from scipy.ndimage import convolve
from .spike_pattern import VisualSpikePattern


class FoveatedRetina:
    """
    Event-based visual encoder with center-surround receptive fields.
    
    Mimics biological retina with:
    - Foveated sampling (log-polar magnification)
    - ON/OFF center-surround cells (Difference of Gaussians)
    - Temporal differencing for event generation
    - Sparse spike output
    """
    
    def __init__(self, resolution=(64, 64), fovea_radius=8, event_threshold=0.15):
        """
        Args:
            resolution: (height, width) of retinal field
            fovea_radius: radius of high-resolution foveal region
            event_threshold: minimum change to generate spike event
        """
        self.resolution = resolution
        self.fovea_radius = fovea_radius
        self.event_threshold = event_threshold
        
        # Create center-surround filters (Difference of Gaussians)
        self.on_kernel = self._create_dog_kernel(sigma_center=1.0, sigma_surround=3.0, sign=1)
        self.off_kernel = self._create_dog_kernel(sigma_center=1.0, sigma_surround=3.0, sign=-1)
        
        # Foveal magnification map (log-polar sampling)
        self.magnification_map = self._build_magnification_map()
        
        # Temporal memory for event detection
        self.prev_frame = None
        
    def _create_dog_kernel(self, sigma_center, sigma_surround, sign, kernel_size=7):
        """Create Difference of Gaussians kernel for center-surround"""
        ax = np.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
        xx, yy = np.meshgrid(ax, ax)
        
        # Center Gaussian
        center = np.exp(-(xx**2 + yy**2) / (2 * sigma_center**2))
        center /= (2 * np.pi * sigma_center**2)
        
        # Surround Gaussian
        surround = np.exp(-(xx**2 + yy**2) / (2 * sigma_surround**2))
        surround /= (2 * np.pi * sigma_surround**2)
        
        # Difference of Gaussians
        dog = sign * (center - surround)
        
        return dog
    
    def _build_magnification_map(self):
        """Build foveal magnification factor map (higher resolution at center)"""
        h, w = self.resolution
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        
        # Distance from center
        distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Log-polar magnification (1.0 at fovea, decreases with distance)
        magnification = 1.0 / (1.0 + distance / self.fovea_radius)
        
        return magnification
    
    def _apply_foveal_sampling(self, frame, gaze_position):
        """Apply foveal magnification centered on gaze position"""
        h, w = self.resolution
        gaze_y, gaze_x = gaze_position
        
        # Shift frame so gaze is at center
        shift_y = h // 2 - int(gaze_y)
        shift_x = w // 2 - int(gaze_x)
        
        # Circular shift (wrap around)
        shifted = np.roll(frame, (shift_y, shift_x), axis=(0, 1))
        
        # Apply magnification (simple version: just weight by magnification)
        # In full implementation, this would resample with varying resolution
        foveated = shifted * self.magnification_map[:, :, np.newaxis]
        
        return foveated
    
    def encode(self, frame, gaze_position):
        """
        Convert visual frame to sparse ON/OFF spike events.
        
        Args:
            frame: RGB image array (H, W, 3) with values in [0, 1]
            gaze_position: (y, x) coordinates of current gaze fixation
            
        Returns:
            VisualSpikePattern with ON and OFF spike arrays
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
        else:
            gray = frame
            
        # Resize to retinal resolution if needed
        if gray.shape != self.resolution:
            from scipy.ndimage import zoom
            scale_y = self.resolution[0] / gray.shape[0]
            scale_x = self.resolution[1] / gray.shape[1]
            gray = zoom(gray, (scale_y, scale_x), order=1)
        
        # Apply foveal sampling
        foveated = self._apply_foveal_sampling(gray[:, :, np.newaxis], gaze_position)[:, :, 0]
        
        # Temporal differencing for event generation
        if self.prev_frame is not None:
            temporal_diff = foveated - self.prev_frame
            
            # Generate ON events (brightness increase)
            on_events = np.maximum(temporal_diff, 0)
            on_events = (on_events > self.event_threshold).astype(np.float32)
            
            # Generate OFF events (brightness decrease)
            off_events = np.maximum(-temporal_diff, 0)
            off_events = (off_events > self.event_threshold).astype(np.float32)
        else:
            # First frame: generate events from absolute intensity
            on_events = (foveated > 0.5).astype(np.float32)
            off_events = (foveated < 0.5).astype(np.float32)
        
        # Store for next iteration
        self.prev_frame = foveated.copy()
        
        # Apply center-surround filtering
        on_spikes = convolve(on_events, self.on_kernel, mode='constant')
        off_spikes = convolve(off_events, self.off_kernel, mode='constant')
        
        # Rectify (only positive responses)
        on_spikes = np.maximum(on_spikes, 0)
        off_spikes = np.maximum(off_spikes, 0)
        
        # Threshold to create sparse spikes
        on_spikes = (on_spikes > 0.1).astype(np.float32)
        off_spikes = (off_spikes > 0.1).astype(np.float32)
        
        return VisualSpikePattern(on=on_spikes, off=off_spikes)
    
    def reset(self):
        """Reset temporal memory"""
        self.prev_frame = None
