"""
Simulation Environment.
Minimal environment: white room with red cube and simple audio narration.
"""

import numpy as np


class WhiteRoomEnvironment:
    """
    A 128x128 white room containing a red cube.
    """
    
    def __init__(self):
        self.room_size = (128, 128)
        self.cube_position = np.array([60, 60])
        self.cube_size = 20
        self.cube_color = np.array([1.0, 0.0, 0.0])  # Red
        
        # Audio generation
        # Narrator occasionally says "red" and "cube"
        self.audio_samples = {
            'red': self._generate_tone(440, duration=0.5), # A4
            'cube': self._generate_tone(880, duration=0.5), # A5
            'silence': np.zeros(512)
        }
        
        self.current_audio = self.audio_samples['silence']
        self.audio_ptr = 0
        self.step_counter = 0
        
    def _generate_tone(self, freq, duration, sample_rate=16000):
        t = np.linspace(0, duration, int(sample_rate * duration))
        return np.sin(2 * np.pi * freq * t)
        
    def get_visual_frame(self, gaze_position):
        """
        Generate global view (approximation).
        Ideally, this returns the full environment frame.
        Retina handles cropping.
        """
        frame = np.ones((128, 128, 3))  # White background
        
        # Draw red cube
        x, y = self.cube_position
        cs = self.cube_size
        
        # Ensure cube is within bounds
        x = np.clip(x, 0, 128 - cs)
        y = np.clip(y, 0, 128 - cs)
        
        frame[y:y+cs, x:x+cs] = self.cube_color
        
        return frame
    
    def get_audio_chunk(self, chunk_size=512):
        """Get next audio chunk"""
        # Occasionally play a word
        if self.step_counter % 100 == 20:
            self.current_audio = self.audio_samples['red']
            self.audio_ptr = 0
        elif self.step_counter % 100 == 70:
            self.current_audio = self.audio_samples['cube']
            self.audio_ptr = 0
            
        # Return chunk
        if self.audio_ptr >= len(self.current_audio):
            chunk = np.zeros(chunk_size)
        else:
            end = min(self.audio_ptr + chunk_size, len(self.current_audio))
            chunk = self.current_audio[self.audio_ptr:end]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            self.audio_ptr = end
            
        self.step_counter += 1
        return chunk
