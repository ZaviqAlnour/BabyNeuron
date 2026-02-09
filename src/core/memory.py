"""
Associative Memory Module.
Implements a Hopfield-like attractor network for binding multimodal patterns
(e.g., visual object + audio word).
"""

import numpy as np


class AssociativeMemory:
    """
    Auto-associative memory for storing and recalling combined sensory patterns.
    """
    
    def __init__(self, n_units):
        """
        Args:
            n_units: Total number of neurons in memory layer
        """
        self.n_units = n_units
        
        # Weight matrix (initialized to zeros)
        # In Hopfield nets, diagonals are zero.
        self.weights = np.zeros((n_units, n_units))
        
        self.patterns_stored = 0
        
    def store_pattern(self, visual_pattern, audio_pattern, strength=1.0):
        """
        Bind visual and audio patterns via Hebbian learning.
        
        Args:
            visual_pattern: 1D array of visual features
            audio_pattern: 1D array of audio features
            strength: Learning rate/emphasis for this pattern
        """
        # 1. Combine modalities
        # Pad or truncate if needed to fit n_units (assuming n_units split 50/50 or adjust dynamically)
        # For simplicity, let's assume visual and audio occupy distinct parts of 'state'
        # Or we just concat and clip.
        
        combined = np.concatenate([visual_pattern, audio_pattern])
        
        # Ensure correct size
        if len(combined) > self.n_units:
            combined = combined[:self.n_units]
        elif len(combined) < self.n_units:
            combined = np.pad(combined, (0, self.n_units - len(combined)))
            
        # 2. Binarize (Hopfield nets work best with bipolar -1 / +1)
        # Threshold at mean
        state = np.where(combined > combined.mean(), 1.0, -1.0)
        
        # 3. Hebbian update (Outer product)
        # dW = strength * (state * state.T) / N
        dW = strength * np.outer(state, state) / self.n_units
        
        # Zero diagonal (no self-connections)
        np.fill_diagonal(dW, 0)
        
        # Update weights
        self.weights += dW
        self.patterns_stored += 1
        
    def recall(self, partial_pattern, n_iterations=10):
        """
        Recover full pattern from partial cue via attractor dynamics.
        
        Args:
            partial_pattern: Input state (e.g. valid visual + noise/zeros audio)
            
        Returns:
            np.ndarray: Completed pattern state
        """
        # Prepare input
        # Pad if needed
        if len(partial_pattern) < self.n_units:
             state = np.pad(partial_pattern, (0, self.n_units - len(partial_pattern)))
        else:
            state = partial_pattern[:self.n_units].copy()
            
        # Binarize input cue if not already
        state = np.where(state > state.mean(), 1.0, -1.0)
        
        # Asynchronous updates or synchronous? 
        # Synchronous is easier for vectorization.
        for _ in range(n_iterations):
            # h = W * s
            in_current = self.weights @ state
            
            # Update state: s = sign(h)
            state = np.sign(in_current)
            # Handle zeros (keep previous state or map to 1?)
            state[state == 0] = 1.0 
            
        return state
        
    def get_stability(self, pattern):
        """Check if pattern is stable (energy minimum)"""
        # Energy E = -0.5 * s.T * W * s
        # Just check if recall changes it.
        recalled = self.recall(pattern, n_iterations=1)
        
        # Binarize input for comparison
        # (Assuming input pattern was continuous, we need to compare binarized versions)
        # If input is already binarized:
        bin_pattern = np.where(pattern > pattern.mean(), 1.0, -1.0)
        
        overlap = np.mean(bin_pattern == recalled)
        return overlap # 1.0 means perfectly stable
