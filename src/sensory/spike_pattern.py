"""Data structures for spike patterns"""

import time
from dataclasses import dataclass
import numpy as np


@dataclass
class SpikePattern:
    """Container for spike train data with timestamp"""
    data: np.ndarray = None
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:

            self.timestamp = time.time()
    
    @property
    def spike_count(self):
        """Total number of spikes in pattern"""
        return np.sum(self.data > 0)
    
    @property
    def spike_rate(self):
        """Average firing rate"""
        return np.mean(self.data)


@dataclass
class VisualSpikePattern(SpikePattern):
    """Spike pattern from visual encoder with ON/OFF channels"""
    on: np.ndarray = None
    off: np.ndarray = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.data is None and self.on is not None:
            # Combine ON and OFF into single array
            self.data = np.concatenate([self.on.flatten(), self.off.flatten()])


@dataclass
class AudioSpikePattern(SpikePattern):
    """Spike pattern from cochlear encoder with frequency channels"""
    channels: np.ndarray = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.data is None and self.channels is not None:
            self.data = self.channels.flatten()
