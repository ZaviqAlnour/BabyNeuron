"""Cochlear filter bank for audio encoding"""

import numpy as np
from scipy.signal import gammatone
from .spike_pattern import AudioSpikePattern


class CochlearEncoder:
    """
    Spectro-temporal spike encoder mimicking basilar membrane.
    
    Uses gammatone filterbank to decompose audio into frequency channels,
    then generates spikes via stochastic threshold crossing with adaptation.
    """
    
    def __init__(self, n_channels=32, sample_rate=16000, chunk_size=512):
        """
        Args:
            n_channels: number of frequency channels (cochlear filters)
            sample_rate: audio sampling rate in Hz
            chunk_size: number of samples per processing chunk
        """
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        
        # Gammatone filterbank (log-spaced frequencies from 50 Hz to 8 kHz)
        self.center_freqs = np.logspace(np.log10(50), np.log10(8000), n_channels)
        
        # Build filterbank
        self.filters = self._build_gammatone_filterbank()
        
        # Compression exponent (mimics cochlear compression)
        self.compression_exp = 0.3
        
        # Firing rate adaptation per channel
        self.adaptation = np.ones(n_channels)
        self.adaptation_rate = 0.01
        
        # Filter states for continuous processing
        self.filter_states = [np.zeros(4) for _ in range(n_channels)]
        
    def _build_gammatone_filterbank(self):
        """Create gammatone filters for each frequency channel"""
        filters = []
        
        for freq in self.center_freqs:
            # Gammatone filter coefficients (4th order)
            # Using simplified implementation
            b, a = self._gammatone_coefficients(freq, self.sample_rate)
            filters.append((b, a))
            
        return filters
    
    def _gammatone_coefficients(self, center_freq, fs, order=4):
        """
        Generate gammatone filter coefficients.
        Simplified version using equivalent bandwidth.
        """
        # Equivalent rectangular bandwidth (ERB)
        erb = 24.7 * (4.37 * center_freq / 1000 + 1)
        
        # Bandwidth parameter
        b_param = 1.019 * erb
        
        # Normalized frequency
        omega = 2 * np.pi * center_freq / fs
        
        # Filter coefficients (simplified IIR approximation)
        # In production, use scipy.signal.gammatone or more accurate implementation
        theta = omega
        phi = 2 * np.pi * b_param / fs
        
        # Poles
        alpha = -np.exp(-phi)
        
        # Numerator (feedforward)
        b = np.array([1.0, 0, 0, 0, 0])
        
        # Denominator (feedback) - 4th order
        a = np.array([1.0, -4*alpha*np.cos(theta), 
                      6*alpha**2, 
                      -4*alpha**3*np.cos(theta), 
                      alpha**4])
        
        return b, a

    
    def _apply_filterbank(self, audio_chunk):
        """Apply gammatone filterbank to audio chunk"""
        from scipy.signal import lfilter
        
        filtered_outputs = np.zeros((self.n_channels, len(audio_chunk)))
        
        for i, (b, a) in enumerate(self.filters):
            # Apply filter with state
            filtered, self.filter_states[i] = lfilter(
                b, a, audio_chunk, zi=self.filter_states[i]
            )
            filtered_outputs[i] = filtered
            
        return filtered_outputs
    
    def encode(self, audio_chunk):
        """
        Convert audio chunk to tonotopic spike pattern.
        
        Args:
            audio_chunk: 1D array of audio samples (length = chunk_size)
            
        Returns:
            AudioSpikePattern with spike array (n_channels, n_time_bins)
        """
        # Ensure correct chunk size
        if len(audio_chunk) < self.chunk_size:
            # Pad with zeros
            audio_chunk = np.pad(audio_chunk, (0, self.chunk_size - len(audio_chunk)))
        elif len(audio_chunk) > self.chunk_size:
            # Truncate
            audio_chunk = audio_chunk[:self.chunk_size]
        
        # Apply gammatone filterbank
        filtered = self._apply_filterbank(audio_chunk)
        
        # Half-wave rectification
        rectified = np.maximum(filtered, 0)
        
        # Cochlear compression (power law)
        compressed = rectified ** self.compression_exp
        
        # Temporal integration (moving average for spike rate)
        # Downsample to create time bins
        n_time_bins = 16  # Reduce temporal resolution
        bin_size = self.chunk_size // n_time_bins
        
        binned = np.zeros((self.n_channels, n_time_bins))
        for t in range(n_time_bins):
            start = t * bin_size
            end = start + bin_size
            binned[:, t] = compressed[:, start:end].mean(axis=1)
        
        # Generate spikes via stochastic threshold crossing with adaptation
        spike_probs = binned / (binned + self.adaptation[:, np.newaxis])
        spikes = (np.random.random(spike_probs.shape) < spike_probs).astype(np.float32)
        
        # Update adaptation (increases with activity, decays otherwise)
        mean_activity = binned.mean(axis=1)
        self.adaptation = (1 - self.adaptation_rate) * self.adaptation + \
                         self.adaptation_rate * (1 + mean_activity)
        
        return AudioSpikePattern(channels=spikes)
    
    def reset(self):
        """Reset filter states and adaptation"""
        self.filter_states = [np.zeros(4) for _ in range(self.n_channels)]
        self.adaptation = np.ones(self.n_channels)
