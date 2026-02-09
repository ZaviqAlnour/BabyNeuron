"""
Spiking Neural Network with STDP learning rule.
Implements a multi-layer SNN with Leaky Integrate-and-Fire (LIF) neurons
and Spike-Timing-Dependent Plasticity (STDP).
"""

import numpy as np


class SpikingNeuralNetwork:
    """
    Multi-layer SNN with STDP as sole learning rule.
    
    Attributes:
        layer_sizes (list): Number of neurons in each layer
        tau_mem (float): Membrane time constant (ms)
        tau_syn (float): Synaptic trace time constant (ms)
        learning_rate (float): Base learning rate for STDP
    """
    
    def __init__(self, layer_sizes, tau_mem=20.0, tau_syn=5.0, learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.learning_rate = learning_rate
        
        # Initialize weights (random sparse connectivity)
        # Weights are stored as a list of matrices [W1, W2, ...]
        self.weights = []
        for i in range(len(layer_sizes) - 1):
            # Glorot-like initialization scaled for SNN
            scale = 1.0 / np.sqrt(layer_sizes[i])
            W = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            # Ensure postive/negative balance but predominantly excitatory start? 
            # For now, allow both, but usually Dale's law applies. 
            # We'll allow signed weights for simplicity of "inhibitory" effects without separate populations.
            self.weights.append(W)
            
        # Neuron state variables
        # Potentials: [layer1_v, layer2_v, ...]
        self.membrane_potentials = [np.zeros(s) for s in layer_sizes]
        
        # Spike traces for STDP: [layer1_trace, layer2_trace, ...]
        self.spike_traces = [np.zeros(s) for s in layer_sizes]
        
        # STDP parameters
        self.A_plus = 0.01   # LTP amplitude
        self.A_minus = 0.012 # LTD amplitude (slightly larger for stability)
        self.tau_plus = 20.0 # LTP time constant (ms)
        self.tau_minus = 20.0 # LTD time constant (ms)
        
        # Refractory period
        self.refractory_time = 5.0 # ms
        self.refractory_counters = [np.zeros(s) for s in layer_sizes]
        
    def reset(self):
        """Reset all state variables"""
        self.membrane_potentials = [np.zeros(s) for s in self.layer_sizes]
        self.spike_traces = [np.zeros(s) for s in self.layer_sizes]
        self.refractory_counters = [np.zeros(s) for s in self.layer_sizes]
        
    def forward(self, input_spikes, dt=1.0):
        """
        Propagate spikes through the network.
        
        Args:
            input_spikes (np.ndarray): Boolean or binary array of input spikes
            dt (float): Simulation time step in ms
            
        Returns:
            list: List of spike arrays for each layer [spikes_L1, spikes_L2, ...]
        """
        # Layer 0 (Input)
        current_spikes = input_spikes.astype(np.float32)
        all_spikes = [current_spikes]
        
        # Update Layer 0 traces
        self._update_traces(0, current_spikes, dt)
        
        # Propagate through layers
        for layer_idx, W in enumerate(self.weights):
            next_layer_idx = layer_idx + 1
            n_neurons = self.layer_sizes[next_layer_idx]
            
            # 1. Decay membrane potential
            decay_factor = np.exp(-dt / self.tau_mem)
            self.membrane_potentials[next_layer_idx] *= decay_factor
            
            # 2. Integrate synaptic input
            # Input flow: I = W * spikes
            synaptic_input = current_spikes @ W
            self.membrane_potentials[next_layer_idx] += synaptic_input
            
            # 3. Handle refractory period
            in_refractory = self.refractory_counters[next_layer_idx] > 0
            self.membrane_potentials[next_layer_idx][in_refractory] = 0.0
            self.refractory_counters[next_layer_idx] -= dt
            self.refractory_counters[next_layer_idx] = np.maximum(self.refractory_counters[next_layer_idx], 0)
            
            # 4. Generate spikes (Threshold crossing)
            spike_threshold = 1.0
            spikes = (self.membrane_potentials[next_layer_idx] >= spike_threshold).astype(np.float32)
            
            # 5. Reset membrane potential and set refractory
            self.membrane_potentials[next_layer_idx] *= (1.0 - spikes)
            self.refractory_counters[next_layer_idx][spikes > 0] = self.refractory_time
            
            # 6. Update traces for this layer
            self._update_traces(next_layer_idx, spikes, dt)
            
            # Prepare for next layer
            current_spikes = spikes
            all_spikes.append(spikes)
            
        return all_spikes
    
    def _update_traces(self, layer_idx, spikes, dt):
        """Update eligibility traces for STDP"""
        decay_factor = np.exp(-dt / self.tau_syn)
        self.spike_traces[layer_idx] *= decay_factor
        self.spike_traces[layer_idx] += spikes
        
    def apply_stdp(self, pre_spikes_list, post_spikes_list, modulation=1.0):
        """
        Apply Spike-Timing-Dependent Plasticity to weights.
        
        Args:
            pre_spikes_list (list): Spikes from pre-synaptic layers in previous step
            post_spikes_list (list): Spikes from post-synaptic layers in current step
            modulation (float): Global learning rate modulator (e.g., from dopamine)
        """
        # Note: In standard STDP, we update when a spike OCCURS.
        # This function assumes it's called every time step or batch of steps.
        # For effective online STDP, we use the traces.
        
        for layer_idx in range(len(self.weights)):
            # Pre-synaptic trace (from layer i)
            pre_trace = self.spike_traces[layer_idx]
            # Post-synaptic trace (from layer i+1)
            post_trace = self.spike_traces[layer_idx + 1]
            
            # Current spikes
            # pre_spike = pre_spikes_list[layer_idx] # Not strictly needed if using traces properly
            # post_spike = post_spikes_list[layer_idx + 1] # Used to trigger LTP
            
            # We need the spikes that JUST happened to trigger the weight update?
            # Actually, standard trace-based STDP:
            # dW += A_plus * pre_trace * post_spike (LTP: pre trace exists when post fires)
            # dW -= A_minus * post_trace * pre_spike (LTD: post trace exists when pre fires)
            
            # Recover the spikes from the traces? No, passed in lists should be the instantaneous spikes.
            # But the lists passed to this function `apply_stdp` usually come from `forward`.
            # Let's assume `pre_spikes_list` contains the spikes from the `forward` pass we just did.
            
            pre_spike_now = pre_spikes_list[layer_idx]
            post_spike_now = pre_spikes_list[layer_idx + 1]
            
            # Sparse STDP Update:
            # Only update weights connecting to active post-neurons (LTP)
            # or from active pre-neurons (LTD).
            
            # 1. LTP: Post fires, Pre trace is high
            # indices where post_spike_now > 0
            post_indices = np.where(post_spike_now > 0)[0]
            if len(post_indices) > 0:
                # We want to add A_plus * pre_trace[i] * post_spike[j] to W[i, j]
                # for all i, and active j.
                # W[:, j] += modulation * A_plus * pre_trace[:] * post_spike[j]
                
                # If binary spikes (1.0), post_spike[j] is 1.
                # So W[:, j] += mod * A_plus * pre_trace
                
                # Create update matrix for relevant columns
                # This is still vector operation but avoids full matrix creation
                # pre_trace is (N_pre,), we need to add it to each active column of W
                
                # Efficient broadcasting:
                # W[:, post_indices] += (modulation * self.A_plus * pre_trace[:, np.newaxis])
                active_update = (modulation * self.A_plus * pre_trace[:, np.newaxis])
                self.weights[layer_idx][:, post_indices] += active_update

            # 2. LTD: Pre fires, Post trace is high
            # indices where pre_spike_now > 0
            pre_indices = np.where(pre_spike_now > 0)[0]
            if len(pre_indices) > 0:
                # W[i, :] -= mod * A_minus * post_trace
                active_update = (modulation * self.A_minus * post_trace[np.newaxis, :])
                self.weights[layer_idx][pre_indices, :] -= active_update
            
            # Bound weights (vectorized clipping is relatively fast if done in place or infrequently)
            # Clip only modified rows/cols? Or full matrix?
            # Full clip is safer but slow. Let's do it every N steps or just trust logic?
            # For correctness we clip. To optimize, maybe clip less often.
            # But let's stick to full clip for stability, main cost was Outer Product allocation.
            np.clip(self.weights[layer_idx], -1.0, 1.0, out=self.weights[layer_idx])

    def save_checkpoint(self, filepath):
        """Save weights to file"""
        np.savez(filepath, *self.weights)
        
    def load_checkpoint(self, filepath):
        """Load weights from file"""
        data = np.load(filepath)
        self.weights = [data[k] for k in data.files]
