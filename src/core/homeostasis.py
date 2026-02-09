"""
Homeostatic Controller.
Maintains intrinsic drives and modulates plasticity based on
prediction error, novelty, and internal stability.
"""

import numpy as np


class HomeostaticController:
    """
    Regulates the system's internal state to maintain homeostasis.
    simulate neuromodulators (Dopamine, Acetylcholine, Norepinephrine).
    """
    
    def __init__(self):
        # drive states (normalized 0-1)
        self.drives = {
            'predictive_accuracy': 0.5,  # sustained low error -> high accuracy
            'encoding_efficiency': 0.5,  # sustained moderate firing -> high efficiency
            'novelty_salience': 0.5      # recent surprise -> high novelty
        }
        
        # homeostatic setpoints (optimal zones)
        self.setpoints = {
            'predictive_accuracy': 0.8,  # we want high accuracy
            'encoding_efficiency': 0.7,  # we want sparse but meaningful codes
            'novelty_salience': 0.3      # we want some novelty but not chaos
        }
        
        # neuromodulator levels (baseline 1.0)
        self.neuromodulators = {
            'dopamine': 1.0,      # reward prediction error / surprise
            'acetylcholine': 1.0, # expected uncertainty / attention
            'norepinephrine': 1.0 # unexpected uncertainty / arousal
        }
        
        # history buffers for computing trends
        self.error_history = []
        self.activity_history = []
        self.history_len = 100
        
    def update(self, prediction_error, mean_firing_rate, novelty_score):
        """
        Update drives and neuromodulators based on current sensory/cognitive state.
        
        Args:
            prediction_error (float): Current global prediction error (free energy)
            mean_firing_rate (float): Average spike rate across network
            novelty_score (float): Computed novelty of current stimulus (0-1)
            
        Returns:
            dict: Current neuromodulator levels
        """
        # 1. Update histories
        self.error_history.append(prediction_error)
        self.activity_history.append(mean_firing_rate)
        
        if len(self.error_history) > self.history_len:
            self.error_history.pop(0)
            self.activity_history.pop(0)
            
        # 2. Compute drive states
        # Predictive accuracy: Inverse of recent average error
        avg_error = np.mean(self.error_history) if self.error_history else 0.0
        # Accuracy = 1 / (1 + error)
        self.drives['predictive_accuracy'] = 1.0 / (1.0 + avg_error)
        
        # Encoding efficiency: Penalize too high or too low activity
        # idealized rate ~ 0.05 - 0.1 (sparse)
        # simplistic efficiency metric: 1 - |rate - target| / target
        target_rate = 0.1
        rate_dev = abs(mean_firing_rate - target_rate)
        self.drives['encoding_efficiency'] = max(0, 1.0 - (rate_dev / target_rate))
        
        # Novelty: Direct input
        self.drives['novelty_salience'] = novelty_score
        
        # 3. Compute Neuromodulators
        
        # Dopamine (DA): Encodes "better than expected" or "meaningful surprise"
        # Increase if prediction error is high (surprise) BUT decreasing (learning progress)
        # For this simple model: DA scales with prediction error (to drive learning)
        # Baseline 1.0 + deviation
        # Dopamine (DA): Encodes "meaningful surprise" or saliency
        # High prediction error = High Dopamine = High Plasticity
        # We must clip this to prevent explosion.
        # Use a log scale for error to handle large dynamic ranges?
        # Or just clip.
        da_response = 1.0 + (prediction_error - 0.05) * 10.0 # High gain on small error deviations
        self.neuromodulators['dopamine'] = np.clip(da_response, 0.1, 5.0)
        
        # Acetylcholine (ACh): Precision / Attention
        # High error -> Low accuracy -> High ACh -> Increase learning rate
        accuracy_gap = self.setpoints['predictive_accuracy'] - self.drives['predictive_accuracy']
        ach_response = 1.0 + accuracy_gap * 2.0
        self.neuromodulators['acetylcholine'] = np.clip(ach_response, 0.1, 5.0)
        
        # Norepinephrine (NE): Encodes "unexpected uncertainty" (arousal/reset)
        # Triggered by massive prediction errors or sudden novelty
        self.neuromodulators['norepinephrine'] = 1.0 + novelty_score * 2.0
        
        return self.neuromodulators
        
    def compute_plasticity_modulation(self):
        """
        Compute global scaling factor for synaptic plasticity (STDP rate).
        High DA + high ACh = high plasticity.
        """
        da = self.neuromodulators['dopamine']
        ach = self.neuromodulators['acetylcholine']
        
        # Interaction effect
        modulation = da * ach
        
        # Normalize/Clip to reasonable range
        return np.clip(modulation, 0.1, 5.0)

    def compute_exploration_drive(self):
        """
        Compute tendency to explore (random actions) vs exploit (prediction minimization).
        High NE -> High exploration.
        """
        return self.neuromodulators['norepinephrine']
