"""
Digital Baby: Main Agent Class.
Integrates all cognitive subsystems into an autopoietic learning entity.
"""

import numpy as np
from .sensory import FoveatedRetina, CochlearEncoder, SpikePattern
from .core import SpikingNeuralNetwork, PredictiveHierarchy, HomeostaticController, AssociativeMemory, ActionSelector


class DigitalBaby:
    """
    The Digital Baby agent.
    
    Architecture:
    1. Sensory Layout: Foveated Retina (64x64) + Cochlea (32 ch)
    2. Core Process: SNN + Predictive Coding Hierarchy
    3. Memory: Associative Attractor Network
    4. Regulation: Homeostatic Controller (Dopamine, ACh, NE)
    5. Action: Active Inference (Expected Free Energy minimization)
    """
    
    def __init__(self):
        # --- 1. Sensory Apparatus ---
        # Optimized: 32x32 resolution for real-time CPU performance
        self.retina = FoveatedRetina(resolution=(32, 32), fovea_radius=8)
        self.cochlea = CochlearEncoder(n_channels=32)
        
        # --- 2. Neural Substrate ---
        # SNN Layer sizes:
        # Input: Visual (32x32 ON + 32x32 OFF = 2048) + Audio (32) = 2080
        # Hidden layers: 512 -> 128 -> 64
        self.snn = SpikingNeuralNetwork(
            layer_sizes=[2080, 512, 128, 64],
            tau_mem=20.0,
            learning_rate=0.01
        )
        
        # Predictive Hierarchy (JEPA)
        # 2080 -> 256 -> 64 -> 32
        self.predictive_hierarchy = PredictiveHierarchy(
            layer_sizes=[2080, 256, 64, 32]
        )
        
        # --- 3. Memory ---
        # Associative memory binding visual and audio representations
        # We'll use the top-level latent states for this.
        # --- 3. Memory ---
        # Associative memory binding visual and audio representations
        # We'll use the top-level latent states for this.
        self.memory = AssociativeMemory(n_units=64) 
        
        # --- 4. Regulation ---
        self.homeostasis = HomeostaticController()
        
        # --- 5. Action ---
        # Active Inference on top-most predictive layer state (32).
        self.action_selector = ActionSelector(
            n_actions=9,
            state_size=32
        )
        
        # State
        self.gaze_position = np.array([16, 16]) # Center of 32x32
        self.alive = True
        self.age_steps = 0
        
    def step(self, environment):
        """
        Execute one cognitive cycle.
        
        Args:
            environment: Simulation environment providing .get_visual(gaze) and .get_audio()
        """
        # 1. ACTIVE PERCEPTION
        # Get sensory data based on CURRENT gaze
        visual_frame = environment.get_visual_frame(self.gaze_position)
        audio_chunk = environment.get_audio_chunk()
        
        # Encode
        visual_spikes = self.retina.encode(visual_frame, self.gaze_position)
        audio_spikes = self.cochlea.encode(audio_chunk)
        
        # Flatten and concatenate inputs
        flat_visual = visual_spikes.data # ON + OFF flattened
        
        # Audio is (channels, time_bins) -> Flatten breaks channel structure unless we aggregate time
        # We need (32,)
        # AudioSpikePattern.data is flattened, so we should look at .channels instead
        if audio_spikes.channels is not None:
             # Mean over time bins (rate coding for this step)
             flat_audio = audio_spikes.channels.mean(axis=1)
        else:
             flat_audio = np.zeros(32)
        
        # Ensure sizes match expected input (2080)
        # Retina: 32*32*2 = 2048
        # Cochlea: 32 channels. 
        # Total: 2080.
        sensory_input = np.concatenate([flat_visual, flat_audio])
        
        # 2. PREDICT & COMPARE
        # Run predictive coding hierarchy (Generative Model)
        # We infer the causes of sensory input
        prediction_errors, free_energy = self.predictive_hierarchy.process(
            sensory_input, n_inference_steps=5
        )
        
        # 3. LEARN (SNN & Plasticity)
        # Compute neuromodulation
        # Novelty? Deviation from attractor?
        # Let's use SNN activity as a proxy for "familiarity" inside Homeostasis or just use free energy
        # For novelty, we can check if the current pattern is stable in Memory
        
        # Propagate SNN (Discriminative / Fast path)
        all_spikes = self.snn.forward(sensory_input)
        mean_firing_rate = np.mean([s.mean() for s in all_spikes])
        
        # Check memory stability (Novelty detection)
        # Use top layer spikes as pattern
        current_pattern = all_spikes[-1] 
        # (Note: SNN outputs are spikes, binary. Memory handles -1/1 or 0/1)
        # We convert 0/1 to -1/1 in memory module roughly
        stability = self.memory.get_stability(current_pattern)
        novelty_score = 1.0 - stability
        
        # Update Homeostasis
        # Normalize free energy to MSE (approximate)
        # Total units in predictive hierarchy roughly: 2080 + 256 + 64 + 32 = 2432
        mse = free_energy / 2432.0
        
        neuromodulators = self.homeostasis.update(
            prediction_error=mse,
            mean_firing_rate=mean_firing_rate,
            novelty_score=novelty_score
        )
        
        # Plasticity Modulation
        plasticity_mod = self.homeostasis.compute_plasticity_modulation()
        
        # Apply STDP to SNN
        # We need pre/post spikes. 'all_spikes' has [Input, L1, L2, L3, L4]
        # We only pass the list of layers to apply_stdp
        # Wait, apply_stdp takes `pre_spikes_list` (all spikes?)
        # Yes, based on my implementation and test.
        self.snn.apply_stdp(all_spikes, all_spikes, modulation=plasticity_mod)
        
        # Learn Predictive Hierarchy (Hebbian on errors)
        self.predictive_hierarchy.update_weights(modulation=plasticity_mod)
        
        # Learn Transition Model (in Action Selector)
        # Current latent state (from predictive hierarchy top layer)
        current_latent_state = self.predictive_hierarchy.get_top_representation()
        self.action_selector.learn_transition(current_latent_state)
        
        # 4. ASSOCIATIVE LEARNING
        # If significant events in both modalities?
        # Or just always learn stable patterns?
        # "Fire together, wire together"
        # If both visual and audio are active above threshold
        if flat_visual.mean() > 0.01 and flat_audio.mean() > 0.01:
            # Store in associative memory
            self.memory.store_pattern(
                visual_pattern=all_spikes[-1], # Using abstract representation
                audio_pattern=np.array([]),    # Fused in SNN, so pattern is single vector
                strength=plasticity_mod
            )
            # Note: My AssociativeMemory.store_pattern takes visual+audio args.
            # But here I'm using the SNN top layer which is already fused?
            # Or should I store the raw sensory embeddings?
            # The prompt said "bind visual attractor and audio attractor".
            # If SNN is multimodal, the top layer IS the bound representation.
            # So I just store/reinforce it.
            # In my implementation of Memory, I concat them.
            # I can pass empty array for second arg if I pass combined to first.
            
        # 5. ACTION SELECTION
        # Select action to minimize Expected Free Energy
        # We use the predictive hierarchy's latent state as the "State"
        # And we use Homeostasis drives to set "Goal State" (optional, or implicit in EFE)
        # ActionSelector needs `current_state`.
        
        action_idx = self.action_selector.select_action(
            current_state=current_latent_state,
            temperature=1.0 / neuromodulators['norepinephrine'] # High NE -> High temp -> Random exploration
        )
        
        # Execute Action
        self._execute_action(action_idx, environment)
        
        self.age_steps += 1
        
        return {
            'action': action_idx,
            'free_energy': free_energy,
            'neuromodulators': neuromodulators
        }
    
    def _execute_action(self, action_idx, environment):
        """Translate abstract action index to motor command"""
        # Actions 0-8: Move gaze
        # 0: Stay
        # 1-8: 8 directions (N, NE, E, SE, S, SW, W, NW)
        step_size = 5
        
        moves = [
            (0, 0),
            (-step_size, 0), (-step_size, step_size), (0, step_size), (step_size, step_size),
            (step_size, 0), (step_size, -step_size), (0, -step_size), (-step_size, -step_size)
        ]
        
        if action_idx < len(moves):
            dy, dx = moves[action_idx]
            self.gaze_position[0] += dy
            self.gaze_position[1] += dx
            
            # Clip to retina bounds
            # Environment bounds (visual field might be larger than retina?)
            # Assuming environment handles bounds or we clip to 0-64? 
            # Wait, retina is 32x32, but environment is "White Room" (128x128).
            # Gaze position is in Environment coordinates.
            self.gaze_position = np.clip(self.gaze_position, 0, 127)
        
        # Else: vocalization? (Not implemented yet)
