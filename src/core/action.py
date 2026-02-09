"""
Action Selection Module.
Implements Active Inference by selecting actions that minimize Expected Free Energy.
Includes a learned transition model for simulating action consequences.
"""

import numpy as np


class ActionSelector:
    """
    Selects actions to minimize Expected Free Energy (G).
    Learns a forward model of state transitions: State(t+1) ~ f(State(t), Action(t))
    """
    
    def __init__(self, n_actions, state_size):
        """
        Args:
            n_actions: Number of discrete actions available
            state_size: Dimensionality of the latent state (from predictive hierarchy)
        """
        self.n_actions = n_actions
        self.state_size = state_size
        
        # Forward model weights (Mental model of physics)
        # We predict next state based on current state and action.
        # Simple linear approximation: s_next = s @ W_s + action_onehot @ W_a
        self.W_s = np.eye(state_size) # Default to identity (persistence)
        self.W_a = np.zeros((n_actions, state_size))
        
        # Learning rate for forward model
        self.learning_rate = 0.01
        
        # Action priors (habits) - logic P(a)
        self.action_counts = np.ones(n_actions)
        
        # Previous state storage for learning
        self.prev_state = None
        self.prev_action = None
        
    def learn_transition(self, current_state):
        """
        Update forward model based on observed transition.
        State(t-1), Action(t-1) -> State(t) (current_state)
        """
        if self.prev_state is not None and self.prev_action is not None:
            # Predict what we thought would happen
            # One-hot action
            action_vec = np.zeros(self.n_actions)
            action_vec[self.prev_action] = 1.0
            
            predicted_state = self.prev_state @ self.W_s + action_vec @ self.W_a
            
            # Prediction error
            error = current_state - predicted_state
            
            # Update weights (Delta rule / Gradient descent)
            # dW_s = lr * prev_state.T * error
            # dW_a = lr * action_vec.T * error
            
            self.W_s += self.learning_rate * np.outer(self.prev_state, error)
            self.W_a += self.learning_rate * np.outer(action_vec, error)
            
            # Update priors (habit learning)
            self.action_counts[self.prev_action] += 1
            
        # Update history
        self.prev_state = current_state.copy()
        
    def select_action(self, current_state, temperature=1.0, goal_state=None):
        """
        Select action via softmax over negative Expected Free Energy.
        
        Args:
            current_state: Current latent state
            temperature: Exploration parameter (inverse precision)
            goal_state: Preferred state (homeostatic setpoint)
            
        Returns:
            int: Selected action index
        """
        # Calculate Expected Free Energy for each action
        G = np.zeros(self.n_actions)
        
        for a in range(self.n_actions):
            G[a] = self._compute_expected_free_energy(a, current_state, goal_state)
            
        # Softmax selection: P(a) ~ exp(-G / temp)
        # Subtract min for numerical stability
        G_shifted = G - np.min(G)
        exp_neg_G = np.exp(-G_shifted / temperature)
        action_probs = exp_neg_G / np.sum(exp_neg_G)
        
        # Sample action
        selected_action = np.random.choice(self.n_actions, p=action_probs)
        
        # Store for next learning step
        self.prev_action = selected_action
        
        return selected_action
    
    def _compute_expected_free_energy(self, action, current_state, goal_state=None):
        """
        G(a) = Epistemic Value + Pragmatic Value
             ~ -Entropy(Predicted) - LogLikelihood(Goal)
        """
        # 1. Simulate outcome
        action_vec = np.zeros(self.n_actions)
        action_vec[action] = 1.0
        predicted_state = current_state @ self.W_s + action_vec @ self.W_a
        
        # 2. Epistemic Value (Information Gain)
        # Approximated by variance/uncertainty of the prediction.
        # Ideally we'd have a variance estimate from the model.
        # For this simple model, we can use the magnitude of the learned update? 
        # Or simply: novel actions (low count) have high epistemic value.
        # Let's use count-based intrinsic motivation for now.
        epistemic_value = 1.0 / np.sqrt(self.action_counts[action])
        
        # 3. Pragmatic Value (Instrumental Value)
        # Distance to goal state.
        if goal_state is not None:
            # Negative squared error (Maximize similarity)
            pragmatic_value = -np.mean((predicted_state - goal_state)**2)
        else:
            pragmatic_value = 0.0
            
        # Free Energy is minimized, so we negate values we want to maximize?
        # G = -Epistemic - Pragmatic
        # We want to Maximize (Epistemic + Pragmatic).
        # So G should be negative of that.
        
        return -(epistemic_value + pragmatic_value)
    
    def reset(self):
        """Reset history"""
        self.prev_state = None
        self.prev_action = None
