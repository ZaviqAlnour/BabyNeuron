"""
Predictive Coding Hierarchy.
Implements a bio-plausible JEPA (Joint Embedding Predictive Architecture)
where each layer predicts the activity of the layer below.
"""

import numpy as np


class PredictiveCodingLayer:
    """
    Single layer of predictive coding hierarchy.
    
    Attributes:
        representation (np.ndarray): Current state/activation of this layer
        prediction_error (np.ndarray): Error between input and top-down prediction
        W_pred (np.ndarray): Top-down prediction weights (Hidden -> Input)
    """
    
    def __init__(self, input_size, hidden_size, output_size=None):
        """
        Args:
            input_size: Size of bottom-up input (from layer below)
            hidden_size: Size of this layer's representation
            output_size: Size of top-down prediction (to layer below). 
                         Usually same as input_size for reconstruction.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size if output_size is not None else input_size
        
        # State variables
        self.representation = np.zeros(hidden_size)
        self.prediction_error = np.zeros(self.output_size)
        
        # Top-down prediction weights (Hidden -> Input)
        # We predict the layer BELOW us.
        # Initialize with small random weights
        scale = 1.0 / np.sqrt(hidden_size)
        self.W_pred = np.random.randn(hidden_size, self.output_size) * scale
        
        # Inference parameters
        self.inference_rate = 0.1
        self.learning_rate = 0.01
        
    def reset(self):
        """Reset state variables"""
        self.representation = np.zeros(self.hidden_size)
        self.prediction_error = np.zeros(self.output_size)
        
    def predict(self):
        """Generate top-down prediction"""
        return self.representation @ self.W_pred
        
    def infer(self, bottom_up_input, top_down_prediction=None, n_steps=10):
        """
        Run inference to update representation and minimize prediction error.
        
        Args:
            bottom_up_input: Input from layer below
            top_down_prediction: Input from layer above (optional constraint)
            n_steps: Number of inference iterations
            
        Returns:
            np.ndarray: The final prediction error of this layer
        """
        for _ in range(n_steps):
            # 1. Generate prediction for layer below
            prediction = self.predict()
            
            # 2. Compute prediction error (Input - Prediction)
            self.prediction_error = bottom_up_input - prediction
            
            # 3. Update representation to minimize error
            # Gradient descent on Error^2: dE/dr = -Error * W_pred
            # dR = learning_rate * Error * W_pred
            error_gradient = self.prediction_error @ self.W_pred.T
            
            # Update representation
            self.representation += self.inference_rate * error_gradient
            
            # 4. Integrate top-down prior if available
            if top_down_prediction is not None:
                # Pull representation towards top-down prediction
                prior_error = top_down_prediction - self.representation
                self.representation += self.inference_rate * prior_error
            
            # Stability: Clip representation to prevent explosion
            # Biological neurons have max firing rates.
            self.representation = np.clip(self.representation, -10.0, 10.0)
                
        return self.prediction_error
    
    def learn(self, modulation=1.0):
        """
        Update prediction weights via Hebbian learning on errors.
        Rule: dW = learning_rate * (Representation * Error)
        """
        # Outer product of representation (cause) and error (effect)
        dW = np.outer(self.representation, self.prediction_error)
        
        self.W_pred += modulation * self.learning_rate * dW
        
        # Stability: Clip weights
        np.clip(self.W_pred, -2.0, 2.0, out=self.W_pred)
        # self.W_pred *= 0.999


class PredictiveHierarchy:
    """
    Complete hierarchical predictive coding network.
    
    Structure:
    Input <- [Layer 0] <- [Layer 1] <- ... <- [Layer N]
    """
    
    def __init__(self, layer_sizes):
        """
        Args:
            layer_sizes (list): Size of layers from bottom (input) to top.
                                e.g. [InputDim, Hidden1, Hidden2, Top]
        """
        self.layers = []
        # Create layers. Layer i predicts Layer i-1.
        # Layer 0 is the first hidden layer, predicting the Input.
        # Wait, usually Layer 0 is just the input buffer.
        # Let's align with:
        # Input (size L0)
        # Layer 0 (Hidden size L1) -> Predicts Input
        # Layer 1 (Hidden size L2) -> Predicts Layer 0 Representation
        
        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            hidden_size = layer_sizes[i+1]
            
            layer = PredictiveCodingLayer(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=input_size 
            )
            self.layers.append(layer)
            
    def process(self, sensory_input, n_inference_steps=20):
        """
        Full pass of inference through hierarchy.
        
        Args:
            sensory_input: Bottom-up input to the hierarchy
            n_inference_steps: Iterations per layer per step
            
        Returns:
            list: List of prediction errors for each layer
            float: Total free energy (sum of squared errors)
        """
        current_input = sensory_input
        errors = []
        
        # Bottom-up and Top-down message passing
        # Simplest scheme: Sequential bottom-up inference
        
        # We need to handle top-down predictions from higher layers.
        # We can do a forward pass (infer bottom-up) then backward pass?
        # Or just infer layer-by-layer bottom-up, as feedback from top 
        # is available from *previous* time step (or we iterate globally).
        
        # For this version, we'll do a simple bottom-up pass where each layer
        # infers its representation based on input from below.
        # Ideally, we should iterate the WHOLE hierarchy n times.
        # But per-layer inference is a good approximation if locally recurrent.
        
        # Get top-down predictions first (from previous state)
        top_down_signals = [None] * len(self.layers)
        for i in range(len(self.layers) - 1):
            # Layer i+1 predicts Layer i
            top_down_signals[i] = self.layers[i+1].predict()
            
        # Inference pass
        for i, layer in enumerate(self.layers):
            # Input is either sensory (for Layer 0) or representation of layer below
            if i == 0:
                bottom_up = sensory_input
            else:
                bottom_up = self.layers[i-1].representation
                
            prior = top_down_signals[i]
            
            # Run inference
            error = layer.infer(bottom_up, top_down_prediction=prior, n_steps=n_inference_steps)
            errors.append(error)
            
        # Compute total free energy
        total_free_energy = sum(np.sum(e**2) for e in errors)
        
        return errors, total_free_energy

    def update_weights(self, modulation=1.0):
        """Trigger learning in all layers"""
        for layer in self.layers:
            layer.learn(modulation)

    def get_top_representation(self):
        """Get the state of the highest layer"""
        return self.layers[-1].representation
