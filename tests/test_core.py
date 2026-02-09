
import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.snn import SpikingNeuralNetwork
from src.core.predictive_coding import PredictiveCodingLayer, PredictiveHierarchy

class TestCore(unittest.TestCase):
    
    def test_snn_initialization(self):
        sizes = [10, 20, 10]
        snn = SpikingNeuralNetwork(sizes)
        self.assertEqual(len(snn.weights), 2)
        self.assertEqual(snn.weights[0].shape, (10, 20))
        self.assertEqual(snn.weights[1].shape, (20, 10))
        
    def test_snn_forward(self):
        sizes = [10, 20]
        snn = SpikingNeuralNetwork(sizes)
        input_spikes = np.ones(10)
        output = snn.forward(input_spikes)
        self.assertEqual(len(output), 2) # [input, layer1]
        self.assertEqual(output[1].shape, (20,))
        
    def test_stdp_update(self):
        sizes = [2, 2]
        snn = SpikingNeuralNetwork(sizes, learning_rate=0.1)
        
        # Initial weights
        initial_weights = snn.weights[0].copy()
        
        # Force pre and post spikes
        # We need traces to be active.
        # Step 1: Fire pre
        input_spikes = np.array([1.0, 0.0])
        snn.forward(input_spikes)
        
        # Step 2: Fire post (force by high weight or just manually set logic? 
        # forward() handles dynamics. 
        # Let's just call apply_stdp with manual spikes and traces)
        
        snn.spike_traces[0] = np.array([1.0, 0.0]) # Pre neuron 0 active
        snn.spike_traces[1] = np.array([0.0, 1.0]) # Post neuron 1 active
        
        # Apply STDP
        # Pre=1, Post=1 -> LTP?
        # In my implementation:
        # dW_plus = A_plus * outer(pre_trace, post_spike)
        # dW_minus = -A_minus * outer(pre_spike, post_trace)
        
        # Let's pass spikes that match the traces for simplicity of testing "LTP"
        pre_spikes = [np.array([1.0, 0.0]), np.array([0.0, 1.0])]
        
        snn.apply_stdp(pre_spikes[0], pre_spikes[1]) # Pass plain arrays not list of arrays?
        # Wait, apply_stdp signature: `apply_stdp(self, pre_spikes_list, post_spikes_list, ...)`
        # Actually it takes `pre_spikes_list`.
        
        snn.apply_stdp(pre_spikes, pre_spikes, modulation=1.0)
        
        # Check if weight changed
        # W[0,1] should increase (Pre 0 -> Post 1)
        self.assertNotEqual(snn.weights[0][0, 1], initial_weights[0, 1])
        
    def test_predictive_coding_inference(self):
        input_size = 5
        hidden_size = 10
        layer = PredictiveCodingLayer(input_size, hidden_size)
        
        input_data = np.random.rand(input_size)
        
        # Run inference
        initial_error = np.sum(layer.prediction_error**2)
        layer.infer(input_data, n_steps=20)
        final_error = np.sum(layer.prediction_error**2)
        
        # Error might not decrease if weights are random and we just update representation?
        # Updating representation minimizes error:
        # Error = Input - (Rep @ W)
        # We adjust Rep to match Input.
        # Yes, error should decrease.
        
        # Note: initial error is 0 because self.prediction_error init at 0?
        # No, infer() calculates it immediately.
        # Wait, I initialized prediction_error to 0 in __init__.
        # infer loop calc error first thing.
        # So comparison:
        # Run 1 step
        layer.infer(input_data, n_steps=1)
        err1 = np.sum(layer.prediction_error**2)
        
        # Run more steps
        layer.infer(input_data, n_steps=20)
        err2 = np.sum(layer.prediction_error**2)
        
        self.assertLess(err2, err1)
        
    def test_predictive_coding_learning(self):
        pc = PredictiveCodingLayer(5, 5)
        pc.representation = np.ones(5)
        pc.prediction_error = np.ones(5)
        
        w_before = pc.W_pred.copy()
        pc.learn()
        w_after = pc.W_pred
        
        # Weights should change
        self.assertFalse(np.array_equal(w_before, w_after))

if __name__ == '__main__':
    unittest.main()
