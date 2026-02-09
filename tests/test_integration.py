
import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.digital_baby import DigitalBaby
from src.environment import WhiteRoomEnvironment

class TestIntegration(unittest.TestCase):
    
    def test_initialization(self):
        baby = DigitalBaby()
        self.assertIsNotNone(baby.retina)
        self.assertIsNotNone(baby.snn)
        
    def test_one_step(self):
        baby = DigitalBaby()
        env = WhiteRoomEnvironment()
        
        # Run one step
        result = baby.step(env)
        
        # Check result structure
        self.assertIn('action', result)
        self.assertIn('free_energy', result)
        self.assertIn('neuromodulators', result)
        
        # Check if gaze moved (or stayed)
        # Action is index.
        # If action=0 (Stay), gaze shouldn't change much.
        # But we verify it ran without error.
        print(f"Action taken: {result['action']}")
        print(f"Free Energy: {result['free_energy']}")
        print(f"Neuromodulators: {result['neuromodulators']}")

if __name__ == '__main__':
    unittest.main()
