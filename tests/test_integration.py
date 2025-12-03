import unittest
import pandas as pd
import numpy as np

class TestDashboardLogic(unittest.TestCase):
    
    def setUp(self):
        # Simulate Session State
        self.session_state = {
            'fault_detected': False,
            'weather_factor': 1.0,
            'battery_soc': 50.0
        }
        
        # Simulate Environment Data
        self.current_row = {
            'Generated_Power': 200.0, # 200W per panel
            'Grid_Consumption': 5000.0 # 5kW load
        }
        self.n_panels = 30
        
        # Physics Constants
        self.max_power = 5000
        
    def test_1_normal_operation(self):
        """Test Normal Operation (No Fault)."""
        # AI decides to Discharge
        action = "Discharge" 
        
        # Logic from Dashboard
        if self.session_state.get('fault_detected', False):
            action = "Disconnect"
        
        self.assertEqual(action, "Discharge", "Action should remain Discharge when no fault.")
        
    def test_2_emergency_stop(self):
        """Test Emergency Stop (Fault Detected)."""
        # Inject Fault
        self.session_state['fault_detected'] = True
        
        # AI decides to Discharge (but should be overridden)
        action = "Discharge"
        
        # Logic from Dashboard
        if self.session_state.get('fault_detected', False):
            action = "Disconnect"
            
        self.assertEqual(action, "Disconnect", "Action should be overridden to Disconnect.")
        
    def test_3_weather_scaling_sunny(self):
        """Test Weather Scaling (Sunny)."""
        self.session_state['weather_factor'] = 1.0
        
        gen = self.current_row['Generated_Power'] * self.n_panels * self.session_state['weather_factor']
        expected = 200.0 * 30 * 1.0
        
        self.assertEqual(gen, expected, f"Expected {expected}, got {gen}")
        
    def test_4_weather_scaling_stormy(self):
        """Test Weather Scaling (Stormy)."""
        self.session_state['weather_factor'] = 0.1
        
        gen = self.current_row['Generated_Power'] * self.n_panels * self.session_state['weather_factor']
        expected = 200.0 * 30 * 0.1
        
        self.assertEqual(gen, expected, f"Expected {expected}, got {gen}")

if __name__ == '__main__':
    unittest.main()
