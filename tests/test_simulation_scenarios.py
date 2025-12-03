import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from simulation_logic import simulate_step

class TestSimulationScenarios(unittest.TestCase):
    
    def setUp(self):
        # Create a mock row of data
        self.mock_row = pd.Series({
            'Generated_Power': 100.0, # 100W per panel * 30 panels = 3000W
            'Grid_Consumption': 1000.0,
            'Timestamp': pd.Timestamp('2023-01-01 12:00:00')
        })
        self.battery_capacity = 10000
        self.max_power = 5000
        self.dt = 1/60 # 1 minute

    def test_scenario_normal_day_surplus(self):
        """
        Scenario A: Normal Sunny Day, Surplus Solar.
        Expectation: Battery Charges.
        """
        print("\n--- Scenario A: Normal Sunny Day (Surplus) ---")
        soc = 50.0
        # Solar = 3000W, Load = 1000W -> Net = +2000W
        result = simulate_step(
            self.mock_row, soc, [True, True, True], 1.0, "Charge", 1000.0
        )
        
        print(f"Net Power: {result['net_power']} W")
        print(f"Action: {result['action_taken']}")
        print(f"Energy Flow: {result['energy_flow']} W (Positive = Charge)")
        
        self.assertTrue(result['net_power'] > 0, "Should have surplus power")
        self.assertTrue(result['energy_flow'] > 0, "Battery should be charging")
        self.assertTrue(result['next_soc'] > soc, "SoC should increase")

    def test_scenario_cloudy_day_deficit(self):
        """
        Scenario B: Cloudy Day (40% Solar), Deficit.
        Expectation: Battery Discharges.
        """
        print("\n--- Scenario B: Cloudy Day (Deficit) ---")
        soc = 50.0
        # Solar = 3000 * 0.4 = 1200W, Load = 1000W -> Net = +200W (Still surplus actually)
        # Let's increase load to force deficit
        cloudy_row = self.mock_row.copy()
        cloudy_row['Grid_Consumption'] = 2000.0 # Load > Solar (1200)
        
        result = simulate_step(
            cloudy_row, soc, [True, True, True], 0.4, "Discharge", 2000.0
        )
        
        print(f"Net Power: {result['net_power']} W")
        print(f"Energy Flow: {result['energy_flow']} W (Negative = Discharge)")
        
        self.assertTrue(result['net_power'] < 0, "Should have power deficit")
        self.assertTrue(result['energy_flow'] < 0, "Battery should be discharging")
        self.assertTrue(result['next_soc'] < soc, "SoC should decrease")

    def test_scenario_fault_isolation(self):
        """
        Scenario C: Fault Injection (String 1 Down).
        Expectation: Solar Derated to 66%, Action = Isolate, No Charge.
        """
        print("\n--- Scenario C: Fault Injection (String 1 Down) ---")
        soc = 50.0
        # Solar = 3000W normally. With 1 string down -> 2000W.
        
        result = simulate_step(
            self.mock_row, soc, [False, True, True], 1.0, "Charge", 1000.0
        )
        
        print(f"Generated Power: {result['generated_power']} W (Expected ~2000)")
        print(f"Action: {result['action_taken']}")
        print(f"Status: {result['status_message']}")
        
        self.assertAlmostEqual(result['generated_power'], 2000.0, delta=1.0)
        self.assertEqual(result['action_taken'], "Isolate")
        self.assertEqual(result['energy_flow'], 0.0, "Battery should NOT charge during fault")
        self.assertTrue(result['fault_detected'])

    def test_scenario_grid_export(self):
        """
        Scenario D: Battery Full + Surplus Solar.
        Expectation: Export to Grid (Grid Net > 0).
        """
        print("\n--- Scenario D: Battery Full + Surplus ---")
        soc = 100.0 # Battery Full
        
        result = simulate_step(
            self.mock_row, soc, [True, True, True], 1.0, "Charge", 1000.0
        )
        
        print(f"SoC: {result['next_soc']}%")
        print(f"Energy Flow: {result['energy_flow']} W (Should be 0 as battery is full)")
        print(f"Grid Net: {result['grid_net']} W (Should be positive export)")
        
        self.assertAlmostEqual(result['energy_flow'], 0.0, delta=1.0)
        self.assertTrue(result['grid_net'] > 0, "Should export excess power to grid")

    def test_scenario_fault_surplus_export(self):
        """
        Scenario E: Fault (String 1 Down) but still Surplus Solar.
        Expectation: Battery Idle (Isolate), Excess Solar Exported to Grid.
        """
        print("\n--- Scenario E: Fault + Surplus ---")
        soc = 50.0
        # Solar = 3000W -> Derated to 2000W. Load = 1000W. Net = +1000W.
        
        result = simulate_step(
            self.mock_row, soc, [False, True, True], 1.0, "Charge", 1000.0
        )
        
        print(f"Generated (Derated): {result['generated_power']} W")
        print(f"Load: {result['grid_consumption']} W")
        print(f"Battery Flow: {result['energy_flow']} W (Should be 0)")
        print(f"Grid Net: {result['grid_net']} W (Should be +1000 Export)")
        
        self.assertEqual(result['energy_flow'], 0.0, "Battery should be isolated (0 flow)")
        self.assertEqual(result['grid_net'], 1000.0, "Excess derated solar should be exported")

    def test_scenario_fault_deficit_import(self):
        """
        Scenario F: Fault (String 1 Down) causing Deficit.
        Expectation: Battery Idle (Isolate), Deficit Imported from Grid.
        """
        print("\n--- Scenario F: Fault + Deficit ---")
        soc = 50.0
        # Solar = 3000W -> Derated to 2000W. Load = 2500W. Net = -500W.
        high_load_row = self.mock_row.copy()
        high_load_row['Grid_Consumption'] = 2500.0
        
        result = simulate_step(
            high_load_row, soc, [False, True, True], 1.0, "Charge", 2500.0
        )
        
        print(f"Generated (Derated): {result['generated_power']} W")
        print(f"Load: {result['grid_consumption']} W")
        print(f"Battery Flow: {result['energy_flow']} W (Should be negative discharge)")
        print(f"Grid Net: {result['grid_net']} W (Should be reduced import)")
        
        self.assertTrue(result['energy_flow'] < 0, "Battery should discharge to support load during fault")
        # Grid Net should be Deficit - Battery Discharge. If Battery covers all, Grid Net is 0.
        # Max discharge is limited by power.
        # Deficit is 500W. Max power 5000W. Battery should cover it all.
        self.assertEqual(result['grid_net'], 0.0, "Battery should cover the deficit")

if __name__ == '__main__':
    unittest.main()
