import numpy as np

def simulate_step(
    current_row, 
    battery_soc, 
    string_status, 
    weather_factor, 
    action, 
    pred_consumption, 
    dt_hours=1/60, 
    battery_capacity_wh=10000, 
    max_power=5000
):
    """
    Executes one step of the simulation physics and control logic.
    Returns the next state and calculated metrics.
    """
    
    # 1. Calculate Real Net Power
    n_panels = 30
    
    # Apply Weather Factor
    generated_power_total = current_row['Generated_Power'] * n_panels * weather_factor
    grid_consumption = current_row['Grid_Consumption']
    net_power = generated_power_total - grid_consumption

    # 2. Calculate Physical Limits
    current_wh = (battery_soc / 100.0) * battery_capacity_wh
    max_charge_wh = battery_capacity_wh - current_wh
    max_discharge_wh = current_wh
    
    max_charge_power_limit = max_charge_wh / dt_hours
    max_discharge_power_limit = max_discharge_wh / dt_hours
    
    eff_max_charge = min(max_power, max_charge_power_limit)
    eff_max_discharge = min(max_power, max_discharge_power_limit)

    target_flow = 0.0
    fault_detected = False
    status_message = ""

    # 3. Determine Target Flow (Hierarchy of Control)
    
    # Priority 0: SMART ISOLATION (Fault Detected)
    active_strings = sum(string_status)
    health_factor = active_strings / 3.0
    
    if active_strings < 3:
        fault_detected = True
        generated_power_total *= health_factor # Derate Solar
        net_power = generated_power_total - grid_consumption # Recalculate Net Power
        
        # Override Action to "Isolate" (No Battery Charging)
        action = "Isolate"
        
        # Safety Logic: Allow Discharge (Support Load), Block Charge (Safety)
        if net_power < 0:
             target_flow = net_power # Discharge to cover deficit
        else:
             target_flow = 0.0 # Do not charge from faulty array 
        
        down_strings = [f"String {i+1}" for i, s in enumerate(string_status) if not s]
        status_message = f"FAULT RESPONSE: {', '.join(down_strings)} Isolated. System operating at {int(health_factor*100)}% capacity."
        
    # Priority 1: AI "Charge"
    elif action == "Charge":
        if net_power > 0:
            target_flow = net_power # Charge only with surplus solar
        else:
            target_flow = max_power # Charge from Grid
        
    # Priority 3: AI "Discharge"
    elif action == "Discharge":
        if net_power < 0:
            target_flow = net_power # Match deficit
        else:
            target_flow = 0.0 # Do not discharge if surplus
        
    # Priority 4: Baseline (Hold)
    else:
        if net_power > 0:
            target_flow = net_power
        else:
            target_flow = net_power
            
    # 4. Apply Physical Constraints
    energy_flow = max(-eff_max_discharge, min(eff_max_charge, target_flow))

    # 5. Update SoC
    delta_wh = energy_flow * dt_hours
    next_soc = battery_soc + (delta_wh / battery_capacity_wh) * 100.0
    next_soc = np.clip(next_soc, 0.0, 100.0)

    # 6. Calculate Grid Interaction
    grid_net = net_power - energy_flow
    
    return {
        "next_soc": next_soc,
        "energy_flow": energy_flow,
        "generated_power": generated_power_total,
        "grid_consumption": grid_consumption,
        "net_power": net_power,
        "grid_net": grid_net,
        "action_taken": action,
        "fault_detected": fault_detected,
        "status_message": status_message
    }
