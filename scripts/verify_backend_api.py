import urllib.request
import urllib.error
import json
import time

API_URL = "http://localhost:8001"

def post_json(endpoint, data=None):
    url = f"{API_URL}{endpoint}"
    if data:
        json_data = json.dumps(data).encode('utf-8')
        req = urllib.request.Request(url, data=json_data, headers={'Content-Type': 'application/json'})
    else:
        req = urllib.request.Request(url, method='POST')
        
    try:
        with urllib.request.urlopen(req) as response:
            if response.status != 200:
                print(f"❌ Error {endpoint}: Status {response.status}")
                return None
            return json.loads(response.read().decode('utf-8'))
    except urllib.error.URLError as e:
        print(f"❌ Connection Error {endpoint}: {e}")
        return None

def run_test():
    print(f"Testing Backend at {API_URL} (using urllib)...")
    
    # 1. Reset
    res = post_json("/reset")
    if res:
        print("✅ Reset Successful")
    else:
        return

    # 2. Warmup (Healthy)
    print("\n--- Phase 1: Warming up (Healthy State) ---")
    # Need 20 steps to fill buffer
    for i in range(25):
        res = post_json("/step", {"action": "Hold", "steps": 1})
        if not res: continue
        
        fault = res.get('fault_prediction')
        
        # Check status
        if i >= 20:
            if fault and fault['detected'] == False and "Buffered" not in fault.get('type', ''):
                print(f"✅ Step {i+1}: Healthy Confirmed ({fault['type']})")
            elif fault and "Data Buffering" in fault.get('type', ''):
                print(f"⚠️ Step {i+1}: Still Buffering... ({fault['reason']})")
            else:
                print(f"❌ Step {i+1}: Unexpected State: {fault}")
        else:
            if i % 5 == 0: print(f"   Step {i+1}: Buffering ({i}/20)...")

    # 3. Inject Fault F5
    print("\n--- Phase 2: Injecting Fault F5 (Parallel Arc) ---")
    post_json("/config", {"injected_fault_type": "Parallel Arc Fault (F5)"})
    print("✅ Fault F5 Injected via /config")

    # 4. Verify Detection
    print("   Running steps to detect fault (Waiting for buffer to stabilize)...")
    detected_stable = False
    
    # Run 30 steps to flush 20-step buffer
    for i in range(30):
        res = post_json("/step", {"action": "Hold", "steps": 1})
        if not res: continue
        
        fault = res.get('fault_prediction')
        
        # Check only after buffer flush
        if i >= 24: 
            if fault and fault['detected']:
                print(f"✅ Step {i+1}: FAULT DETECTED! Type: {fault['type']}")
                if "F5" in fault['type']:
                    detected_stable = True
                else:
                    print(f"⚠️ Warning: Detected {fault['type']} instead of F5")
                    detected_stable = False # reset if unstable
                    
    if detected_stable:
        print("✅ Correct Fault Type (F5) Identified Stably.")
    else:
        print("❌ Failed to detect F5 stably after 30 steps.")

    # 5. Clear Fault
    print("\n--- Phase 3: Clearing Faults ---")
    post_json("/config", {"injected_fault_type": ""})
    print("✅ Fault Cleared via /config")
    
    # Run steps to see it return to healthy
    for i in range(5):
        post_json("/step", {"action": "Hold", "steps": 1})
    
    res = post_json("/step", {"action": "Hold", "steps": 1})
    fault = res.get('fault_prediction') if res else None
    
    if fault and not fault['detected']:
         print(f"✅ System Returned to Healthy: {fault['type']}")
    else:
         print(f"❌ System did not recover: {fault}")

if __name__ == "__main__":
    run_test()
