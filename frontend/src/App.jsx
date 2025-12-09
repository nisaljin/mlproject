import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import EnergyFlow from './components/EnergyFlow';
import TelemetryCharts from './components/TelemetryCharts';
import SystemLogs from './components/SystemLogs';
import DashboardMetrics from './components/DashboardMetrics';
import FaultDiagnostics from './components/FaultDiagnostics';
import API_URL from './config';

import { Menu } from 'lucide-react';

function App() {
  const [state, setState] = useState(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [config, setConfig] = useState({
    weather_factor: 1.0,
    string_status: [true, true, true]
  });
  const [isPlaying, setIsPlaying] = useState(false);
  const [speed, setSpeed] = useState(1); // Simulation speed (steps per frame)

  // Fetch initial state
  useEffect(() => {
    const fetchState = async () => {
      try {
        // 1. Get Config
        const configRes = await axios.get(`${API_URL}/state`);
        setConfig({
          weather_factor: configRes.data.weather_factor,
          string_status: configRes.data.string_status
        });

        // 2. Get Initial Dashboard State (Step 1) so UI renders immediately
        const stepRes = await axios.post(`${API_URL}/step`, {
          action: "Hold",
          steps: 1
        });
        setState(stepRes.data);

      } catch (err) {
        console.error("Failed to fetch initial state", err);
      }
    };
    fetchState();
  }, []);

  // Simulation Loop
  // WebSocket Connection
  const ws = React.useRef(null);

  useEffect(() => {
    // Convert HTTP URL to WS URL
    const wsUrl = API_URL.replace(/^http/, 'ws') + '/ws';
    ws.current = new WebSocket(wsUrl);

    ws.current.onopen = () => {
      console.log('Connected to WebSocket');
      // Sync initial state
      if (ws.current) {
        ws.current.send(JSON.stringify({
          action: isPlaying ? "start" : "stop",
          speed: speed
        }));
      }
    };

    ws.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      setState(data);
    };

    ws.current.onclose = () => {
      console.log('WebSocket Disconnected');
    };

    return () => {
      if (ws.current) ws.current.close();
    };
  }, []);

  // Sync Controls with WebSocket
  useEffect(() => {
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({
        action: isPlaying ? "start" : "stop",
        speed: speed
      }));
    }
  }, [isPlaying, speed]);

  // Remove old interval loop
  // (Replaced by the above WS logic)

  const handleReset = async () => {
    // Send reset via WS if available for immediate update
    if (ws.current && ws.current.readyState === WebSocket.OPEN) {
      ws.current.send(JSON.stringify({ action: "reset" }));
    } else {
      // Fallback
      await axios.post(`${API_URL}/reset`);
    }
    setIsPlaying(false);
  };

  // Mock fault data if backend not ready yet, but prioritize state
  const faultData = state?.fault_prediction || (state?.fault_detected ? {
    detected: true,
    type: "Line-Line Fault (Simulated)",
    confidence: 0.98,
    reason: "Abnormal current detected on String 2 while voltage remained nominal. Characteristic of Line-Line short in PV array."
  } : null);

  return (
    <div className="flex h-screen bg-helios-dark text-white overflow-hidden font-sans selection:bg-helios-accent/30">
      <Sidebar
        config={config}
        setConfig={setConfig}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
        speed={speed}
        setSpeed={setSpeed}
        onReset={handleReset}
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
      />

      <main className="flex-1 p-4 md:p-6 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-800 scrollbar-track-transparent relative w-full">
        {state ? (
          <div className="max-w-[1600px] mx-auto space-y-6">
            {/* Header */}
            <div className="flex justify-between items-end mb-2">
              <div className="flex items-center gap-3">
                <button
                  onClick={() => setIsSidebarOpen(true)}
                  className="md:hidden p-2 -ml-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-lg transition-colors"
                >
                  <Menu size={24} />
                </button>
                <div>
                  <h2 className="text-2xl md:text-3xl font-bold tracking-tight">Dashboard</h2>
                  <p className="text-gray-400 text-xs md:text-sm font-medium">Real-time Energy Optimization & Grid Stability</p>
                </div>
              </div>
              <div className="text-right">
                <div className="text-3xl font-mono font-bold text-helios-accent tracking-widest">
                  {new Date(state.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
                <div className="text-xs text-gray-500 font-bold uppercase tracking-wider">Simulation Time</div>
              </div>
            </div>

            {/* Key Metrics */}
            <DashboardMetrics state={state} />

            {/* Visualisation Grid */}
            <div className="grid grid-cols-1 md:grid-cols-12 gap-6 h-auto md:h-[420px]">
              {/* Energy Flow (Larger) */}
              <div className="col-span-1 md:col-span-8 h-[400px] md:h-full">
                <EnergyFlow state={state} isPlaying={isPlaying} />
              </div>
              {/* Fault Diagnostics (Smaller side panel) */}
              <div className="col-span-1 md:col-span-4 h-auto md:h-full min-h-[200px]">
                <FaultDiagnostics faultData={faultData} />
              </div>
            </div>

            {/* Charts */}
            <TelemetryCharts
              historySoc={state.history_soc}
              historySolar={state.history_solar}
              historyLoad={state.history_load}
              historyTimestamp={state.history_timestamp}
            />

            {/* Logs (Expandable) */}
            <SystemLogs
              logsGuardian={state.logs_guardian}
              logsPlanner={state.logs_planner}
            />

          </div>
        ) : (
          <div className="h-full flex flex-col items-center justify-center text-gray-500 space-y-4">
            <div className="w-20 h-20 rounded-full border-4 border-gray-800 border-t-helios-accent animate-spin" />
            <p className="text-lg font-medium animate-pulse">Initializing Helios System...</p>
          </div>
        )
        }
      </main >
    </div >
  );
}

export default App;
