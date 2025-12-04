import React, { useState, useEffect } from 'react';
import axios from 'axios';
import Sidebar from './components/Sidebar';
import EnergyFlow from './components/EnergyFlow';
import TelemetryCharts from './components/TelemetryCharts';
import SystemLogs from './components/SystemLogs';
import DashboardMetrics from './components/DashboardMetrics';
import API_URL from './config';

function App() {
  const [state, setState] = useState(null);
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
        const res = await axios.get(`${API_URL}/state`);
        setConfig({
          weather_factor: res.data.weather_factor,
          string_status: res.data.string_status
        });
      } catch (err) {
        console.error("Failed to fetch state", err);
      }
    };
    fetchState();
  }, []);

  // Simulation Loop
  useEffect(() => {
    let interval;
    if (isPlaying) {
      interval = setInterval(async () => {
        try {
          const response = await axios.post(`${API_URL}/step`, {
            action: "Hold",
            steps: speed
          });
          setState(response.data);
        } catch (error) {
          console.error("Step failed", error);
          setIsPlaying(false);
        }
      }, 100); // 100ms speed
    }
    return () => clearInterval(interval);
  }, [isPlaying]);

  const handleReset = async () => {
    await axios.post(`${API_URL}/reset`);
    setIsPlaying(false);
    setState(null);
  };

  return (
    <div className="flex h-screen bg-helios-dark text-white overflow-hidden font-sans">
      <Sidebar
        config={config}
        setConfig={setConfig}
        isPlaying={isPlaying}
        setIsPlaying={setIsPlaying}
        speed={speed}
        setSpeed={setSpeed}
        onReset={handleReset}
      />

      <main className="flex-1 p-8 overflow-y-auto">
        {state ? (
          <div className="max-w-6xl mx-auto space-y-8">
            {/* Header */}
            <div className="flex justify-between items-center mb-6">
              <div>
                <h2 className="text-2xl font-bold">Dashboard</h2>
                <p className="text-gray-400 text-sm">Real-time Energy Optimization & Grid Stability</p>
              </div>
              <div className="text-right">
                <div className="text-2xl font-mono font-bold text-helios-accent">
                  {new Date(state.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
                <div className="text-xs text-gray-500">Simulation Time</div>
              </div>
            </div>

            {/* Key Metrics */}
            <DashboardMetrics state={state} />

            {/* Energy Flow Visualization */}
            <EnergyFlow state={state} />

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
          <div className="h-full flex items-center justify-center text-gray-500">
            Press Play to Start Simulation
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
