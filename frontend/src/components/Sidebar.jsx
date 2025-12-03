import React from 'react';
import { Play, Pause, RotateCcw, Zap, Cloud, Sun, CloudRain, AlertTriangle } from 'lucide-react';
import axios from 'axios';

const Sidebar = ({ config, setConfig, isPlaying, setIsPlaying, speed, setSpeed, onReset }) => {
    const [targetString, setTargetString] = React.useState("String 1");
    const [faultType, setFaultType] = React.useState("Line-Line (F1)");

    const handleInjectFault = async (faultType, targetString) => {
        // In a real app, we'd send this to the backend. 
        // For now, let's toggle the string status in the config.
        const newStatus = [...config.string_status];
        const targetIdx = parseInt(targetString.split(' ')[1]) - 1;
        newStatus[targetIdx] = false; // Trip the string

        try {
            await axios.post('http://localhost:8001/config', { string_status: newStatus });
            setConfig({ ...config, string_status: newStatus });
        } catch (err) {
            console.error("Failed to inject fault", err);
        }
    };

    const handleClearFaults = async () => {
        const newStatus = [true, true, true];
        try {
            await axios.post('http://localhost:8001/config', { string_status: newStatus });
            setConfig({ ...config, string_status: newStatus });
        } catch (err) {
            console.error("Failed to clear faults", err);
        }
    };

    const handleWeatherChange = async (e) => {
        const weather = e.target.value;
        let factor = 1.0;
        if (weather.includes("Cloudy")) factor = 0.4;
        if (weather.includes("Stormy")) factor = 0.1;

        try {
            await axios.post('http://localhost:8001/config', { weather_factor: factor });
            setConfig({ ...config, weather_factor: factor });
        } catch (err) {
            console.error("Failed to set weather", err);
        }
    };

    return (
        <div className="w-64 bg-helios-card h-screen p-4 border-r border-gray-700 flex flex-col gap-6">
            <div className="flex items-center gap-2 mb-4">
                <Zap className="text-helios-accent w-6 h-6" />
                <h1 className="text-xl font-bold tracking-wider">HELIOS AI</h1>
            </div>

            {/* Controls */}
            <div className="space-y-3">
                <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Controls</h2>
                <div className="flex gap-2">
                    <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className={`flex-1 flex items-center justify-center gap-2 p-2 rounded font-medium transition-colors ${isPlaying ? 'bg-helios-warning text-black' : 'bg-helios-success text-black'}`}
                    >
                        {isPlaying ? <><Pause size={16} /> Pause</> : <><Play size={16} /> Play</>}
                    </button>
                    <button
                        onClick={onReset}
                        className="p-2 bg-gray-700 rounded hover:bg-gray-600 transition-colors"
                    >
                        <RotateCcw size={16} />
                    </button>
                </div>

                {/* Speed Slider */}
                <div className="pt-2">
                    <div className="flex justify-between text-xs text-gray-400 mb-1">
                        <span>Speed</span>
                        <span>{speed}x</span>
                    </div>
                    <input
                        type="range"
                        min="1"
                        max="100"
                        value={speed}
                        onChange={(e) => setSpeed(parseInt(e.target.value))}
                        className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-helios-accent"
                    />
                </div>
            </div>

            {/* Fault Injection */}
            <div className="space-y-3">
                <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Fault Injection</h2>
                <div className="p-3 bg-gray-800 rounded border border-gray-700 space-y-3">
                    <select
                        value={faultType}
                        onChange={(e) => setFaultType(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-sm text-gray-300"
                    >
                        <option>Line-Line (F1)</option>
                        <option>Line-Ground (F2)</option>
                        <option>Symmetric (F7)</option>
                    </select>
                    <select
                        value={targetString}
                        onChange={(e) => setTargetString(e.target.value)}
                        className="w-full bg-gray-900 border border-gray-700 rounded p-2 text-sm text-gray-300"
                    >
                        <option>String 1</option>
                        <option>String 2</option>
                        <option>String 3</option>
                    </select>
                    <button
                        onClick={() => handleInjectFault(faultType, targetString)}
                        className="w-full py-2 bg-helios-danger text-white rounded text-sm font-medium hover:bg-red-600 transition-colors flex items-center justify-center gap-2"
                    >
                        <AlertTriangle size={14} /> Inject Fault
                    </button>
                    <button
                        onClick={handleClearFaults}
                        className="w-full py-1 bg-gray-700 text-xs rounded hover:bg-gray-600"
                    >
                        Clear Faults
                    </button>
                </div>
            </div>

            {/* Weather */}
            <div className="space-y-3">
                <h2 className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Weather Scenario</h2>
                <div className="relative">
                    <select
                        onChange={handleWeatherChange}
                        className="w-full bg-gray-800 border border-gray-700 rounded p-2 pl-8 appearance-none text-sm"
                    >
                        <option>Sunny (Normal)</option>
                        <option>Cloudy (Low Solar)</option>
                        <option>Stormy (No Solar)</option>
                    </select>
                    <Sun className="absolute left-2 top-2.5 text-yellow-500 w-4 h-4 pointer-events-none" />
                </div>
            </div>

            <div className="mt-auto text-xs text-gray-500">
                v2.0.0 React + FastAPI
            </div>
        </div>
    );
};

export default Sidebar;
