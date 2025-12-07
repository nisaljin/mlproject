import React from 'react';
import { Play, Pause, RotateCcw, Zap, Cloud, Sun, AlertTriangle, Settings, Activity, X } from 'lucide-react';
import axios from 'axios';
import API_URL from '../config';

const Sidebar = ({ config, setConfig, isPlaying, setIsPlaying, speed, setSpeed, onReset, isOpen, onClose }) => {
    const [targetString, setTargetString] = React.useState("String 1");
    const [faultType, setFaultType] = React.useState("Line-Line (F1)");

    const handleInjectFault = async (faultType, targetString) => {
        // In a real app, we'd send this to the backend. 
        // For now, let's toggle the string status in the config.
        const newStatus = [...config.string_status];
        const targetIdx = parseInt(targetString.split(' ')[1]) - 1;
        newStatus[targetIdx] = false; // Trip the string

        try {
            await axios.post(`${API_URL}/config`, {
                string_status: newStatus,
                injected_fault_type: faultType
            });
            setConfig({ ...config, string_status: newStatus });
        } catch (err) {
            console.error("Failed to inject fault", err);
        }
    };

    const handleClearFaults = async () => {
        const newStatus = [true, true, true];
        try {
            await axios.post(`${API_URL}/config`, {
                string_status: newStatus,
                injected_fault_type: ""
            });
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
            await axios.post(`${API_URL}/config`, { weather_factor: factor });
            setConfig({ ...config, weather_factor: factor });
        } catch (err) {
            console.error("Failed to set weather", err);
        }
    };

    return (
        <>
            {/* Mobile Overlay */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 md:hidden animate-in fade-in duration-200"
                    onClick={onClose}
                />
            )}

            {/* Sidebar Container */}
            <div className={`
                fixed inset-y-0 left-0 z-50 w-72 bg-helios-card border-r border-gray-800 flex flex-col shadow-2xl transition-transform duration-300 ease-in-out
                md:static md:translate-x-0 md:h-screen md:shadow-none
                ${isOpen ? 'translate-x-0' : '-translate-x-full'}
            `}>
                {/* Branding */}
                <div className="p-6 border-b border-gray-800 flex justify-between items-center">
                    <div className="flex items-center gap-3">
                        <div className="p-2 bg-helios-accent/20 rounded-lg">
                            <Zap className="text-helios-accent w-6 h-6" />
                        </div>
                        <div>
                            <h1 className="text-xl font-bold tracking-wider text-white">HELIOS AI</h1>
                            <p className="text-[10px] text-gray-500 font-mono tracking-widest uppercase">Grid Stability</p>
                        </div>
                    </div>
                    {/* Mobile Close Button */}
                    <button
                        onClick={onClose}
                        className="md:hidden text-gray-400 hover:text-white transition-colors"
                    >
                        <X size={24} />
                    </button>
                </div>

                <div className="flex-1 overflow-y-auto p-6 space-y-8">
                    {/* Controls */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-xs font-bold text-gray-500 uppercase tracking-wider">
                            <Settings className="w-3 h-3" /> Simulation Control
                        </div>

                        <div className="grid grid-cols-2 gap-2">
                            <button
                                onClick={() => setIsPlaying(!isPlaying)}
                                className={`flex items-center justify-center gap-2 p-3 rounded-lg font-bold transition-all duration-200 border ${isPlaying ? 'bg-helios-warning/20 border-helios-warning text-helios-warning' : 'bg-helios-success/20 border-helios-success text-helios-success hover:bg-helios-success/30'}`}
                            >
                                {isPlaying ? <><Pause size={18} /> PAUSE</> : <><Play size={18} /> RUN</>}
                            </button>
                            <button
                                onClick={onReset}
                                className="flex items-center justify-center gap-2 p-3 bg-gray-800 border border-gray-700 rounded-lg text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
                            >
                                <RotateCcw size={18} /> RESET
                            </button>
                        </div>

                        {/* Speed Slider */}
                        <div className="bg-gray-900/50 p-4 rounded-lg border border-gray-800">
                            <div className="flex justify-between text-xs text-gray-400 mb-3 font-medium">
                                <span>Time Dilation</span>
                                <span className="text-helios-accent">{speed}x</span>
                            </div>
                            <input
                                type="range"
                                min="1"
                                max="100"
                                value={speed}
                                onChange={(e) => setSpeed(parseInt(e.target.value))}
                                className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-helios-accent hover:accent-helios-accent/80 transition-all"
                            />
                        </div>
                    </div>

                    {/* Scenarios */}
                    <div className="space-y-4">
                        <div className="flex items-center gap-2 text-xs font-bold text-gray-500 uppercase tracking-wider">
                            <Activity className="w-3 h-3" /> Scenarios
                        </div>

                        {/* Weather */}
                        <div className="space-y-2">
                            <label className="text-xs text-gray-400 font-medium ml-1">Environment</label>
                            <div className="relative">
                                <select
                                    onChange={handleWeatherChange}
                                    className="w-full bg-gray-800 text-white border border-gray-700 rounded-lg p-3 pl-10 appearance-none text-sm focus:ring-1 focus:ring-helios-accent focus:border-helios-accent transition-all outline-none"
                                >
                                    <option>Sunny (Normal)</option>
                                    <option>Cloudy (Low Solar)</option>
                                    <option>Stormy (No Solar)</option>
                                </select>
                                <Sun className="absolute left-3 top-3.5 text-helios-warning w-4 h-4 pointer-events-none" />
                            </div>
                        </div>

                        {/* Fault Injection */}
                        <div className="space-y-2 pt-2">
                            <label className="text-xs text-gray-400 font-medium ml-1">Fault Simulation</label>
                            <div className="p-4 bg-gray-800/50 rounded-xl border border-gray-700/50 space-y-3">
                                <select
                                    value={faultType}
                                    onChange={(e) => setFaultType(e.target.value)}
                                    className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2.5 text-sm text-gray-300 focus:border-helios-danger focus:ring-1 focus:ring-helios-danger outline-none"
                                >
                                    <option>Line-Line (F1)</option>
                                    <option>Line-Ground (F2)</option>
                                    <option>Open Circuit (F3)</option>
                                    <option>Series Arc (F4)</option>
                                    <option>Parallel Arc (F5)</option>
                                    <option>Partial Shading (F6)</option>
                                    <option>Symmetric (F7)</option>
                                </select>
                                <select
                                    value={targetString}
                                    onChange={(e) => setTargetString(e.target.value)}
                                    className="w-full bg-gray-900 border border-gray-700 rounded-lg p-2.5 text-sm text-gray-300 outline-none"
                                >
                                    <option>String 1</option>
                                    <option>String 2</option>
                                    <option>String 3</option>
                                </select>
                                <button
                                    onClick={() => handleInjectFault(faultType, targetString)}
                                    className="w-full py-2.5 bg-helios-danger text-white rounded-lg text-sm font-bold hover:bg-red-600 transition-colors flex items-center justify-center gap-2 shadow-lg shadow-red-900/20"
                                >
                                    <AlertTriangle size={16} /> INJECT FAULT
                                </button>
                                <button
                                    onClick={handleClearFaults}
                                    className="w-full py-2 bg-gray-700 text-xs font-semibold rounded-lg hover:bg-gray-600 text-gray-300 transition-colors"
                                >
                                    Normalize System
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="p-6 border-t border-gray-800">
                    <div className="text-[10px] text-gray-600 font-medium text-center">
                        Helios Grid Sentinel v2.1
                    </div>
                </div>
            </div>
        </>
    );
};

export default Sidebar;
