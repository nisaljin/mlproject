import React from 'react';
import { motion } from 'framer-motion';
import { Sun, Home, Battery, Zap } from 'lucide-react';

const EnergyFlow = ({ state }) => {
    const { generated_power, grid_consumption, battery_soc, energy_flow, grid_net, action } = state;

    // Determine Flow Directions
    const isCharging = energy_flow > 0;
    const isDischarging = energy_flow < 0;
    const isExporting = grid_net > 0;
    const isImporting = grid_net < 0;

    return (
        <div className="bg-helios-card rounded-xl p-6 shadow-lg border border-gray-800 relative overflow-hidden">
            <h2 className="text-lg font-semibold mb-8 flex items-center gap-2">
                <Zap className="text-helios-accent" /> Real-time Energy Flow
            </h2>

            <div className="grid grid-cols-3 gap-8 text-center relative z-10">

                {/* Solar */}
                <div className="flex flex-col items-center gap-2">
                    <div className="w-16 h-16 rounded-full bg-yellow-500/20 flex items-center justify-center border-2 border-yellow-500">
                        <Sun className="w-8 h-8 text-yellow-500" />
                    </div>
                    <div className="text-2xl font-bold text-white">{generated_power.toFixed(0)} W</div>
                    <div className="text-xs text-gray-400">Solar Array</div>
                </div>

                {/* Battery (Center) */}
                <div className="flex flex-col items-center gap-2 mt-12">
                    <div className={`w-20 h-20 rounded-xl flex items-center justify-center border-2 transition-colors ${isCharging ? 'bg-green-500/20 border-green-500' : isDischarging ? 'bg-red-500/20 border-red-500' : 'bg-gray-700/20 border-gray-600'}`}>
                        <Battery className={`w-10 h-10 ${isCharging ? 'text-green-500' : isDischarging ? 'text-red-500' : 'text-gray-400'}`} />
                    </div>
                    <div className="text-3xl font-bold text-white">{battery_soc.toFixed(1)}%</div>
                    <div className={`text-xs font-bold px-2 py-1 rounded ${action === 'Charge' ? 'bg-green-500/20 text-green-400' : action === 'Discharge' ? 'bg-red-500/20 text-red-400' : 'bg-gray-700 text-gray-400'}`}>
                        {action.toUpperCase()}
                    </div>
                </div>

                {/* Home */}
                <div className="flex flex-col items-center gap-2">
                    <div className="w-16 h-16 rounded-full bg-blue-500/20 flex items-center justify-center border-2 border-blue-500">
                        <Home className="w-8 h-8 text-blue-500" />
                    </div>
                    <div className="text-2xl font-bold text-white">{grid_consumption.toFixed(0)} W</div>
                    <div className="text-xs text-gray-400">Home Load</div>
                </div>

                {/* Grid (Bottom) */}
                <div className="col-start-2 flex flex-col items-center gap-2 mt-4">
                    <div className={`w-16 h-16 rounded-full flex items-center justify-center border-2 ${isExporting ? 'bg-purple-500/20 border-purple-500' : isImporting ? 'bg-orange-500/20 border-orange-500' : 'bg-gray-700/20 border-gray-600'}`}>
                        <Zap className={`w-8 h-8 ${isExporting ? 'text-purple-500' : isImporting ? 'text-orange-500' : 'text-gray-400'}`} />
                    </div>
                    <div className="text-xl font-bold text-white">{Math.abs(grid_net).toFixed(0)} W</div>
                    <div className="text-xs text-gray-400">{isExporting ? 'Exporting' : isImporting ? 'Importing' : 'Balanced'}</div>
                </div>

            </div>

            {/* Animated Lines (SVG Overlay) */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-50" viewBox="0 0 100 100" preserveAspectRatio="none">
                {/* Solar to Battery (Left to Center) */}
                {generated_power > 0 && (
                    <motion.path
                        d="M 16 30 Q 33 30 50 55"
                        fill="none"
                        stroke="#eab308"
                        strokeWidth="0.5"
                        strokeDasharray="2 2"
                        animate={{ strokeDashoffset: [0, -4] }}
                        transition={{ repeat: Infinity, duration: 0.5, ease: "linear" }}
                    />
                )}

                {/* Battery to Home (Center to Right) */}
                {isDischarging && (
                    <motion.path
                        d="M 50 55 Q 66 30 84 30"
                        fill="none"
                        stroke="#ef4444"
                        strokeWidth="0.5"
                        strokeDasharray="2 2"
                        animate={{ strokeDashoffset: [0, -4] }}
                        transition={{ repeat: Infinity, duration: 0.5, ease: "linear" }}
                    />
                )}

                {/* Solar to Home (Direct Arch) */}
                {generated_power > 0 && (
                    <motion.path
                        d="M 16 25 Q 50 5 84 25"
                        fill="none"
                        stroke="#eab308"
                        strokeWidth="0.3"
                        strokeDasharray="1 1"
                        animate={{ strokeDashoffset: [0, -4] }}
                        transition={{ repeat: Infinity, duration: 1, ease: "linear" }}
                    />
                )}
            </svg>
        </div>
    );
};

export default EnergyFlow;
