import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertTriangle, CheckCircle, Activity } from 'lucide-react';

const FaultDiagnostics = ({ faultData }) => {
    // faultData: { detected: boolean, type: string, confidence: number, reason: string }

    // Default valid data if null
    const data = faultData || { detected: false, type: "Unknown", confidence: 0, reason: "No data" };

    return (
        <div className="bg-helios-card rounded-xl p-6 border border-gray-800 h-full flex flex-col relative overflow-hidden group">
            <div className="absolute inset-0 bg-gradient-to-br from-helios-accent/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity pointer-events-none" />

            <h2 className="text-lg font-semibold mb-4 flex items-center gap-2 text-white relative z-10">
                <Activity className="text-helios-accent" /> AI Diagnostics
            </h2>

            <div className="flex-1 flex flex-col justify-center relative z-10">
                <AnimatePresence mode="wait">
                    {data.detected ? (
                        <motion.div
                            key="fault"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            transition={{ duration: 0.3 }}
                            className="bg-helios-danger/10 border border-helios-danger/50 rounded-lg p-5 shadow-[0_0_30px_rgba(239,68,68,0.2)]"
                        >
                            <div className="flex items-start gap-4">
                                <div className="p-3 bg-helios-danger/20 rounded-full shrink-0 animate-pulse">
                                    <AlertTriangle className="w-8 h-8 text-helios-danger" />
                                </div>
                                <div>
                                    <div className="flex items-center justify-between mb-1">
                                        <h3 className="text-xl font-bold text-white leading-none tracking-tight">{data.type}</h3>
                                    </div>

                                    <div className="flex items-center gap-3 text-sm font-semibold mb-3">
                                        <span className="text-helios-danger tracking-wider">CRITICAL ALERT</span>
                                        <span className="bg-helios-danger/20 text-helios-danger px-2 py-0.5 rounded text-xs border border-helios-danger/30">
                                            {(data.confidence * 100).toFixed(1)}% CONFIDENCE
                                        </span>
                                    </div>

                                    <p className="text-gray-300 text-sm leading-relaxed border-l-2 border-helios-danger/50 pl-3 mb-4">
                                        {data.reason}
                                    </p>

                                    {/* Explainability Bars - Fault Context */}
                                    {data.explainability && (
                                        <div className="space-y-3 pt-3 border-t border-white/10">
                                            <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Fault Contributors (SHAP)</h4>
                                            {Object.entries(data.explainability).map(([key, value]) => (
                                                <div key={key}>
                                                    <div className="flex justify-between text-xs mb-1 font-medium text-gray-400">
                                                        <span>{key}</span>
                                                        <span className={value > 0.4 ? "text-helios-danger font-bold" : "text-gray-500"}>
                                                            {(value * 100).toFixed(0)}% Impact
                                                        </span>
                                                    </div>
                                                    <div className="h-1.5 w-full bg-gray-700/50 rounded-full overflow-hidden">
                                                        <motion.div
                                                            initial={{ width: 0 }}
                                                            animate={{ width: `${value * 100}%` }}
                                                            transition={{ duration: 0.5, ease: "easeOut" }}
                                                            className={`h-full rounded-full ${value > 0.4 ? 'bg-helios-danger shadow-[0_0_10px_rgba(239,68,68,0.6)]' : 'bg-gray-600'}`}
                                                        />
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="normal"
                            initial={{ opacity: 0, x: 20 }}
                            animate={{ opacity: 1, x: 0 }}
                            exit={{ opacity: 0, x: -20 }}
                            transition={{ duration: 0.3 }}
                            className="bg-helios-success/10 border border-helios-success/50 rounded-lg p-5 shadow-[0_0_30px_rgba(34,197,94,0.1)]"
                        >
                            <div className="flex items-start gap-4">
                                <div className="p-3 bg-helios-success/20 rounded-full shrink-0">
                                    <CheckCircle className="w-8 h-8 text-helios-success" />
                                </div>
                                <div>
                                    <div className="flex items-center justify-between mb-1">
                                        <h3 className="text-xl font-bold text-white leading-none tracking-tight">{data.type || "F0: Nominal"}</h3>
                                    </div>

                                    <div className="flex items-center gap-3 text-sm font-semibold mb-3">
                                        <span className="text-helios-success tracking-wider uppercase">Optimal State</span>
                                        <span className="bg-helios-success/20 text-helios-success px-2 py-0.5 rounded text-xs border border-helios-success/30">
                                            {(data.confidence ? data.confidence * 100 : 99.9).toFixed(1)}% CONFIDENCE
                                        </span>
                                    </div>

                                    <p className="text-gray-300 text-sm leading-relaxed border-l-2 border-helios-success/50 pl-3 mb-4">
                                        {data.reason || "Operating parameters within normal limits."}
                                    </p>

                                    {/* Explainability Bars */}
                                    {data.explainability && (
                                        <div className="space-y-3 pt-3 border-t border-white/10">
                                            <h4 className="text-xs font-bold text-gray-500 uppercase tracking-widest mb-2">Signal Contribution</h4>
                                            {Object.entries(data.explainability).map(([key, value]) => (
                                                <div key={key}>
                                                    <div className="flex justify-between text-xs mb-1 font-medium text-gray-400">
                                                        <span>{key}</span>
                                                        <span className={value > 0.3 ? "text-helios-success" : "text-gray-500"}>
                                                            {(value * 100).toFixed(0)}%
                                                        </span>
                                                    </div>
                                                    <div className="h-1.5 w-full bg-gray-700/50 rounded-full overflow-hidden">
                                                        <motion.div
                                                            initial={{ width: 0 }}
                                                            animate={{ width: `${value * 100}%` }}
                                                            transition={{ duration: 0.5, ease: "easeOut" }}
                                                            className={`h-full rounded-full ${value > 0.3 ? 'bg-helios-success shadow-[0_0_10px_rgba(34,197,94,0.5)]' : 'bg-gray-600'}`}
                                                        />
                                                    </div>
                                                </div>
                                            ))}
                                        </div>
                                    )}
                                </div>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </div>
        </div>
    );
};

export default FaultDiagnostics;
