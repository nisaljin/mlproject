import React, { useState } from 'react';
import { ChevronDown, ChevronRight, Activity, Server } from 'lucide-react';

const LogItem = ({ log }) => {
    const [isExpanded, setIsExpanded] = useState(false);

    return (
        <div className="border-b border-gray-800 last:border-0">
            <div
                className="flex items-center gap-2 p-2 cursor-pointer hover:bg-white/5 transition-colors"
                onClick={() => setIsExpanded(!isExpanded)}
            >
                {isExpanded ? <ChevronDown size={14} className="text-gray-500" /> : <ChevronRight size={14} className="text-gray-500" />}
                <span className="text-xs font-mono text-gray-400">{log.timestamp}</span>
                <span className={`text-xs font-bold ${log.module === 'Guardian' ? 'text-green-400' : 'text-blue-400'}`}>
                    [{log.module}]
                </span>
                <span className="text-xs text-gray-300 truncate flex-1">{log.output}</span>
            </div>

            {isExpanded && (
                <div className="p-2 pl-8 bg-black/20 text-xs font-mono text-gray-400 space-y-2">
                    <div>
                        <div className="text-gray-500 uppercase text-[10px] tracking-wider mb-1">Input Parameters</div>
                        <pre className="bg-black/40 p-2 rounded overflow-x-auto">
                            {JSON.stringify(log.input, null, 2)}
                        </pre>
                    </div>
                    <div>
                        <div className="text-gray-500 uppercase text-[10px] tracking-wider mb-1">Model Response</div>
                        <div className="text-white">{log.output}</div>
                    </div>
                </div>
            )}
        </div>
    );
};

const SystemLogs = ({ logsGuardian, logsPlanner }) => {
    return (
        <div className="grid grid-cols-2 gap-6">
            {/* Guardian Logs */}
            <div className="bg-helios-card rounded-xl border border-gray-800 flex flex-col h-64">
                <div className="p-3 border-b border-gray-800 flex items-center gap-2 bg-gray-900/50 rounded-t-xl">
                    <Activity size={16} className="text-green-500" />
                    <h3 className="text-sm font-semibold text-gray-200">Guardian Logs (Stability)</h3>
                </div>
                <div className="flex-1 overflow-y-auto custom-scrollbar">
                    {logsGuardian.map((log, i) => (
                        <LogItem key={i} log={log} />
                    ))}
                </div>
            </div>

            {/* Planner Logs */}
            <div className="bg-helios-card rounded-xl border border-gray-800 flex flex-col h-64">
                <div className="p-3 border-b border-gray-800 flex items-center gap-2 bg-gray-900/50 rounded-t-xl">
                    <Server size={16} className="text-blue-500" />
                    <h3 className="text-sm font-semibold text-gray-200">Planner Logs (Optimization)</h3>
                </div>
                <div className="flex-1 overflow-y-auto custom-scrollbar">
                    {logsPlanner.map((log, i) => (
                        <LogItem key={i} log={log} />
                    ))}
                </div>
            </div>
        </div>
    );
};

export default SystemLogs;
