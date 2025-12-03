import React from 'react';
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from 'recharts';

const TelemetryCharts = ({ historySoc, historySolar, historyLoad, historyTimestamp }) => {
    // Combine data for charts
    const data = historySoc.map((soc, i) => ({
        time: historyTimestamp ? historyTimestamp[i] : i,
        soc: soc,
        solar: historySolar[i],
        load: historyLoad[i]
    }));

    return (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 h-64">
            {/* Power Telemetry */}
            <div className="bg-helios-card p-4 rounded-xl border border-gray-800 shadow-lg flex flex-col">
                <h3 className="text-gray-400 text-sm font-medium mb-2">Power Telemetry (W)</h3>
                <div className="flex-1 min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={data}>
                            <defs>
                                <linearGradient id="colorSolar" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#eab308" stopOpacity={0.8} />
                                    <stop offset="95%" stopColor="#eab308" stopOpacity={0} />
                                </linearGradient>
                                <linearGradient id="colorLoad" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                            <XAxis
                                dataKey="time"
                                stroke="#9ca3af"
                                fontSize={10}
                                tickFormatter={(val, i) => i % 10 === 0 ? val : ''}
                                tickLine={false}
                                axisLine={false}
                            />
                            <YAxis stroke="#9ca3af" fontSize={10} tickLine={false} axisLine={false} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', borderRadius: '8px' }}
                                itemStyle={{ color: '#e5e7eb' }}
                                formatter={(value) => value.toFixed(1)}
                                labelStyle={{ color: '#9ca3af' }}
                            />
                            <Legend verticalAlign="top" height={36} />
                            <Area type="monotone" dataKey="solar" stroke="#eab308" fillOpacity={1} fill="url(#colorSolar)" name="Solar Gen" />
                            <Area type="monotone" dataKey="load" stroke="#3b82f6" fillOpacity={1} fill="url(#colorLoad)" name="Grid Load" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Battery SoC */}
            <div className="bg-helios-card p-4 rounded-xl border border-gray-800 shadow-lg flex flex-col">
                <h3 className="text-gray-400 text-sm font-medium mb-2">Battery SoC (%)</h3>
                <div className="flex-1 min-h-0">
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={data}>
                            <defs>
                                <linearGradient id="colorSoC" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#22c55e" stopOpacity={0.8} />
                                    <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                            <XAxis
                                dataKey="time"
                                stroke="#9ca3af"
                                fontSize={10}
                                tickFormatter={(val, i) => i % 10 === 0 ? val : ''}
                                tickLine={false}
                                axisLine={false}
                            />
                            <YAxis domain={[0, 100]} stroke="#9ca3af" fontSize={10} tickLine={false} axisLine={false} />
                            <Tooltip
                                contentStyle={{ backgroundColor: '#1f2937', borderColor: '#374151', borderRadius: '8px' }}
                                itemStyle={{ color: '#e5e7eb' }}
                                formatter={(value) => `${value.toFixed(1)}%`}
                                labelStyle={{ color: '#9ca3af' }}
                            />
                            <Legend verticalAlign="top" height={36} />
                            <Area type="monotone" dataKey="soc" stroke="#22c55e" fillOpacity={1} fill="url(#colorSoC)" name="State of Charge" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>
        </div>
    );
};

export default TelemetryCharts;
