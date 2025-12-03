import React from 'react';
import { Activity, Battery, Zap, ArrowRight, TrendingUp, TrendingDown } from 'lucide-react';

const MetricCard = ({ title, value, subValue, icon: Icon, color }) => (
    <div className="bg-helios-card p-4 rounded-xl border border-gray-800 shadow-lg flex items-center gap-4">
        <div className={`p-3 rounded-lg bg-opacity-10 ${color.bg} ${color.text}`}>
            <Icon size={24} />
        </div>
        <div>
            <h3 className="text-gray-400 text-xs font-medium uppercase tracking-wider">{title}</h3>
            <div className="text-xl font-bold text-white">{value}</div>
            {subValue && <div className="text-xs text-gray-500 mt-1">{subValue}</div>}
        </div>
    </div>
);

const DashboardMetrics = ({ state }) => {
    if (!state) return null;

    const {
        predicted_load,
        grid_consumption,
        action,
        grid_net,
        battery_soc,
        generated_power
    } = state;

    // Determine Action Color
    let actionColor = { bg: 'bg-gray-500', text: 'text-gray-500' };
    if (action === 'Charge') actionColor = { bg: 'bg-green-500', text: 'text-green-500' };
    if (action === 'Discharge') actionColor = { bg: 'bg-red-500', text: 'text-red-500' };
    if (action === 'Hold') actionColor = { bg: 'bg-yellow-500', text: 'text-yellow-500' };

    // Determine Grid Status
    const isExporting = grid_net < 0;
    const gridValue = `${Math.abs(grid_net).toFixed(0)} W`;
    const gridLabel = isExporting ? 'Exporting to Grid' : 'Importing from Grid';
    const gridColor = isExporting ? { bg: 'bg-green-500', text: 'text-green-500' } : { bg: 'bg-blue-500', text: 'text-blue-500' };

    // Load Prediction Accuracy
    const loadDiff = predicted_load - grid_consumption;
    const loadDiffLabel = loadDiff > 0 ? `+${loadDiff.toFixed(0)} W Over` : `${loadDiff.toFixed(0)} W Under`;

    return (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
            {/* AI Action */}
            <MetricCard
                title="AI Decision"
                value={action}
                subValue="Optimal Strategy"
                icon={Activity}
                color={actionColor}
            />

            {/* Predicted Load */}
            <MetricCard
                title="Predicted Load"
                value={`${predicted_load.toFixed(0)} W`}
                subValue={`Actual: ${grid_consumption.toFixed(0)} W (${loadDiffLabel})`}
                icon={TrendingUp}
                color={{ bg: 'bg-purple-500', text: 'text-purple-500' }}
            />

            {/* Grid Interaction */}
            <MetricCard
                title="Grid Net"
                value={gridValue}
                subValue={gridLabel}
                icon={Zap}
                color={gridColor}
            />

            {/* Solar Generation */}
            <MetricCard
                title="Solar Generation"
                value={`${generated_power.toFixed(0)} W`}
                subValue="Current Output"
                icon={Zap}
                color={{ bg: 'bg-yellow-500', text: 'text-yellow-500' }}
            />
        </div>
    );
};

export default DashboardMetrics;
