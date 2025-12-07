import React, { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Sun, Home, Battery, Zap } from 'lucide-react';

const Card = ({ title, value, icon: Icon, color, subtext, active }) => (
    <div className={`relative z-10 flex flex-col items-center gap-1 p-2 rounded-xl border transition-all duration-300 min-w-[90px]
        ${active ? `bg-${color}/10 border-${color}/50 shadow-[0_0_10px_rgba(var(--color-${color}),0.15)]` : 'bg-helios-card border-gray-800 opacity-50'}`}>
        <div className={`w-10 h-10 rounded-full flex items-center justify-center border transition-colors duration-300 ${active ? `bg-${color}/20 border-${color} text-${color}` : 'bg-gray-800 border-gray-700 text-gray-500'}`}>
            <Icon className="w-5 h-5" />
        </div>
        <div className="text-center">
            <div className="text-lg font-bold text-white leading-tight">{value}</div>
            <div className="text-[9px] font-bold text-gray-400 uppercase">{title}</div>
        </div>
        {subtext && <div className={`text-[9px] font-bold px-1.5 py-0.5 rounded-full ${active ? `bg-${color}/20 text-${color}` : 'bg-gray-800 text-gray-500'}`}>{subtext}</div>}
    </div>
);

const EnergyFlow = ({ state, isPlaying }) => {
    const {
        generated_power,
        grid_consumption,
        battery_soc,
        grid_net,
        action
    } = state;

    // Logic
    const isCharging = battery_soc < 100 && (action === "Charge" || grid_net < 0 && battery_soc < 100);
    const isDischarging = action === "Discharge" || (grid_net === 0 && generated_power < grid_consumption);
    const hasSolar = generated_power > 10;
    const isImporting = grid_net < -10;
    const isExporting = grid_net > 10;
    const hasLoad = grid_consumption > 10;

    // SVG Coordinate System: 0 0 100 50 (Aspect Ratio 2:1)
    // We will use these percentages for both CSS Positioning and SVG Paths
    const P = {
        Solar: { x: 15, y: 20 },
        Home: { x: 85, y: 20 },
        Battery: { x: 50, y: 50 }, // Center
        Grid: { x: 50, y: 85 }    // Bottom
    };

    return (
        <div className="bg-helios-card rounded-xl p-4 shadow-xl border border-gray-800 relative h-full flex flex-col overflow-hidden">
            {/* Header */}
            <div className="flex items-center gap-2 mb-2 z-20">
                <div className="w-1 h-5 bg-helios-accent rounded-full" />
                <h2 className="text-md font-bold text-white tracking-wide">ENERGY FLOW</h2>
            </div>

            {/* Layout - Positioning driven by P coordinates */}
            <div className="flex-1 relative z-10 w-full h-full">

                {/* Solar (Top Left) */}
                <div className="absolute -translate-x-1/2 -translate-y-1/2" style={{ left: `${P.Solar.x}%`, top: `${P.Solar.y}%` }}>
                    <Card
                        title="Solar Array"
                        value={`${generated_power.toFixed(0)} W`}
                        icon={Sun}
                        color="helios-warning"
                        active={hasSolar}
                    />
                </div>

                {/* Home (Top Right) */}
                <div className="absolute -translate-x-1/2 -translate-y-1/2" style={{ left: `${P.Home.x}%`, top: `${P.Home.y}%` }}>
                    <Card
                        title="Home Load"
                        value={`${grid_consumption.toFixed(0)} W`}
                        icon={Home}
                        color="helios-blue"
                        active={hasLoad}
                    />
                </div>

                {/* Battery (Center) */}
                <div className="absolute -translate-x-1/2 -translate-y-1/2 z-20" style={{ left: `${P.Battery.x}%`, top: `${P.Battery.y}%` }}>
                    <Card
                        title="Battery"
                        value={`${battery_soc.toFixed(1)}%`}
                        icon={Battery}
                        color={isCharging ? "helios-success" : isDischarging ? "helios-danger" : "gray-500"}
                        subtext={action}
                        active={true}
                    />
                </div>

                {/* Grid (Bottom Center) - Restored */}
                <div className="absolute -translate-x-1/2 -translate-y-1/2" style={{ left: `${P.Grid.x}%`, top: `${P.Grid.y}%` }}>
                    <Card
                        title={isExporting ? "Grid Export" : isImporting ? "Grid Import" : "Grid Idle"}
                        value={`${Math.abs(grid_net).toFixed(0)} W`}
                        icon={Zap}
                        color={isExporting ? "purple-500" : isImporting ? "helios-blue" : "gray-500"}
                        active={Math.abs(grid_net) > 10}
                    />
                </div>

            </div>

            {/* Animation Layer - Strictly conditioned on isPlaying */}
            {isPlaying && (
                <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
                    <defs>
                        {/* Define gradients if needed */}
                    </defs>

                    {/* Solar -> Battery (TopLeft -> Center) */}
                    {isCharging && (
                        <FlowPath
                            d={`M ${P.Solar.x + 5} ${P.Solar.y + 5} Q 30 50 ${P.Battery.x - 6} ${P.Battery.y - 6}`}
                            color="#fbbf24" // Amber-400
                        />
                    )}

                    {/* Battery -> Home (Center -> TopRight) */}
                    {isDischarging && (
                        <FlowPath
                            d={`M ${P.Battery.x + 6} ${P.Battery.y - 6} Q 70 50 ${P.Home.x - 5} ${P.Home.y + 5}`}
                            color="#f87171" // Red-400
                        />
                    )}

                    {/* Solar -> Home (Direct Top Arch) 
                        TopLeft -> TopRight
                    */}
                    {hasSolar && hasLoad && (
                        <FlowPath
                            d={`M ${P.Solar.x + 5} ${P.Solar.y} Q 50 5 ${P.Home.x - 5} ${P.Home.y}`}
                            color="#fbbf24"
                            dashed
                        />
                    )}

                    {/* Grid Import (Grid -> Battery/System) 
                        Bottom -> Center
                    */}
                    {isImporting && (
                        <FlowPath
                            d={`M ${P.Grid.x} ${P.Grid.y - 10} L ${P.Battery.x} ${P.Battery.y + 12}`}
                            color="#38bdf8" // Sky-400
                        />
                    )}

                    {/* Solar -> Grid Export (Solar -> Grid)
                        TopLeft -> Bottom (Curved around left side)
                        "The Purple One" - Now logically flowing from Solar to Grid
                    */}
                    {isExporting && (
                        <FlowPath
                            d={`M ${P.Solar.x} ${P.Solar.y + 10} Q 10 60 ${P.Grid.x - 8} ${P.Grid.y}`}
                            color="#a855f7" // Purple-500
                        />
                    )}
                </svg>
            )}
        </div>
    );
};

// Refined "High Tech" Path Animation
// - Background Rail: Thin, low opacity
// - Moving Particle: Crisp circle + trailing opacity, no heavy blur
const FlowPath = ({ d, color, dashed }) => (
    <>
        {/* Rail */}
        <path
            d={d}
            stroke={color}
            strokeWidth="1"
            strokeOpacity="0.2"
            fill="none"
            strokeDasharray={dashed ? "3 3" : ""}
        />

        {/* Moving Energy Packet (Crisp) */}
        <motion.circle r="2" fill={color}>
            <motion.animateMotion
                dur="1.5s"
                repeatCount="indefinite"
                path={d}
                rotate="auto"
                calcMode="linear"
            />
        </motion.circle>

        {/* Trailing "Comet" Dash (Optional, subtle) */}
        <motion.path
            d={d}
            stroke={color}
            strokeWidth="1.5"
            fill="none"
            initial={{ pathLength: 0, pathOffset: 0, opacity: 0 }}
            animate={{
                pathLength: 0.3,
                pathOffset: 1,
                opacity: [0, 1, 0]
            }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
        />
    </>
);

export default EnergyFlow;
