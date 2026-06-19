import React, { useState } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { ShieldAlert, Activity, Info, AlertTriangle, LayoutDashboard, Globe } from 'lucide-react';
import { AreaChart, Area, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell, Legend } from 'recharts';
// --- HELPER FUNCTIONS ---
const getRegimeName = (index) => {
  const names = {
    0: "Bear (High Volatility)",
    1: "Chop (Sideways/Transition)",
    2: "Bull (Low Volatility)",
    3: "Shock (Tail-Risk Event)"
  };
  return names[index] || `Regime ${index}`;
};

const getExecutionMandate = (assetClass, exposure, isBlackSwan) => {
  // 1. Black Swan Override
  if (isBlackSwan) {
    if (assetClass === "COMMODITY_PRECIOUS_METAL") {
      return "MANDATE: Tail-risk event detected. Flight-to-safety protocol engaged. Maximize safe-haven commodity exposure.";
    }
    return "MANDATE: Immediate capital preservation required. Liquidate high-beta positions. Move to cash equivalents and deploy tail-risk hedges.";
  }

  // 2. Equity & Crypto (Return Assets)
  if (assetClass === "EQUITY" || assetClass === "CRYPTO") {
    if (exposure >= 0.7) return "MANDATE: High-conviction structural trend detected. Maintain unhedged directional exposure and capture risk premium.";
    if (exposure >= 0.3) return "MANDATE: Transition regime. Volatility expansion likely. Scale down gross exposure and tighten trailing stops.";
    return "MANDATE: Structural bear/whipsaw risk. Directional conviction statistically insignificant. Move to cash or delta-neutral strategies.";
  }

  // 3. Fixed Income (Protection Asset)
  if (assetClass === "FIXED_INCOME") {
    if (exposure >= 0.5) return "MANDATE: Stable macro yield environment. Deploy capital into duration.";
    return "MANDATE: Hostile rate/inflation regime detected. Liquidate duration exposure to preserve capital.";
  }

  // 4. Gold (Safe Haven Hedge)
  if (assetClass === "COMMODITY_PRECIOUS_METAL") {
    if (exposure >= 0.8) return "MANDATE: Institutional safe-haven flows detected. Maintain maximum defensive commodity exposure.";
    return "MANDATE: Sideways consolidation. Capital bleed risk high. Rotate out of safe-havens.";
  }

  // 5. FX / Default
  return exposure >= 0.5 
    ? "MANDATE: Favorable structural environment. Maintain targeted allocation." 
    : "MANDATE: Defensive posture required. Limit gross exposure.";
};

const availableAssets = [
  { value: '^GSPC', label: 'S&P 500 Index (^GSPC)' },
  { value: '^NSEI', label: 'Nifty 50 Index (^NSEI)' },
  { value: 'BTC-USD', label: 'Bitcoin (BTC-USD)' },
  { value: 'TLT', label: '20+ Yr Treasury Bonds (TLT)' },
  { value: 'GC=F', label: 'Gold Futures (GC=F)' },
  { value: 'EURUSD=X', label: 'EUR/USD Forex' }
];

// --- COMPONENTS ---
const Navigation = () => {
  const location = useLocation();
  const navClass = (path) => `flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${location.pathname === path ? 'bg-blue-600/20 text-blue-400' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'}`;
  
  return (
    <nav className="border-b border-slate-800 bg-slate-950 px-8 py-4 sticky top-0 z-50">
      <div className="max-w-6xl mx-auto flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <div className="bg-blue-600/20 p-2 rounded-lg border border-blue-500/30">
             <Activity className="h-5 w-5 text-blue-500" />
          </div>
          <span className="text-xl font-bold tracking-tight text-slate-100">Event<span className="text-blue-500">Horizon</span></span>
        </div>
        <div className="flex space-x-2 bg-slate-900 p-1 rounded-xl border border-slate-800">
          <Link to="/" className={navClass('/')}><LayoutDashboard className="h-4 w-4" /> <span>Terminal</span></Link>
          <Link to="/screener" className={navClass('/screener')}><Globe className="h-4 w-4" /> <span>Screener</span></Link>
          <Link to="/analytics" className={navClass('/analytics')}><LineChart className="h-4 w-4" /> <span>Analytics</span></Link>
        </div>
      </div>
    </nav>
  );
};

const Terminal = () => {
  const [ticker, setTicker] = useState('^GSPC');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  
  // NEW: State flag to track if the engine has been started
  const [engineStarted, setEngineStarted] = useState(false);

  const fetchRegimeIntelligence = async () => {
    setLoading(true);
    setError(null);
    setEngineStarted(true); // Lock the engine state to 'ON'
    
    try {
      const response = await axios.get(`http://127.0.0.1:8000/predict/${ticker}`);
      setData(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || "Failed to connect to the Intelligence Engine API.");
    } finally {
      setLoading(false);
    }
  };

  // NEW: React Effect Hook. This watches the 'ticker' state.
  // If the ticker changes AND the engine is already started, it automatically fetches.
  React.useEffect(() => {
    if (engineStarted) {
      fetchRegimeIntelligence();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticker]);

  const formatChartData = (probs) => {
    if (!probs) return [];
    return Object.entries(probs).map(([regime, prob]) => {
      const idx = parseInt(regime.split(" ")[1]);
      return {
        name: getRegimeName(idx).split(" ")[0],
        probability: parseFloat((prob * 100).toFixed(2))
      };
    });
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* HEADER & SEARCH */}
      <div className="rounded-xl border border-slate-800 bg-slate-900 shadow-sm p-6 flex flex-col md:flex-row justify-between items-center">
        <div>
          <h2 className="text-xl font-semibold text-white">Deep-Dive Intelligence</h2>
          <p className="text-slate-400 text-sm">HMM Causal Inference & Policy Optimization</p>
        </div>
        <div className="flex space-x-3 mt-4 md:mt-0 w-full md:w-auto">
          <select 
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="flex h-10 w-full md:w-64 rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            {availableAssets.map(asset => (
              <option key={asset.value} value={asset.value}>{asset.label}</option>
            ))}
          </select>
          
          {/* Automatically hide the button once the engine is running */}
          {!engineStarted && (
            <button 
              onClick={fetchRegimeIntelligence}
              disabled={loading}
              className="inline-flex h-10 items-center justify-center rounded-md bg-blue-600 px-6 text-sm font-medium text-white transition-colors hover:bg-blue-700 disabled:opacity-50"
            >
              {loading ? "Computing..." : "Run Engine"}
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="rounded-lg border border-red-900 bg-red-950/50 p-4 text-red-200 flex items-start">
          <ShieldAlert className="h-5 w-5 mr-3 mt-0.5 flex-shrink-0" />
          <div><h5 className="mb-1 font-medium">Connection Error</h5><div className="text-sm opacity-90">{error}</div></div>
        </div>
      )}

      {data && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* LEFT: STATE & PROBABILITIES */}
          <div className="md:col-span-2 rounded-xl border border-slate-800 bg-slate-900 shadow-xl p-6">
            <div className="flex justify-between items-start mb-6">
              <div>
                <p className="text-sm uppercase tracking-wider font-bold text-slate-400">Target Asset</p>
                <h3 className="text-3xl font-semibold text-white mt-1">
                  {data.metadata.ticker} <span className="text-slate-500 text-xl font-normal ml-1">({data.metadata.asset_class})</span>
                </h3>
              </div>
              <div className="text-right">
                <div className="text-sm text-slate-400">T-0 Close ({data.metadata.latest_date})</div>
                <div className="text-2xl font-mono font-bold text-emerald-400 mt-1">${data.metadata.latest_close.toLocaleString()}</div>
              </div>
            </div>
            
            <div className="flex flex-wrap gap-3 mb-8">
              <div className="inline-flex items-center rounded-md border border-blue-900/50 bg-blue-950/30 px-3 py-1.5 text-sm font-medium text-blue-400">
                Latent State: {getRegimeName(data.regime_intelligence.current_dominant_state)}
              </div>
              {data.regime_intelligence.is_black_swan && (
                <div className="inline-flex items-center rounded-md border border-red-500 bg-red-950 px-3 py-1.5 text-sm font-bold text-red-400 animate-pulse">
                  <AlertTriangle className="w-4 h-4 mr-2" /> BLACK SWAN PARAMETERS ENGAGED
                </div>
              )}
            </div>

            <h3 className="text-sm font-semibold text-slate-400 uppercase mb-4">Current Latent State Probabilities (T-0)</h3>
            <div className="h-56 w-full border-t border-slate-800 pt-6">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={formatChartData(data.regime_intelligence.forecast_t1_probabilities)}>
                  <XAxis dataKey="name" stroke="#64748b" tick={{fill: '#94a3b8', fontSize: 12}} />
                  <YAxis stroke="#64748b" tick={{fill: '#94a3b8', fontSize: 12}} domain={[0, 100]} />
                  <Tooltip cursor={{fill: '#1e293b'}} contentStyle={{backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', borderRadius: '8px'}} formatter={(value) => [`${value}%`, 'Probability']} />
                  <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
                    {formatChartData(data.regime_intelligence.forecast_t1_probabilities).map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.probability > 50 ? '#3b82f6' : '#475569'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* RIGHT: POLICY ENGINE */}
          <div className="rounded-xl border border-slate-800 bg-slate-900 shadow-xl flex flex-col p-6">
            <p className="text-sm uppercase tracking-wider font-bold text-slate-400 flex items-center mb-6">
              <Activity className="w-4 h-4 mr-2" /> Output Policy
            </p>
            <div className="flex-grow flex flex-col justify-center items-center py-6 border-y border-slate-800 mb-6">
              <div className={`text-7xl font-black tracking-tighter ${data.policy_action.recommended_exposure >= 0.8 ? 'text-emerald-500' : data.policy_action.recommended_exposure <= 0.3 ? 'text-amber-500' : 'text-blue-500'}`}>
                {(data.policy_action.recommended_exposure * 100).toFixed(0)}%
              </div>
              <div className="text-slate-400 text-xs mt-3 font-semibold uppercase tracking-widest">Optimized Target Exposure</div>
            </div>
            
            <div className="rounded-lg border border-slate-800 bg-slate-950 p-4 text-slate-200 mt-auto">
              <div className="flex items-center text-blue-400 mb-2">
                <Info className="h-4 w-4 mr-2" />
                <h5 className="font-semibold text-sm">Execution Mandate</h5>
              </div>
              <div className="text-sm text-slate-400 leading-relaxed italic">
                "{getExecutionMandate(data.metadata.asset_class, data.policy_action.recommended_exposure, data.regime_intelligence.is_black_swan)}"
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

const Screener = () => {
  const [assets, setAssets] = useState([]);
  const [loading, setLoading] = useState(true);

  React.useEffect(() => {
    const fetchScreener = async () => {
      try {
        const response = await axios.get(`http://127.0.0.1:8000/screener`);
        setAssets(response.data.heatmap);
      } catch (error) {
        console.error("Screener fetch failed:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchScreener();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-slate-400">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
        <p>Scanning Global Macro Regimes...</p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="mb-6 flex justify-between items-end">
        <div>
          <h2 className="text-2xl font-bold text-white">Global Macro Heatmap</h2>
          <p className="text-slate-400 text-sm mt-1">Live cross-asset regime identification and risk mandates.</p>
        </div>
        <div className="text-xs text-slate-500 font-mono">
          SYSTEM LIVE • {assets.length} ASSETS MONITORED
        </div>
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900 shadow-2xl overflow-hidden">
        <table className="w-full text-left text-sm text-slate-300">
          <thead className="bg-slate-950/80 text-xs uppercase text-slate-500 border-b border-slate-800">
            <tr>
              <th className="px-6 py-5 font-semibold tracking-wider">Target Asset</th>
              <th className="px-6 py-5 font-semibold tracking-wider">Class</th>
              <th className="px-6 py-5 font-semibold text-right tracking-wider">T-0 Close</th>
              <th className="px-6 py-5 font-semibold text-center tracking-wider">Latent State</th>
              <th className="px-6 py-5 font-semibold text-center tracking-wider w-48">Exposure Heat</th>
              <th className="px-6 py-5 font-semibold text-right tracking-wider">Target</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/50">
            {assets.map((asset) => {
              // Extract the base regime name
              const rawRegime = asset.is_black_swan ? "Shock" : getRegimeName(asset.current_state).split(" ")[0];
              
              // Color mappings for the Regime Badge
              const regimeColors = {
                "Bull": "bg-emerald-950/50 text-emerald-400 border-emerald-800/50",
                "Bear": "bg-rose-950/50 text-rose-400 border-rose-800/50",
                "Chop": "bg-slate-800 text-slate-300 border-slate-600",
                "Shock": "bg-red-900 text-red-100 border-red-500 animate-pulse font-bold"
              };
              const badgeStyle = regimeColors[rawRegime] || regimeColors["Chop"];

              // Color gradients for the Exposure Bar
              let expColor = "bg-blue-500"; 
              if (asset.target_exposure >= 0.8) expColor = "bg-emerald-500";
              else if (asset.target_exposure <= 0.3) expColor = "bg-rose-500";
              else if (asset.target_exposure <= 0.5) expColor = "bg-amber-500";

              return (
                <tr key={asset.ticker} className="hover:bg-slate-800/30 transition-colors">
                  <td className="px-6 py-5 font-bold text-white text-base">{asset.ticker}</td>
                  <td className="px-6 py-5 text-xs font-mono text-slate-500">{asset.asset_class}</td>
                  <td className="px-6 py-5 font-mono text-right text-emerald-400/90 font-medium">
                    ${asset.latest_close.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </td>
                  <td className="px-6 py-5 text-center">
                    <span className={`inline-flex px-3 py-1 rounded border text-xs tracking-widest uppercase ${badgeStyle}`}>
                      {rawRegime}
                    </span>
                  </td>
                  <td className="px-6 py-5 align-middle">
                    <div className="w-full bg-slate-950 rounded-full h-2.5 border border-slate-800 overflow-hidden">
                      <div 
                        className={`h-full rounded-full ${expColor} transition-all duration-1000 ease-out`} 
                        style={{ width: `${Math.max(asset.target_exposure * 100, 5)}%` }}
                      ></div>
                    </div>
                  </td>
                  <td className="px-6 py-5 font-black text-right text-white text-xl">
                    {(asset.target_exposure * 100).toFixed(0)}%
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};

const Analytics = () => {
  const [targetAsset, setTargetAsset] = useState('^GSPC');
  const [analyticsData, setAnalyticsData] = useState(null);
  const [loading, setLoading] = useState(true);

  React.useEffect(() => {
    const fetchAnalytics = async () => {
      setLoading(true);
      try {
        const response = await axios.get(`http://127.0.0.1:8000/analytics/${targetAsset}`);
        setAnalyticsData(response.data);
      } catch (error) {
        console.error("Failed to fetch analytics:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchAnalytics();
  }, [targetAsset]); // This dependency array means it refetches whenever you change the dropdown

  const MetricCard = ({ title, value, bench, inverse = false }) => {
    const isBetter = inverse 
      ? parseFloat(value) > parseFloat(bench)
      : parseFloat(value) > parseFloat(bench);
      
    return (
      <div className="rounded-xl border border-slate-800 bg-slate-900 p-5 flex flex-col justify-between">
        <h4 className="text-slate-400 text-xs font-bold uppercase tracking-wider mb-2">{title}</h4>
        <div className="flex items-end justify-between">
          <span className={`text-3xl font-black ${isBetter ? 'text-blue-400' : 'text-slate-200'}`}>{value}</span>
          <div className="text-right flex flex-col">
            <span className="text-[10px] text-slate-500 uppercase tracking-widest">Benchmark</span>
            <span className="text-sm font-mono text-slate-400">{bench}</span>
          </div>
        </div>
      </div>
    );
  };

  if (loading || !analyticsData) {
    return (
      <div className="flex flex-col items-center justify-center h-96 text-slate-400">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mb-4"></div>
        <p>Loading Out-Of-Sample Backtest Data...</p>
      </div>
    );
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <div className="flex flex-col md:flex-row justify-between items-end mb-8 border-b border-slate-800 pb-6">
        <div>
          <h2 className="text-2xl font-bold text-white">Out-Of-Sample Validation</h2>
          <p className="text-slate-400 text-sm mt-1">Historical performance of the HMM Policy Engine vs. Naive Buy & Hold.</p>
        </div>
        <select 
          value={targetAsset}
          onChange={(e) => setTargetAsset(e.target.value)}
          className="mt-4 md:mt-0 flex h-10 w-full md:w-64 rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          <option value="^GSPC">S&P 500 Index (^GSPC)</option>
          <option value="^NSEI">Nifty 50 Index (^NSEI)</option>
          <option value="BTC-USD">Bitcoin (BTC-USD)</option>
          <option value="TLT">20+ Yr Treasury Bonds (TLT)</option>
          <option value="GC=F">Gold Futures (GC=F)</option>
          <option value="EURUSD=X">EUR/USD Forex</option>
        </select>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <MetricCard title="Sharpe Ratio" value={analyticsData.metrics.sharpe} bench={analyticsData.metrics.benchmark_sharpe} />
        <MetricCard title="Max Drawdown" value={analyticsData.metrics.max_drawdown} bench={analyticsData.metrics.benchmark_drawdown} inverse={true} />
        <MetricCard title="Ann. Return (CAGR)" value={analyticsData.metrics.cagr} bench={analyticsData.metrics.benchmark_cagr} />
        <MetricCard title="Calmar Ratio" value={analyticsData.metrics.calmar} bench={analyticsData.metrics.benchmark_calmar} />
      </div>

      <div className="rounded-xl border border-slate-800 bg-slate-900 shadow-xl p-6 h-[450px] flex flex-col">
        <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-6">Underwater Curve (Capital Bleed Analysis)</h3>
        <div className="flex-grow w-full">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={analyticsData.equity_curve} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
              
              <XAxis dataKey="date" stroke="#64748b" tick={{fill: '#94a3b8', fontSize: 12}} dy={10} />
              
              {/* Force the Y-Axis to start at 0% and go negative */}
              <YAxis stroke="#64748b" tick={{fill: '#94a3b8', fontSize: 12}} tickFormatter={(value) => `${value.toFixed(0)}%`} domain={['auto', 0]} />
              
              <Tooltip 
                contentStyle={{backgroundColor: '#0f172a', borderColor: '#334155', color: '#f8fafc', borderRadius: '8px'}}
                formatter={(value) => [`${value.toFixed(2)}%`, undefined]}
                labelStyle={{ color: '#94a3b8', marginBottom: '4px' }}
              />
              <Legend verticalAlign="top" height={36} iconType="circle" wrapperStyle={{ fontSize: '12px', color: '#cbd5e1' }}/>
              
              {/* Benchmark bleeds in Red */}
              <Area type="monotone" name="Benchmark Drawdown" dataKey="benchmark_dd" stroke="#ef4444" strokeWidth={2} fill="#ef4444" fillOpacity={0.2} strokeDasharray="5 5" />
              
              {/* Strategy holds strong in Blue */}
              <Area type="monotone" name="HMM Policy Drawdown" dataKey="strategy_dd" stroke="#3b82f6" strokeWidth={3} fill="#3b82f6" fillOpacity={0.4} activeDot={{ r: 6, fill: '#3b82f6', stroke: '#0f172a', strokeWidth: 2 }} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

// --- MAIN ROUTER ---
export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-slate-950 font-sans text-slate-50 selection:bg-blue-500/30">
        <Navigation />
        <main className="p-8">
          <Routes>
            <Route path="/" element={<Terminal />} />
            <Route path="/screener" element={<Screener />} />
            <Route path="/analytics" element={<Analytics />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}