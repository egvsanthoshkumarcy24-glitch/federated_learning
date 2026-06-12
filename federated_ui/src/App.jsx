import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, Area, AreaChart, ComposedChart, Scatter, ReferenceLine } from 'recharts';
import { Activity, Shield, Cpu, Database, Network, Server, Settings, Upload, Play, AlertTriangle, FileText, ChevronRight, CheckCircle, Info, Lock, Globe, FileUp, List, Save, X } from 'lucide-react';

const API_BASE = 'http://localhost:8000/api';

const GLOBAL_STYLES = `
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;700&display=swap');

@keyframes pulse-cyan { 0%,100%{opacity:1;} 50%{opacity:0.4;} }
@keyframes float { 0%,100%{transform:translateY(0px);} 50%{transform:translateY(-10px);} }
@keyframes glow-pulse { 0%,100%{box-shadow:0 0 20px rgba(0,212,255,0.3);} 50%{box-shadow:0 0 40px rgba(0,212,255,0.6), 0 0 80px rgba(0,212,255,0.2);} }
@keyframes scan-line { 0%{transform:translateY(-100%);} 100%{transform:translateY(100vh);} }
@keyframes data-flow { 0%{stroke-dashoffset:1000;} 100%{stroke-dashoffset:0;} }
@keyframes shimmer { 0%{background-position:-200% 0;} 100%{background-position:200% 0;} }
@keyframes node-ping { 0%{transform:scale(1);opacity:1;} 100%{transform:scale(2.5);opacity:0;} }
@keyframes slide-in-right { from{opacity:0;transform:translateX(30px);} to{opacity:1;transform:translateX(0);} }
@keyframes fade-up { from{opacity:0;transform:translateY(40px);} to{opacity:1;transform:translateY(0);} }
@keyframes particle-float { 0%{transform:translateY(0) translateX(0); opacity:0;} 10%{opacity:1;} 90%{opacity:1;} 100%{transform:translateY(-100vh) translateX(50px); opacity:0;} }
@keyframes spin { from{transform:rotate(0deg);} to{transform:rotate(360deg);} }

* { box-sizing: border-box; margin: 0; padding: 0; }
body { 
  background: #050810; 
  color: #e2e8f0; 
  font-family: 'Inter', system-ui, sans-serif; 
  overflow-x: hidden;
}
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0e1a; }
::-webkit-scrollbar-thumb { background: #00d4ff; border-radius: 2px; }

.glass-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(0,212,255,0.15);
  border-radius: 16px;
  backdrop-filter: blur(20px);
  -webkit-backdrop-filter: blur(20px);
  box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05);
  transition: all 0.3s ease;
}
.glass-card:hover {
  transform: translateY(-4px);
  border-color: rgba(0,212,255,0.3);
  box-shadow: 0 12px 40px rgba(0,0,0,0.5), 0 0 20px rgba(0,212,255,0.2), inset 0 1px 0 rgba(255,255,255,0.1);
}
.gradient-text {
  background: linear-gradient(135deg, #00d4ff, #7c3aed);
  -webkit-background-clip: text;
  color: transparent;
}
.page-enter { animation: fade-up 0.6s ease both; }
.custom-range::-webkit-slider-thumb {
  -webkit-appearance: none; height: 16px; width: 16px; border-radius: 50%;
  background: #00d4ff; cursor: pointer; box-shadow: 0 0 10px #00d4ff;
}
.custom-range::-webkit-slider-runnable-track {
  width: 100%; height: 4px; cursor: pointer; background: rgba(255,255,255,0.1); border-radius: 2px;
}
.custom-select {
  width: 100%; padding: 12px; border-radius: 8px; background: rgba(0,0,0,0.3); border: 1px solid rgba(255,255,255,0.1); color: #fff; outline: none; appearance: none;
}
.custom-select:focus { border-color: #00d4ff; }
`;

const AnimatedCounter = ({ end, duration = 2000, suffix = "", prefix = "" }) => {
  const [count, setCount] = useState(0);
  useEffect(() => {
    if (!end) return;
    let start = 0;
    const increment = Math.max(1, end / (duration / 16));
    const timer = setInterval(() => {
      start += increment;
      if (start >= end) {
        setCount(end);
        clearInterval(timer);
      } else {
        setCount(start);
      }
    }, 16);
    return () => clearInterval(timer);
  }, [end, duration]);
  return <span>{prefix}{Math.floor(count)}{suffix}</span>;
};

const GlowButton = ({ children, onClick, color = '#00d4ff', bgGradient = 'linear-gradient(135deg, #00d4ff, #7c3aed)', style = {}, disabled=false }) => {
  const [hover, setHover] = useState(false);
  return (
    <button
      onClick={onClick} disabled={disabled}
      onMouseEnter={() => setHover(true)} onMouseLeave={() => setHover(false)}
      style={{
        padding: '12px 24px', borderRadius: '8px', border: 'none',
        background: disabled ? 'rgba(255,255,255,0.05)' : bgGradient,
        color: disabled ? '#64748b' : '#fff', fontSize: '15px', fontWeight: 600,
        cursor: disabled ? 'not-allowed' : 'pointer',
        boxShadow: hover && !disabled ? `0 0 20px ${color}80` : 'none',
        transform: hover && !disabled ? 'translateY(-2px)' : 'none',
        transition: 'all 0.3s ease', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px',
        ...style
      }}
    >
      {children}
    </button>
  );
};

const LandingPage = ({ setCurrentPage }) => {
  const particles = Array.from({length: 40}).map((_, i) => ({
    left: `${Math.random() * 100}%`, top: `${Math.random() * 100}%`,
    delay: `${Math.random() * 5}s`, size: `${Math.random() * 2 + 2}px`,
    color: Math.random() > 0.5 ? '#00d4ff' : '#7c3aed'
  }));

  return (
    <div className="page-enter" style={{ minHeight: '100vh', paddingTop: '70px', position: 'relative', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center' }}>
      {particles.map((p, i) => (
        <div key={i} style={{ position: 'absolute', left: p.left, top: p.top, width: p.size, height: p.size, background: p.color, borderRadius: '50%', animation: `particle-float 10s linear infinite`, animationDelay: p.delay, opacity: 0 }} />
      ))}
      <div style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '2px', background: '#00d4ff', opacity: 0.15, animation: 'scan-line 4s linear infinite' }} />
      
      <div style={{ textAlign: 'center', zIndex: 10 }}>
         <h1 style={{ fontSize: 'clamp(48px, 8vw, 120px)', fontWeight: 900, lineHeight: 1, marginBottom: '10px' }}>FEDERATED</h1>
         <h1 className="gradient-text" style={{ fontSize: 'clamp(48px, 8vw, 120px)', fontWeight: 900, lineHeight: 1, marginBottom: '20px' }}>INTELLIGENCE</h1>
         <p style={{ color: '#64748b', fontSize: '18px', maxWidth: '600px', margin: '0 auto 40px' }}>Privacy-Preserving Anomaly Detection for Cold-Chain IoT Infrastructure</p>
         
         <div style={{ display: 'flex', gap: '20px', justifyContent: 'center', marginBottom: '80px' }}>
            <GlowButton onClick={() => setCurrentPage('login')}>Enter Platform →</GlowButton>
            <button onClick={() => setCurrentPage('about')} className="glass-card" style={{ padding: '12px 24px', borderRadius: '8px', color: '#e2e8f0', fontSize: '15px', fontWeight: 600, cursor: 'pointer', border: '1px solid rgba(255,255,255,0.1)' }}>Learn How It Works</button>
         </div>
      </div>
    </div>
  );
};

const AboutPage = () => {
  return (
    <div className="page-enter" style={{ minHeight: '100vh', paddingTop: '100px', paddingBottom: '50px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <h1 style={{ fontSize: '42px', fontWeight: 800, marginBottom: '10px' }}>How <span className="gradient-text">Federated Learning</span> Works</h1>
      <p style={{ color: '#64748b', fontSize: '16px', maxWidth: '700px', textAlign: 'center', marginBottom: '60px' }}>Train advanced AI models across distributed cold-chain nodes without ever moving or exposing raw sensor data. Total privacy, global intelligence.</p>
    </div>
  );
};

const LoginPage = ({ setCurrentPage, setSelectedCompany, companies }) => {
  return (
    <div className="page-enter" style={{ minHeight: '100vh', paddingTop: '100px', padding: '100px 40px 50px', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
      <h1 style={{ fontSize: '36px', fontWeight: 800, marginBottom: '10px' }}>Select Your Workspace</h1>
      <p style={{ color: '#64748b', marginBottom: '50px' }}>Federated Learning Client Portal</p>
      
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '30px', width: '100%', maxWidth: '1200px' }}>
        {companies.map((company, index) => (
          <div key={company.id} className="glass-card" style={{ padding: '30px', display: 'flex', flexDirection: 'column', alignItems: 'center', animation: `fade-up 0.5s ease both ${index*0.1}s` }}>
            <div style={{ width: '80px', height: '80px', borderRadius: '50%', background: `linear-gradient(135deg, ${company.color}40, transparent)`, border: `2px solid ${company.color}`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '32px', fontWeight: 'bold', color: company.color, marginBottom: '20px', boxShadow: `0 0 20px ${company.color}40` }}>
              {company.id}
            </div>
            <h2 style={{ fontSize: '20px', fontWeight: 700, marginBottom: '10px' }}>{company.name}</h2>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px', background: company.status === 'active' ? 'rgba(0,255,136,0.1)' : 'rgba(100,116,139,0.1)', color: company.status === 'active' ? '#00ff88' : '#64748b', padding: '4px 12px', borderRadius: '20px', fontSize: '12px', fontWeight: 'bold', marginBottom: '20px' }}>
              {company.status === 'active' && <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#00ff88', animation: 'pulse-cyan 2s infinite' }} />} 
              {company.status.toUpperCase()}
            </div>
            <div style={{ width: '100%', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '20px', marginBottom: '20px', fontSize: '13px', color: '#64748b', display: 'flex', flexDirection: 'column', gap: '10px' }}>
               <div style={{ display: 'flex', justifyContent: 'space-between' }}><span>Configured Features:</span> <span style={{color:'#e2e8f0'}}>{company.sensors?.length || 0} columns</span></div>
            </div>
            <GlowButton color={company.color} bgGradient={`linear-gradient(135deg, ${company.color}, #0a0e1a)`} style={{ width: '100%', marginTop: 'auto' }} onClick={() => { setSelectedCompany(company); setCurrentPage('dashboard'); }}>
              Enter Workspace <ChevronRight size={16} />
            </GlowButton>
          </div>
        ))}
        
        <div className="glass-card" style={{ padding: '30px', display: 'flex', flexDirection: 'column', alignItems: 'center', animation: `fade-up 0.5s ease both 0.5s`, gridColumn: '1 / -1', maxWidth: '600px', margin: '0 auto', background: 'rgba(245,158,11,0.05)', borderColor: 'rgba(245,158,11,0.2)' }}>
           <Shield size={48} color="#f59e0b" style={{ marginBottom: '20px', filter: 'drop-shadow(0 0 10px rgba(245,158,11,0.5))' }} />
           <h2 style={{ fontSize: '24px', fontWeight: 700, marginBottom: '5px', color: '#f59e0b' }}>Federation Admin</h2>
           <p style={{ color: '#64748b', marginBottom: '25px' }}>Global Orchestration Command Center</p>
           <GlowButton color="#f59e0b" bgGradient="linear-gradient(135deg, #f59e0b, #d97706)" style={{ width: '100%', maxWidth: '300px' }} onClick={() => setCurrentPage('admin')}>
              Access Command Center <ChevronRight size={16} />
           </GlowButton>
        </div>
      </div>
    </div>
  );
};

const DashboardPage = ({ company, setCurrentPage, refreshCompanies }) => {
  const [activeTab, setActiveTab] = useState('upload');
  
  const fileInputRef = useRef(null);
  const [datasetConfig, setDatasetConfig] = useState(company.config || {});
  const [isUploading, setIsUploading] = useState(false);
  
  // Column Configuration State
  const [pendingConfig, setPendingConfig] = useState(null);
  const [selectedFeatures, setSelectedFeatures] = useState([]);
  const [selectedTimestamp, setSelectedTimestamp] = useState('');
  const [selectedLabel, setSelectedLabel] = useState('');
  
  const [isTraining, setIsTraining] = useState(false);
  const [trainHistory, setTrainHistory] = useState(null);
  const [trainStatusMsg, setTrainStatusMsg] = useState('');
  
  const [isDetecting, setIsDetecting] = useState(false);
  const [anomalyData, setAnomalyData] = useState(null);
  const [threshold, setThreshold] = useState(0.05);
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  useEffect(() => {
    if (isTraining) {
      setTrainStatusMsg("Executing Local Training...");
    } else if (trainHistory) {
      setTrainStatusMsg("Local training completed.");
    }
  }, [isTraining, trainHistory]);

  const tabs = [
    { id: 'upload', label: 'Dataset Upload', icon: <FileUp size={16} /> },
    { id: 'train', label: 'Train Model', icon: <Cpu size={16} /> },
    { id: 'detect', label: 'Detect Anomalies', icon: <AlertTriangle size={16} /> },
  ];

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setIsUploading(true);
    setErrorMsg('');
    setSuccessMsg('');
    const formData = new FormData();
    formData.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/upload/${company.name}`, {
        method: 'POST',
        body: formData
      });
      const data = await res.json();
      if (data.status === 'success') {
          // Update base path config
          setDatasetConfig(prev => ({...prev, dataset_path: data.dataset_path || prev.dataset_path || 'uploaded.csv'}));
          
          // Set up pending schema state
          setPendingConfig({
              all_columns: data.all_columns || [],
              suggested_timestamp: data.suggested_timestamp || '<None (Use Row Order)>',
              suggested_features: data.suggested_features || [],
              suggested_label: data.suggested_label || '<None>'
          });
          
          setSelectedTimestamp(data.suggested_timestamp || '<None (Use Row Order)>');
          setSelectedFeatures(data.suggested_features || []);
          setSelectedLabel(data.suggested_label || '<None>');
      }
    } catch(err) {
      setErrorMsg("Upload failed: " + err.message);
    }
    setIsUploading(false);
  };

  const toggleFeature = (col) => {
      setSelectedFeatures(prev => prev.includes(col) ? prev.filter(c => c !== col) : [...prev, col]);
  };

  const handleSaveConfig = async () => {
      setErrorMsg('');
      setSuccessMsg('');
      try {
          const cfgUpdate = {
              timestamp_col: selectedTimestamp === '<None (Use Row Order)>' ? null : selectedTimestamp,
              feature_cols: selectedFeatures,
              label_col: selectedLabel === '<None>' ? null : selectedLabel
          };
          
          await fetch(`${API_BASE}/config/${company.name}`, {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(cfgUpdate)
          });
          
          setDatasetConfig(prev => ({...prev, ...cfgUpdate}));
          setPendingConfig(null);
          setSuccessMsg("Column Configuration Saved Successfully.");
          refreshCompanies();
      } catch (err) {
          setErrorMsg("Failed to save configuration: " + err.message);
      }
  };

  const handleTrain = async () => {
    if (!datasetConfig?.feature_cols || datasetConfig.feature_cols.length === 0) {
      setErrorMsg("Training aborted. Dataset must be uploaded and configuration must be saved.");
      return;
    }
    setIsTraining(true);
    setErrorMsg('');
    setSuccessMsg('');
    try {
      const res = await fetch(`${API_BASE}/train/${company.name}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ epochs: 10, window_size: 24, batch_size: 32 })
      });
      const data = await res.json();
      if (data.status === 'success') {
        setTrainHistory(data.history);
      } else {
        setErrorMsg(data.detail || "Training failed.");
      }
    } catch(err) {
      setErrorMsg("Training failed: " + err.message);
    }
    setIsTraining(false);
  };

  const handleDetect = async () => {
    if (!trainHistory && !datasetConfig?.window_size) {
      setErrorMsg("Detection aborted. Model must be trained first.");
      return;
    }
    setIsDetecting(true);
    setErrorMsg('');
    setSuccessMsg('');
    try {
      const res = await fetch(`${API_BASE}/detect/${company.name}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold })
      });
      const data = await res.json();
      if (data.status === 'success') {
        setAnomalyData(data);
      } else {
        setErrorMsg(data.detail || "Detection failed.");
      }
    } catch(err) {
      setErrorMsg("Detection failed: " + err.message);
    }
    setIsDetecting(false);
  };

  return (
    <div className="page-enter" style={{ minHeight: '100vh', paddingTop: '90px', padding: '90px 40px 50px', maxWidth: '1200px', margin: '0 auto' }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '15px', marginBottom: '30px' }}>
        <button onClick={() => setCurrentPage('login')} style={{ background: 'transparent', border: '1px solid rgba(255,255,255,0.1)', color: '#e2e8f0', padding: '8px 16px', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '5px' }}>← Back</button>
        <h1 style={{ fontSize: '28px', fontWeight: 800 }}>
          <span style={{ color: company.color }}>{company.name}</span> Workspace
        </h1>
      </div>

      <div style={{ display: 'flex', gap: '10px', marginBottom: '40px', background: 'rgba(255,255,255,0.03)', padding: '6px', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.05)' }}>
        {tabs.map(t => (
          <button 
            key={t.id} 
            onClick={() => { setActiveTab(t.id); setErrorMsg(''); setSuccessMsg(''); }}
            style={{ 
              flex: 1, padding: '12px', borderRadius: '8px', border: 'none', 
              background: activeTab === t.id ? `linear-gradient(135deg, ${company.color}40, transparent)` : 'transparent',
              color: activeTab === t.id ? '#fff' : '#64748b',
              fontWeight: 600, display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '8px', cursor: 'pointer',
              transition: 'all 0.2s', boxShadow: activeTab === t.id ? `inset 0 0 0 1px ${company.color}` : 'none'
            }}
          >
            {t.icon} {t.label}
          </button>
        ))}
      </div>

      {errorMsg && (
        <div className="page-enter" style={{ background: 'rgba(255,68,68,0.1)', border: '1px solid #ff4444', padding: '15px', borderRadius: '8px', marginBottom: '30px', color: '#ff4444', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <AlertTriangle size={20} /> {errorMsg}
        </div>
      )}
      
      {successMsg && (
        <div className="page-enter" style={{ background: 'rgba(0,255,136,0.1)', border: '1px solid #00ff88', padding: '15px', borderRadius: '8px', marginBottom: '30px', color: '#00ff88', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <CheckCircle size={20} /> {successMsg}
        </div>
      )}

      <div className="glass-card" style={{ padding: '40px', minHeight: '500px' }}>
        
        {activeTab === 'upload' && (
          <div className="page-enter">
            <h2 style={{ fontSize: '20px', marginBottom: '20px' }}>Dataset Configuration Workflow</h2>
            
            <input type="file" ref={fileInputRef} onChange={handleUpload} accept=".csv" style={{ display: 'none' }} />
            
            {(!datasetConfig?.dataset_path) && (
              <div 
                 onClick={() => fileInputRef.current.click()}
                 style={{ border: `2px dashed ${company.color}60`, borderRadius: '12px', padding: '40px', textAlign: 'center', background: `rgba(0,0,0,0.2)`, marginBottom: '30px', cursor: 'pointer' }}
              >
                 <Upload size={48} color={company.color} style={{ marginBottom: '15px', opacity: isUploading ? 0.4 : 0.8 }} />
                 <h3 style={{ fontSize: '18px', marginBottom: '5px' }}>{isUploading ? 'Parsing Dataset Schema...' : 'Click to Browse & Upload Dataset'}</h3>
                 <p style={{ color: '#64748b', fontSize: '14px' }}>Accepts .csv format (sensor telemetry)</p>
              </div>
            )}

            {datasetConfig?.dataset_path && (
                <div className="page-enter" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', background: 'rgba(255,255,255,0.05)', border: `1px solid rgba(255,255,255,0.1)`, borderRadius: '12px', padding: '20px', marginBottom: '30px' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
                        <div style={{ padding: '10px', background: `${company.color}20`, borderRadius: '8px' }}>
                            <FileText size={24} color={company.color} />
                        </div>
                        <div>
                            <div style={{ color: '#64748b', fontSize: '13px', marginBottom: '4px' }}>Active Dataset</div>
                            <div style={{ color: '#fff', fontSize: '16px', fontWeight: 600 }}>{datasetConfig.dataset_path.split(/[\\/]/).pop()}</div>
                        </div>
                    </div>
                    <button onClick={() => fileInputRef.current.click()} style={{ background: 'transparent', border: `1px dashed ${company.color}`, color: company.color, padding: '8px 16px', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600 }}>
                        <Upload size={14} /> Replace Dataset
                    </button>
                </div>
            )}

            {pendingConfig ? (
                <div className="page-enter" style={{ background: 'rgba(0,0,0,0.3)', padding: '30px', borderRadius: '12px', border: `1px solid ${company.color}40` }}>
                    <h3 style={{ marginBottom: '20px', display: 'flex', alignItems: 'center', gap: '10px' }}><Settings size={20} color={company.color} /> Column Configuration Panel</h3>
                    
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '30px' }}>
                        <div>
                            <label style={{ display: 'block', color: '#64748b', marginBottom: '10px', fontSize: '14px' }}>Timestamp Column</label>
                            <select className="custom-select" value={selectedTimestamp} onChange={(e) => setSelectedTimestamp(e.target.value)}>
                                <option value="<None (Use Row Order)>">&lt;None (Use Row Order)&gt;</option>
                                {pendingConfig.all_columns.map(c => <option key={c} value={c}>{c}</option>)}
                            </select>
                        </div>
                        <div>
                            <label style={{ display: 'block', color: '#64748b', marginBottom: '10px', fontSize: '14px' }}>Label / Target Column (Optional)</label>
                            <select className="custom-select" value={selectedLabel} onChange={(e) => setSelectedLabel(e.target.value)}>
                                <option value="<None>">&lt;None&gt;</option>
                                {pendingConfig.all_columns.map(c => <option key={c} value={c}>{c}</option>)}
                            </select>
                        </div>
                    </div>

                    <div style={{ marginBottom: '30px' }}>
                        <label style={{ display: 'block', color: '#64748b', marginBottom: '10px', fontSize: '14px' }}>Feature Columns (Model Inputs)</label>
                        <div style={{ padding: '15px', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px', background: 'rgba(0,0,0,0.2)' }}>
                            <p style={{ fontSize: '12px', color: '#64748b', marginBottom: '15px' }}>Click to select/deselect features to include in the LSTM Autoencoder training.</p>
                            <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap' }}>
                                {pendingConfig.all_columns.map(c => {
                                    if (c === selectedTimestamp || c === selectedLabel) return null; // hide timestamp and label from features
                                    const isSelected = selectedFeatures.includes(c);
                                    return (
                                        <div 
                                            key={c} 
                                            onClick={() => toggleFeature(c)}
                                            style={{ 
                                                background: isSelected ? `${company.color}30` : 'rgba(255,255,255,0.05)', 
                                                border: `1px solid ${isSelected ? company.color : 'rgba(255,255,255,0.1)'}`, 
                                                color: isSelected ? '#fff' : '#64748b', 
                                                padding: '6px 16px', borderRadius: '20px', fontSize: '14px',
                                                cursor: 'pointer', transition: 'all 0.2s',
                                                display: 'flex', alignItems: 'center', gap: '6px'
                                            }}
                                        >
                                            {c} {isSelected && <X size={14} />}
                                        </div>
                                    )
                                })}
                            </div>
                        </div>
                    </div>

                    <GlowButton color={company.color} onClick={handleSaveConfig} style={{ width: '100%' }}>
                        <Save size={18} /> Save Schema & Configuration
                    </GlowButton>
                </div>
            ) : (datasetConfig?.feature_cols && datasetConfig.feature_cols.length > 0) && (
                <div className="page-enter" style={{ background: 'rgba(0,255,136,0.05)', padding: '20px', borderRadius: '12px', border: '1px solid rgba(0,255,136,0.2)', marginBottom: '30px' }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '15px' }}>
                        <h3 style={{ color: '#00ff88', fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}><CheckCircle size={18} /> Active Configuration Summary</h3>
                        <button onClick={async () => {
                            try {
                                const res = await fetch(`${API_BASE}/dataset-info/${company.name}`);
                                const data = await res.json();
                                if (data.status === 'success') {
                                    setPendingConfig({
                                        all_columns: data.all_columns,
                                        suggested_timestamp: datasetConfig.timestamp_col || '<None (Use Row Order)>',
                                        suggested_features: datasetConfig.feature_cols || [],
                                        suggested_label: datasetConfig.label_col || '<None>'
                                    });
                                    setSelectedTimestamp(datasetConfig.timestamp_col || '<None (Use Row Order)>');
                                    setSelectedFeatures(datasetConfig.feature_cols || []);
                                    setSelectedLabel(datasetConfig.label_col || '<None>');
                                }
                            } catch (err) { setErrorMsg("Failed to fetch dataset info: " + err.message); }
                        }} style={{ background: 'transparent', border: `1px solid #00ff88`, color: '#00ff88', padding: '6px 12px', borderRadius: '6px', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '12px' }}>
                            <Settings size={14} /> Edit Schema
                        </button>
                    </div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px', fontSize: '14px' }}>
                        <div><span style={{ color: '#64748b' }}>Timestamp Column:</span> <span style={{ color: '#fff' }}>{datasetConfig.timestamp_col || 'None (Row Order)'}</span></div>
                        <div><span style={{ color: '#64748b' }}>Label Column:</span> <span style={{ color: '#fff' }}>{datasetConfig.label_col || 'None'}</span></div>
                        <div style={{ gridColumn: '1 / -1' }}>
                            <span style={{ color: '#64748b', display: 'block', marginBottom: '8px' }}>Selected Features ({datasetConfig.feature_cols.length}):</span>
                            <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
                                {datasetConfig.feature_cols.map(f => (
                                    <div key={f} style={{ background: 'rgba(255,255,255,0.1)', padding: '4px 10px', borderRadius: '4px', fontSize: '12px' }}>{f}</div>
                                ))}
                            </div>
                        </div>
                    </div>
                </div>
            )}
          </div>
        )}

        {activeTab === 'train' && (
          <div className="page-enter">
             {!isTraining && !trainHistory && (
               <GlowButton 
                 color={company.color} 
                 bgGradient={`linear-gradient(135deg, ${company.color}, #0a0e1a)`} 
                 style={{ width: '100%', padding: '16px' }} 
                 onClick={handleTrain}
                 disabled={!datasetConfig?.feature_cols || datasetConfig.feature_cols.length === 0}
               >
                 <Play size={20} /> Start Local Training (LSTM Autoencoder)
               </GlowButton>
             )}

             {(!datasetConfig?.feature_cols || datasetConfig.feature_cols.length === 0) && (
               <p style={{ textAlign: 'center', color: '#ff4444', marginTop: '15px', fontSize: '14px' }}>
                 <AlertTriangle size={14} style={{ verticalAlign: 'middle', marginRight: '5px' }} />
                 Dataset configuration is missing. Please upload and save the column schema before training.
               </p>
             )}

             {isTraining && (
               <div style={{ textAlign: 'center', padding: '40px' }}>
                 <div style={{ width: '40px', height: '40px', border: `3px solid rgba(255,255,255,0.1)`, borderTopColor: company.color, borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 20px' }} />
                 <h3 style={{ color: company.color }}>{trainStatusMsg || 'Training Local Model...'}</h3>
                 <p style={{ color: '#64748b', marginTop: '10px' }}>This may take a moment depending on dataset size.</p>
               </div>
             )}

             {trainHistory && (
               <div className="page-enter">
                 <h3 style={{ marginBottom: '20px', fontSize: '16px' }}>Local Training Loss Curve</h3>
                 <div style={{ height: '300px', width: '100%' }}>
                   <ResponsiveContainer>
                     <LineChart data={trainHistory}>
                       <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                       <XAxis dataKey="epoch" stroke="#64748b" />
                       <YAxis stroke="#64748b" />
                       <RechartsTooltip contentStyle={{ background: 'rgba(10,14,26,0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                       <Line type="monotone" dataKey="trainLoss" stroke={company.color} strokeWidth={3} dot={false} />
                       <Line type="monotone" dataKey="valLoss" stroke="#00ff88" strokeWidth={3} dot={false} />
                     </LineChart>
                   </ResponsiveContainer>
                 </div>
               </div>
             )}
          </div>
        )}

        {activeTab === 'detect' && (
          <div className="page-enter">
             <div style={{ display: 'flex', gap: '20px', alignItems: 'center', marginBottom: '30px', background: 'rgba(0,0,0,0.2)', padding: '20px', borderRadius: '12px' }}>
                <div style={{ flex: 1 }}>
                   <label style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '15px', color: '#64748b' }}>
                      <span>Anomaly Threshold (MSE)</span>
                      <span style={{ color: '#fff', fontWeight: 'bold' }}>{threshold.toFixed(3)}</span>
                   </label>
                   <input type="range" min="0.01" max="0.2" step="0.005" value={threshold} onChange={(e)=>setThreshold(parseFloat(e.target.value))} className="custom-range" />
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
                   <GlowButton color="#ff4444" bgGradient="linear-gradient(135deg, #ff4444, #991b1b)" onClick={handleDetect} disabled={isDetecting || (!trainHistory && !datasetConfig?.window_size)}>
                     {isDetecting ? 'Analyzing...' : 'Run Local Detection'}
                   </GlowButton>
                   {(!trainHistory && !datasetConfig?.window_size) && (
                      <span style={{ color: '#ff4444', fontSize: '12px', textAlign: 'center' }}>Model not trained</span>
                   )}
                </div>
             </div>

             {isDetecting && (
               <div style={{ textAlign: 'center', padding: '40px' }}>
                 <div style={{ width: '40px', height: '40px', border: `3px solid rgba(255,255,255,0.1)`, borderTopColor: '#ff4444', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 20px' }} />
                 <h3 style={{ color: '#ff4444' }}>Inferencing Time-Series...</h3>
               </div>
             )}

             {anomalyData && !isDetecting && (
               <div className="page-enter">
                 <div style={{ display: 'flex', gap: '20px', marginBottom: '30px' }}>
                    <div className="glass-card" style={{ flex: 1, padding: '20px', borderLeft: '4px solid #ff4444' }}>
                       <div style={{ color: '#64748b', fontSize: '14px', marginBottom: '5px' }}>Total Anomalies Detected</div>
                       <div style={{ fontSize: '32px', fontWeight: 800, color: '#ff4444' }}>{anomalyData.total_anomalies}</div>
                    </div>
                    <div className="glass-card" style={{ flex: 1, padding: '20px', borderLeft: '4px solid #00d4ff' }}>
                       <div style={{ color: '#64748b', fontSize: '14px', marginBottom: '5px' }}>Windows Scanned</div>
                       <div style={{ fontSize: '32px', fontWeight: 800, color: '#fff' }}>{anomalyData.total_scanned}</div>
                    </div>
                 </div>

                 <div style={{ height: '350px', width: '100%' }}>
                   <ResponsiveContainer>
                     <ComposedChart data={anomalyData.results.map(d => ({...d, dynamicAnomaly: d.isAnomaly ? d.mse : null}))}>
                       <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                       <XAxis dataKey="time" stroke="#64748b" tick={{fontSize: 10}} />
                       <YAxis stroke="#64748b" />
                       <RechartsTooltip contentStyle={{ background: 'rgba(10,14,26,0.9)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '8px' }} />
                       <Line type="monotone" dataKey="mse" stroke="#64748b" strokeWidth={2} dot={false} />
                       <ReferenceLine y={threshold} stroke="#ff4444" strokeDasharray="5 5" />
                       <Scatter dataKey="dynamicAnomaly" fill="#ff4444" />
                     </ComposedChart>
                   </ResponsiveContainer>
                 </div>
               </div>
             )}
          </div>
        )}
      </div>
    </div>
  );
};

const AdminPage = ({ setCurrentPage, setSelectedCompany, companies }) => {
  const [fedStatus, setFedStatus] = useState('idle');
  const [rounds, setRounds] = useState(3);
  const [statusData, setStatusData] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');
  const [fedStatusMsg, setFedStatusMsg] = useState('');

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/federate/status`);
      const data = await res.json();
      setStatusData(data);
      if (data.rounds_completed >= rounds) {
         // Optionally update status if it was running, but since we can't easily reference fedStatus directly here without circular deps, we can do it in the useEffect below.
      }
    } catch(err) { console.error(err); }
  }, [rounds]);

  useEffect(() => {
    if (fedStatus === 'running' && statusData?.rounds_completed >= rounds) {
       setFedStatus('complete');
    }
  }, [statusData, fedStatus, rounds]);

  useEffect(() => {
    if (fedStatus === 'running') {
      setFedStatusMsg("Federated Orchestration in Progress...");
    } else if (fedStatus === 'complete') {
      setFedStatusMsg("Federation Cycle Completed.");
    }
  }, [fedStatus]);

  useEffect(() => {
    fetchStatus();
    let interval = null;
    if (fedStatus === 'running') {
      interval = setInterval(fetchStatus, 3000);
    }
    return () => clearInterval(interval);
  }, [fedStatus, fetchStatus]);

  const handleTrigger = async () => {
    setFedStatus('running');
    setErrorMsg('');
    try {
      const res = await fetch(`${API_BASE}/federate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ num_rounds: rounds, window_size: 24, n_features: 2 })
      });
      const data = await res.json();
      if (data.status !== 'started') {
        setErrorMsg(data.detail || "Failed to start federation");
        setFedStatus('idle');
      }
      // Removed setTimeout. The useEffect tracking statusData will set it to complete automatically.
    } catch(err) {
      setErrorMsg("Error: " + err.message);
      setFedStatus('idle');
    }
  };

  return (
    <div className="page-enter" style={{ minHeight: '100vh', paddingTop: '70px', display: 'flex' }}>
      <div style={{ width: '250px', background: 'rgba(255,255,255,0.02)', borderRight: '1px solid rgba(0,212,255,0.1)', padding: '30px 20px', display: 'flex', flexDirection: 'column' }}>
        <h2 style={{ fontSize: '18px', fontWeight: 800, color: '#f59e0b', marginBottom: '40px', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <Shield size={20} /> FEDAI Admin
        </h2>
        
        <div style={{ color: '#fff', padding: '12px 15px', background: 'rgba(245,158,11,0.1)', borderRadius: '8px', marginBottom: '10px', fontWeight: 600, borderLeft: '3px solid #f59e0b' }}>Dashboard</div>
        
        <div style={{ marginTop: '20px', marginBottom: '10px', fontSize: '12px', color: '#64748b', textTransform: 'uppercase', letterSpacing: '1px' }}>Network Nodes</div>
        {companies.map(c => (
          <div key={c.id} onClick={() => { setSelectedCompany(c); setCurrentPage('dashboard'); }} style={{ color: '#e2e8f0', padding: '10px 15px', display: 'flex', alignItems: 'center', gap: '10px', fontSize: '14px', cursor: 'pointer', borderRadius: '8px', transition: 'background 0.2s' }} onMouseEnter={e=>e.currentTarget.style.background='rgba(255,255,255,0.05)'} onMouseLeave={e=>e.currentTarget.style.background='transparent'}>
            <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: c.status === 'active' ? '#00ff88' : '#64748b' }} /> {c.name}
          </div>
        ))}
        
        <div style={{ marginTop: 'auto' }}>
          <GlowButton color="#64748b" bgGradient="transparent" style={{ border: '1px solid rgba(255,255,255,0.1)', width: '100%' }} onClick={() => setCurrentPage('landing')}>Exit Admin</GlowButton>
        </div>
      </div>

      <div style={{ flex: 1, padding: '40px', overflowY: 'auto' }}>
        <h1 style={{ fontSize: '28px', fontWeight: 800, marginBottom: '30px' }}>Global Orchestration</h1>
        
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '20px', marginBottom: '40px' }}>
           <div className="glass-card" style={{ padding: '20px', borderTop: '2px solid #00ff88' }}>
              <div style={{ color: '#64748b', fontSize: '14px', marginBottom: '10px' }}>Active Companies</div>
              <div style={{ fontSize: '32px', fontWeight: 800, color: '#00ff88' }}>{companies.filter(c=>c.status==='active').length} / {companies.length}</div>
           </div>
           <div className="glass-card" style={{ padding: '20px', borderTop: '2px solid #00d4ff' }}>
              <div style={{ color: '#64748b', fontSize: '14px', marginBottom: '10px' }}>Latest Global Model</div>
              <div style={{ fontSize: '32px', fontWeight: 800, color: '#00d4ff' }}>v{statusData?.latest_model_version || 0}</div>
           </div>
           <div className="glass-card" style={{ padding: '20px', borderTop: '2px solid #7c3aed' }}>
              <div style={{ color: '#64748b', fontSize: '14px', marginBottom: '10px' }}>FL Rounds Completed</div>
              <div style={{ fontSize: '32px', fontWeight: 800, color: '#7c3aed' }}>{statusData?.rounds_completed || 0}</div>
           </div>
        </div>

        {errorMsg && <div style={{ color: '#ff4444', marginBottom: '20px' }}>{errorMsg}</div>}

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '30px', marginBottom: '40px' }}>
           <div className="glass-card" style={{ padding: '30px' }}>
              <h3 style={{ fontSize: '18px', marginBottom: '25px', display: 'flex', alignItems: 'center', gap: '10px' }}><Settings size={18} color="#f59e0b" /> Federation Controls</h3>
              
              <div style={{ marginBottom: '20px' }}>
                <label style={{ display: 'block', color: '#64748b', marginBottom: '8px' }}>Target FL Rounds</label>
                <div style={{ display: 'flex', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', overflow: 'hidden', border: '1px solid rgba(255,255,255,0.1)' }}>
                  <button onClick={()=>setRounds(r=>Math.max(1, r-1))} style={{ padding: '12px 20px', background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer', borderRight: '1px solid rgba(255,255,255,0.1)' }}>-</button>
                  <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 'bold' }}>{rounds}</div>
                  <button onClick={()=>setRounds(r=>r+1)} style={{ padding: '12px 20px', background: 'transparent', border: 'none', color: '#fff', cursor: 'pointer', borderLeft: '1px solid rgba(255,255,255,0.1)' }}>+</button>
                </div>
              </div>

              <GlowButton color="#ff4444" bgGradient="linear-gradient(135deg, #ff4444, #991b1b)" style={{ width: '100%', animation: fedStatus === 'running' ? 'glow-pulse 1.5s infinite' : 'none' }} onClick={handleTrigger} disabled={fedStatus === 'running'}>
                 {fedStatus === 'running' ? (fedStatusMsg || 'Federating...') : 'Trigger Federated Round'}
              </GlowButton>
           </div>

           <div className="glass-card" style={{ padding: '30px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'radial-gradient(circle at center, rgba(0,212,255,0.05) 0%, transparent 70%)' }}>
              <svg viewBox="0 0 500 400" style={{ width: '100%', height: '100%' }}>
                {companies.map((c, i) => {
                  const angle = (i * (360/companies.length) - 90) * (Math.PI / 180);
                  const x = 250 + Math.cos(angle) * 140;
                  const y = 200 + Math.sin(angle) * 120;
                  return (
                    <g key={i}>
                      <line x1="250" y1="200" x2={x} y2={y} stroke="rgba(255,255,255,0.1)" strokeWidth="2" />
                      {fedStatus === 'running' && c.status === 'active' && (
                        <circle cx="250" cy="200" r="4" fill="#00ff88">
                           <animate attributeName="cx" values={`${x};250;${x}`} dur="2s" repeatCount="indefinite" />
                           <animate attributeName="cy" values={`${y};200;${y}`} dur="2s" repeatCount="indefinite" />
                        </circle>
                      )}
                    </g>
                  )
                })}
                <circle cx="250" cy="200" r="50" fill="#0a0e1a" stroke="#00d4ff" strokeWidth="4" style={{ animation: fedStatus === 'running' ? 'glow-pulse 1s infinite' : 'none' }} />
                <text x="250" y="195" fill="#fff" fontSize="12" fontWeight="bold" textAnchor="middle">SERVER</text>
                
                {companies.map((c, i) => {
                  const angle = (i * (360/companies.length) - 90) * (Math.PI / 180);
                  const x = 250 + Math.cos(angle) * 140;
                  const y = 200 + Math.sin(angle) * 120;
                  return (
                    <g key={c.id}>
                      {fedStatus === 'running' && c.status === 'active' && <circle cx={x} cy={y} r="30" fill="none" stroke={c.color} strokeWidth="2" style={{ animation: `node-ping 1.5s infinite ${i*0.2}s`, transformOrigin: `${x}px ${y}px` }} />}
                      <circle cx={x} cy={y} r="25" fill="#0a0e1a" stroke={c.status === 'active' ? c.color : '#64748b'} strokeWidth="3" />
                      <text x={x} y={y+5} fill="#fff" fontSize="12" fontWeight="bold" textAnchor="middle">{c.id}</text>
                    </g>
                  )
                })}
              </svg>
           </div>
        </div>

        {statusData?.convergence_data && statusData.convergence_data.length > 0 && (
          <div className="glass-card" style={{ padding: '30px', marginBottom: '40px' }}>
             <h3 style={{ fontSize: '18px', marginBottom: '25px' }}>Global Model Convergence (MSE Loss)</h3>
             <div style={{ height: '300px', width: '100%' }}>
               <ResponsiveContainer>
                 <AreaChart data={statusData.convergence_data}>
                   <defs>
                     <linearGradient id="colorLoss" x1="0" y1="0" x2="0" y2="1">
                       <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3}/>
                       <stop offset="95%" stopColor="#00d4ff" stopOpacity={0}/>
                     </linearGradient>
                   </defs>
                   <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                   <XAxis dataKey="round" stroke="#64748b" />
                   <YAxis stroke="#64748b" domain={['auto', 'auto']} />
                   <RechartsTooltip contentStyle={{ background: 'rgba(10,14,26,0.9)', border: '1px solid rgba(0,212,255,0.2)', borderRadius: '8px' }} />
                   <Area type="monotone" dataKey="loss" stroke="#00d4ff" strokeWidth={3} fillOpacity={1} fill="url(#colorLoss)" />
                 </AreaChart>
               </ResponsiveContainer>
             </div>
          </div>
        )}

      </div>
    </div>
  );
};

export default function App() {
  const [currentPage, setCurrentPage] = useState('landing');
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [companies, setCompanies] = useState([]);

  const fetchCompanies = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/companies`);
      const data = await res.json();
      setCompanies(data);
    } catch(err) { console.error("API Connection Error:", err); }
  }, []);

  useEffect(() => {
    fetchCompanies();
  }, [fetchCompanies]);

  useEffect(() => {
    const styleSheet = document.createElement("style");
    styleSheet.innerText = GLOBAL_STYLES;
    document.head.appendChild(styleSheet);
    return () => styleSheet.remove();
  }, []);

  if (currentPage === 'landing') return <LandingPage setCurrentPage={setCurrentPage} />;
  if (currentPage === 'about') return <AboutPage setCurrentPage={setCurrentPage} />;
  if (currentPage === 'login') return <LoginPage setCurrentPage={setCurrentPage} setSelectedCompany={setSelectedCompany} companies={companies} />;
  if (currentPage === 'dashboard') return <DashboardPage company={selectedCompany} setCurrentPage={setCurrentPage} refreshCompanies={fetchCompanies} />;
  if (currentPage === 'admin') return <AdminPage setCurrentPage={setCurrentPage} setSelectedCompany={setSelectedCompany} companies={companies} />;
  
  return null;
}
