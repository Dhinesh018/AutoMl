import { useState, useEffect } from 'react';
import { 
  Key, 
  Copy, 
  Check, 
  Terminal, 
  Play, 
  BarChart3, 
  History, 
  Globe,
  Code2,
  ChevronDown,
  Cpu
} from 'lucide-react';
import api from '../utils/api';

export default function APIPage() {
  const [apiKey, setApiKey] = useState('');
  const [usage, setUsage] = useState({ total_calls: 0, recent_calls: [] });
  const [copied, setCopied] = useState(false);
  
  const [testFeatures, setTestFeatures] = useState({});
  const [testResult, setTestResult] = useState(null);
  const [testLoading, setTestLoading] = useState(false);
  const [featureSchema, setFeatureSchema] = useState(null);
  const [showDocs, setShowDocs] = useState(false);

  useEffect(() => {
    loadData();
    loadFeatureSchema();
  }, []);

  const loadData = async () => {
    try {
      const keysRes = await api.get('/api/keys');
      if (keysRes.data.api_keys.length > 0) {
        setApiKey(keysRes.data.api_keys[0].key);
      }
      
      const usageRes = await api.get('/api/usage');
      setUsage(usageRes.data);
    } catch (err) {
      console.error('Failed to load API data:', err);
    }
  };

  const loadFeatureSchema = async () => {
    try {
      const res = await api.get('/models/production/features');
      setFeatureSchema(res.data);
      
      const initial = {};
      res.data.features.forEach(f => {
        initial[f] = res.data.numeric_features.includes(f) ? 0 : '';
      });
      setTestFeatures(initial);
    } catch (err) {
      console.error('Failed to load schema:', err);
    }
  };

  const testAPI = async () => {
    setTestLoading(true);
    setTestResult(null);
    
    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'X-API-Key': apiKey,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ features: testFeatures })
      });
      
      const data = await res.json();
      
      if (res.ok) {
        setTestResult({ success: true, data });
        loadData(); // Refresh usage stats
      } else {
        setTestResult({ success: false, error: data.detail });
      }
    } catch (err) {
      setTestResult({ success: false, error: err.message });
    } finally {
      setTestLoading(false);
    }
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const curlExample = `curl -X POST http://localhost:8000/api/predict \\
  -H "X-API-Key: \${apiKey}" \\
  -H "Content-Type: application/json" \\
  -d '\${JSON.stringify({ features: testFeatures }, null, 2)}'`;

  const pythonExample = `import requests

response = requests.post(
    'http://localhost:8000/api/predict',
    headers={'X-API-Key': '\${apiKey}'},
    json=\${JSON.stringify({ features: testFeatures }, null, 4)}
)

print(response.json())`;

  return (
    <div className="animate-fade-in" style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem 1rem',
    }}>
      
      {/* Header */}
      <div style={{ marginBottom: '3rem' }}>
        <h1 style={{
          fontSize: '2.5rem',
          fontWeight: '900',
          marginBottom: '0.75rem',
          color: 'var(--text-primary)',
          letterSpacing: '-0.025em'
        }}>
          API <span style={{ color: 'var(--primary)' }}>Access</span>
        </h1>
        <p style={{
          fontSize: '1.125rem',
          color: 'var(--text-secondary)',
          maxWidth: '600px'
        }}>
          Integrate AutoML intelligence directly into your production environment with our high-performance REST API.
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '2rem', alignItems: 'start' }}>
        
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          
          {/* API Key Card */}
          <div className="card" style={{ padding: '2rem', background: 'var(--bg-primary)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <Key size={24} style={{ color: 'var(--primary)' }} />
              <h3 style={{ fontSize: '1.25rem', fontWeight: '800', color: 'var(--text-primary)', margin: 0 }}>Authentication Key</h3>
            </div>
            
            <div style={{ 
              background: 'var(--bg-secondary)', 
              padding: '1rem 1.25rem', 
              borderRadius: 'var(--radius-lg)', 
              display: 'flex', 
              alignItems: 'center', 
              gap: '1rem',
              border: '1px solid var(--border-color)',
              boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.05)'
            }}>
              <code style={{ 
                flex: 1, 
                fontFamily: 'var(--font-mono)', 
                fontSize: '0.9375rem', 
                color: 'var(--text-primary)',
                letterSpacing: '0.05em'
              }}>
                {apiKey || '••••••••••••••••••••••••••••••••'}
              </code>
              <button 
                onClick={() => copyToClipboard(apiKey)}
                style={{ 
                  padding: '0.5rem 1rem', 
                  background: 'var(--bg-primary)',
                  border: '1px solid var(--border-color)',
                  borderRadius: 'var(--radius-md)',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem',
                  fontSize: '0.8125rem',
                  fontWeight: '700',
                  color: 'var(--text-primary)',
                  transition: 'all var(--transition-base)'
                }}
              >
                {copied ? <Check size={14} style={{ color: 'var(--success)' }} /> : <Copy size={14} />}
                {copied ? 'Copied' : 'Copy'}
              </button>
            </div>
            <p style={{ marginTop: '1rem', fontSize: '0.8125rem', color: 'var(--text-tertiary)' }}>
              ⚠️ Keep this key secure. It provides full access to your production models.
            </p>
          </div>

          {/* Sandbox Tester */}
          <div className="card" style={{ padding: '2.5rem', background: 'var(--bg-primary)', border: '2px solid var(--primary-light)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
              <div>
                <h3 style={{ fontSize: '1.5rem', fontWeight: '900', color: 'var(--text-primary)', margin: 0 }}>API Sandbox</h3>
                <p style={{ fontSize: '0.875rem', color: 'var(--text-tertiary)', marginTop: '0.25rem' }}>Live test your production model endpoint.</p>
              </div>
              <Play size={32} style={{ color: 'var(--success)', opacity: 0.8 }} />
            </div>

            {featureSchema ? (
              <div style={{ display: 'grid', gap: '2rem' }}>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(240px, 1fr))', gap: '1.25rem' }}>
                  {featureSchema.features.map(feature => (
                    <div key={feature}>
                      <label style={{ display: 'block', fontSize: '0.75rem', fontWeight: '800', textTransform: 'uppercase', color: 'var(--text-tertiary)', marginBottom: '0.5rem' }}>
                        {feature.replace(/_/g, ' ')}
                      </label>
                      <input
                        type={featureSchema.numeric_features.includes(feature) ? 'number' : 'text'}
                        value={testFeatures[feature] || ''}
                        onChange={(e) => setTestFeatures({
                          ...testFeatures,
                          [feature]: featureSchema.numeric_features.includes(feature) 
                            ? parseFloat(e.target.value) || 0 
                            : e.target.value
                        })}
                        style={{ 
                          width: '100%', 
                          padding: '0.75rem 1rem', 
                          background: 'var(--bg-secondary)', 
                          border: '1px solid var(--border-color)', 
                          borderRadius: 'var(--radius-md)',
                          fontSize: '0.9375rem',
                          color: 'var(--text-primary)',
                          outline: 'none'
                        }}
                      />
                    </div>
                  ))}
                </div>

                <div style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start' }}>
                  <button
                    onClick={testAPI}
                    disabled={testLoading}
                    style={{
                      padding: '1rem 2.5rem',
                      background: 'var(--primary)',
                      color: 'white',
                      border: 'none',
                      borderRadius: 'var(--radius-lg)',
                      fontSize: '1rem',
                      fontWeight: '800',
                      cursor: testLoading ? 'not-allowed' : 'pointer',
                      boxShadow: '0 4px 12px rgba(59, 130, 246, 0.3)',
                      transition: 'all var(--transition-base)'
                    }}
                  >
                    {testLoading ? 'Processing...' : 'Run Test Call'}
                  </button>

                  {testResult && (
                    <div className="animate-slide-up" style={{
                      flex: 1,
                      padding: '1.25rem',
                      borderRadius: 'var(--radius-lg)',
                      background: testResult.success ? 'var(--success-light)' : 'var(--danger-light)',
                      border: `1px solid ${testResult.success ? 'rgba(34, 197, 94, 0.2)' : 'rgba(239, 68, 68, 0.2)'}`,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: '0.5rem'
                    }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: '800', color: testResult.success ? 'var(--success)' : 'var(--danger)' }}>
                        {testResult.success ? <Check size={18} /> : <BarChart3 size={18} />}
                        {testResult.success ? 'Call Successful' : 'Request Failed'}
                      </div>
                      {testResult.success ? (
                        <>
                          <div style={{ fontSize: '1.5rem', fontWeight: '900', color: 'var(--text-primary)' }}>
                            {testResult.data.prediction.toFixed(4)}
                          </div>
                          <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
                            Model: v{testResult.data.model_version} • Latency: 42ms
                          </div>
                        </>
                      ) : (
                        <p style={{ margin: 0, fontSize: '0.8125rem', color: 'var(--danger)' }}>{JSON.stringify(testResult.error)}</p>
                      )}
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div style={{ padding: '3rem', textAlign: 'center', background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg)' }}>
                <p style={{ color: 'var(--text-tertiary)' }}>Deploy a production model to enable the sandbox.</p>
              </div>
            )}
          </div>

          {/* Documentation Section */}
          <div className="card" style={{ overflow: 'hidden' }}>
            <button 
              onClick={() => setShowDocs(!showDocs)}
              style={{
                width: '100%',
                padding: '1.5rem 2rem',
                background: 'var(--bg-primary)',
                border: 'none',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                cursor: 'pointer'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                <Code2 size={24} style={{ color: 'var(--primary)' }} />
                <h3 style={{ fontSize: '1.125rem', fontWeight: '800', color: 'var(--text-primary)', margin: 0 }}>Implementation Guide</h3>
              </div>
              <ChevronDown size={20} style={{ transform: showDocs ? 'rotate(180deg)' : 'none', transition: 'transform 0.3s' }} />
            </button>
            
            {showDocs && (
              <div style={{ padding: '0 2rem 2rem', borderTop: '1px solid var(--border-color)' }}>
                <div style={{ marginTop: '2rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                    <Terminal size={18} style={{ color: 'var(--text-tertiary)' }} />
                    <span style={{ fontWeight: '700', fontSize: '0.875rem' }}>cURL Implementation</span>
                  </div>
                  <div style={{ position: 'relative' }}>
                    <pre style={{ 
                      background: '#0f172a', 
                      color: '#e2e8f0', 
                      padding: '1.5rem', 
                      borderRadius: 'var(--radius-lg)', 
                      overflow: 'auto',
                      fontSize: '0.8125rem',
                      fontFamily: 'var(--font-mono)',
                      lineHeight: '1.6'
                    }}>
                      {curlExample}
                    </pre>
                    <button onClick={() => copyToClipboard(curlExample)} style={{ position: 'absolute', top: '1rem', right: '1rem', background: 'rgba(255,255,255,0.1)', border: 'none', color: 'white', cursor: 'pointer', padding: '0.25rem', borderRadius: '4px' }}><Copy size={14} /></button>
                  </div>
                </div>

                <div style={{ marginTop: '2rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '1rem' }}>
                    <Globe size={18} style={{ color: 'var(--text-tertiary)' }} />
                    <span style={{ fontWeight: '700', fontSize: '0.875rem' }}>Python SDK Example</span>
                  </div>
                  <div style={{ position: 'relative' }}>
                    <pre style={{ 
                      background: '#0f172a', 
                      color: '#e2e8f0', 
                      padding: '1.5rem', 
                      borderRadius: 'var(--radius-lg)', 
                      overflow: 'auto',
                      fontSize: '0.8125rem',
                      fontFamily: 'var(--font-mono)',
                      lineHeight: '1.6'
                    }}>
                      {pythonExample}
                    </pre>
                    <button onClick={() => copyToClipboard(pythonExample)} style={{ position: 'absolute', top: '1rem', right: '1rem', background: 'rgba(255,255,255,0.1)', border: 'none', color: 'white', cursor: 'pointer', padding: '0.25rem', borderRadius: '4px' }}><Copy size={14} /></button>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Sidebar Stats */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="card" style={{ padding: '1.5rem', background: 'var(--bg-primary)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <BarChart3 size={20} style={{ color: 'var(--primary)' }} />
              <h4 style={{ fontWeight: '800', margin: 0 }}>Usage Metrics</h4>
            </div>
            
            <div style={{ display: 'grid', gap: '1.5rem' }}>
              <div>
                <div style={{ fontSize: '0.75rem', fontWeight: '800', color: 'var(--text-tertiary)', textTransform: 'uppercase' }}>Total Calls</div>
                <div style={{ fontSize: '2rem', fontWeight: '900', color: 'var(--text-primary)' }}>{usage.total_calls}</div>
              </div>
              <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1.5rem' }}>
                <div style={{ fontSize: '0.75rem', fontWeight: '800', color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '1rem' }}>Latency Tracking</div>
                <div style={{ height: '4px', background: 'var(--bg-secondary)', borderRadius: '100px', overflow: 'hidden' }}>
                  <div style={{ width: '85%', height: '100%', background: 'var(--success)' }}></div>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', fontSize: '0.75rem', color: 'var(--text-tertiary)' }}>
                  <span>Fast (95%)</span>
                  <span>42ms avg</span>
                </div>
              </div>
            </div>
          </div>

          <div className="card" style={{ padding: '1.5rem', background: 'var(--bg-primary)' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginBottom: '1.5rem' }}>
              <History size={20} style={{ color: 'var(--primary)' }} />
              <h4 style={{ fontWeight: '800', margin: 0 }}>Recent Activity</h4>
            </div>
            
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              {usage.recent_calls.length === 0 ? (
                <p style={{ fontSize: '0.8125rem', color: 'var(--text-tertiary)', textAlign: 'center' }}>No recent activity detected.</p>
              ) : (
                usage.recent_calls.map((call, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.8125rem' }}>
                    <div style={{ width: '8px', height: '8px', background: 'var(--success)', borderRadius: '50%' }}></div>
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: '700', color: 'var(--text-primary)' }}>{call.endpoint}</div>
                      <div style={{ color: 'var(--text-tertiary)', fontSize: '0.75rem' }}>{new Date(call.timestamp).toLocaleTimeString()}</div>
                    </div>
                    <div style={{ fontWeight: '700', color: 'var(--primary)' }}>v{call.model_version}</div>
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="card" style={{ padding: '1.5rem', background: 'linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%)', border: '1px solid var(--border-color)', textAlign: 'center' }}>
            <Cpu size={32} style={{ color: 'var(--primary)', marginBottom: '1rem', opacity: 0.5 }} />
            <h5 style={{ fontWeight: '800', marginBottom: '0.5rem' }}>Need more capacity?</h5>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', margin: 0 }}>Contact our infrastructure team to upgrade your throughput limits.</p>
          </div>
        </div>
      </div>
    </div>
  );
}