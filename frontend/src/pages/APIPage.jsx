import { useState, useEffect } from 'react';
import  api  from '../utils/api';

export default function APIPage() {
  const [apiKey, setApiKey] = useState('');
  const [usage, setUsage] = useState({ total_calls: 0, recent_calls: [] });
  const [copied, setCopied] = useState(false);
  
  // 🔥 NEW: Test API state
  const [testFeatures, setTestFeatures] = useState({});
  const [testResult, setTestResult] = useState(null);
  const [testLoading, setTestLoading] = useState(false);
  const [featureSchema, setFeatureSchema] = useState(null);

  useEffect(() => {
    loadData();
    loadFeatureSchema();
  }, []);

  const loadData = async () => {
    const keysRes = await api.get('/api/keys');
    if (keysRes.data.api_keys.length > 0) {
      setApiKey(keysRes.data.api_keys[0].key);
    }
    
    const usageRes = await api.get('/api/usage');
    setUsage(usageRes.data);
  };

  const loadFeatureSchema = async () => {
    try {
      const res = await api.get('/models/production/features');
      setFeatureSchema(res.data);
      
      // Initialize test features with empty values
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
      // Call PUBLIC API endpoint using API key
      const res = await fetch('http://localhost:8000/api/predict', {
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
  -H "X-API-Key: ${apiKey}" \\
  -H "Content-Type: application/json" \\
  -d '${JSON.stringify({ features: testFeatures }, null, 2)}'`;

  const pythonExample = `import requests

response = requests.post(
    'http://localhost:8000/api/predict',
    headers={'X-API-Key': '${apiKey}'},
    json=${JSON.stringify({ features: testFeatures }, null, 4)}
)

print(response.json())`;

  return (
    <div style={{ padding: 40, maxWidth: 1200, margin: '0 auto' }}>
      <h1>🔑 API Access</h1>
      <p style={{ color: '#666' }}>Use your API to make predictions from any application</p>
      
      {/* API Key Section */}
      <div style={{ background: '#f5f5f5', padding: 20, borderRadius: 8, marginTop: 20 }}>
        <h3>Your API Key</h3>
        <div style={{ display: 'flex', gap: 10, alignItems: 'center' }}>
          <code style={{ flex: 1, background: 'white', padding: 10, borderRadius: 4, fontFamily: 'monospace' }}>
            {apiKey || 'Loading...'}
          </code>
          <button 
            onClick={() => copyToClipboard(apiKey)}
            style={{ padding: '10px 20px', cursor: 'pointer' }}
          >
            {copied ? '✅ Copied!' : '📋 Copy'}
          </button>
        </div>
      </div>

      {/* 🔥 NEW: Interactive API Tester */}
      <div style={{ background: '#e8f5e9', padding: 20, borderRadius: 8, marginTop: 30, border: '2px solid #4caf50' }}>
        <h3>🧪 Test Your API (No Code Required!)</h3>
        <p>Fill in the values below and click "Test API" to see it work:</p>
        
        {featureSchema && (
          <div style={{ marginTop: 20 }}>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: 15 }}>
              {featureSchema.features.map(feature => (
                <div key={feature}>
                  <label style={{ display: 'block', marginBottom: 5, fontWeight: 'bold' }}>
                    {feature}
                    <span style={{ fontWeight: 'normal', fontSize: '0.9em', color: '#666' }}>
                      {featureSchema.numeric_features.includes(feature) ? ' (number)' : ' (text)'}
                    </span>
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
                      padding: 8, 
                      borderRadius: 4, 
                      border: '1px solid #ccc',
                      fontSize: '14px'
                    }}
                  />
                </div>
              ))}
            </div>

            <button
              onClick={testAPI}
              disabled={testLoading}
              style={{
                marginTop: 20,
                padding: '12px 30px',
                background: '#4caf50',
                color: 'white',
                border: 'none',
                borderRadius: 6,
                fontSize: '16px',
                cursor: testLoading ? 'not-allowed' : 'pointer',
                fontWeight: 'bold'
              }}
            >
              {testLoading ? '⏳ Testing...' : '🚀 Test API'}
            </button>

            {testResult && (
              <div style={{
                marginTop: 20,
                padding: 15,
                borderRadius: 6,
                background: testResult.success ? '#d4edda' : '#f8d7da',
                border: `1px solid ${testResult.success ? '#c3e6cb' : '#f5c6cb'}`
              }}>
                {testResult.success ? (
                  <>
                    <h4 style={{ margin: '0 0 10px 0', color: '#155724' }}>✅ Success!</h4>
                    <p style={{ margin: 0 }}>
                      <strong>Prediction:</strong> {testResult.data.prediction.toFixed(2)}
                    </p>
                    <p style={{ margin: '5px 0 0 0', fontSize: '0.9em', color: '#155724' }}>
                      Model v{testResult.data.model_version} • Credits used: 1
                    </p>
                  </>
                ) : (
                  <>
                    <h4 style={{ margin: '0 0 10px 0', color: '#721c24' }}>❌ Error</h4>
                    <p style={{ margin: 0, color: '#721c24' }}>{JSON.stringify(testResult.error)}</p>
                  </>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Usage Stats */}
      <div style={{ marginTop: 30 }}>
        <h3>📊 Usage Stats</h3>
        <p>Total API calls: <strong>{usage.total_calls}</strong></p>
      </div>

      {/* Code Examples - Collapsible */}
      <details style={{ marginTop: 30 }}>
        <summary style={{ cursor: 'pointer', fontSize: '18px', fontWeight: 'bold' }}>
          💻 For Developers: Code Examples
        </summary>
        
        <div style={{ marginTop: 20 }}>
          <h4>cURL Example</h4>
          <pre style={{ 
            background: '#1e1e1e', 
            color: '#d4d4d4', 
            padding: 15, 
            borderRadius: 8, 
            overflow: 'auto',
            fontSize: '13px'
          }}>
            {curlExample}
          </pre>
          <button onClick={() => copyToClipboard(curlExample)}>📋 Copy cURL</button>
        </div>

        <div style={{ marginTop: 30 }}>
          <h4>🐍 Python Example</h4>
          <pre style={{ 
            background: '#1e1e1e', 
            color: '#d4d4d4', 
            padding: 15, 
            borderRadius: 8, 
            overflow: 'auto',
            fontSize: '13px'
          }}>
            {pythonExample}
          </pre>
          <button onClick={() => copyToClipboard(pythonExample)}>📋 Copy Python</button>
        </div>
      </details>

      {/* Recent Calls */}
      <div style={{ marginTop: 30 }}>
        <h3>🕒 Recent API Calls</h3>
        {usage.recent_calls.length === 0 ? (
          <p style={{ color: '#666' }}>No API calls yet. Test your API above!</p>
        ) : (
          <table style={{ width: '100%', borderCollapse: 'collapse', marginTop: 10 }}>
            <thead>
              <tr style={{ background: '#f5f5f5' }}>
                <th style={{ padding: 10, textAlign: 'left', borderBottom: '2px solid #ddd' }}>Endpoint</th>
                <th style={{ padding: 10, textAlign: 'left', borderBottom: '2px solid #ddd' }}>Model</th>
                <th style={{ padding: 10, textAlign: 'left', borderBottom: '2px solid #ddd' }}>Time</th>
              </tr>
            </thead>
            <tbody>
              {usage.recent_calls.map((call, i) => (
                <tr key={i} style={{ borderBottom: '1px solid #eee' }}>
                  <td style={{ padding: 10 }}>{call.endpoint}</td>
                  <td style={{ padding: 10 }}>v{call.model_version}</td>
                  <td style={{ padding: 10 }}>{new Date(call.timestamp).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}