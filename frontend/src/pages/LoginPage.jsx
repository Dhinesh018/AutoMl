import React, { useState } from 'react'
import api from '../utils/api'

const LoginPage = ({ onLogin }) => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isSignup, setIsSignup] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    try {
      const endpoint = isSignup ? '/auth/signup' : '/auth/login'; 
      const response = await api.post(endpoint, { email, password });
      const token = response.data.access_token;
      
      if (token) {
        localStorage.setItem('token', token);
        api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
        onLogin();
      } else {
        setError('No token received');
      }
    } catch (err) {
      console.error('Auth error:', err);
      setError(err.response?.data?.detail || 'Authentication failed');
    }
  };

  return (
    <div style={styles.container}>
      <div style={styles.card}>
        {/* Technical Header Section */}
        <div style={styles.header}>
          <div style={styles.badgeRow}>
            <span style={styles.versionBadge}>v1.0.0</span>
            <span style={styles.oasBadge}>OAS 3.1</span>
          </div>
          <h1 style={styles.mainTitle}>🤖 LLM-Augmented AutoML</h1>
          <p style={styles.subTitle}>Intelligent AutoML with LLM-Powered Model Selection</p>
          <code style={styles.endpointPath}>GET /openapi.json</code>
        </div>

        <hr style={styles.divider} />

        <h2 style={styles.formTitle}>{isSignup ? 'Create Account' : 'Secure Login'}</h2>
        
        <form onSubmit={handleSubmit} style={styles.form}>
          <div style={styles.inputGroup}>
            <label style={styles.label}>Email Address</label>
            <input
              type="email"
              placeholder="engineer@system.ai"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              style={styles.input}
              required
            />
          </div>
          
          <div style={styles.inputGroup}>
            <label style={styles.label}>Password</label>
            <input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              style={styles.input}
              required
            />
          </div>
          
          {error && <div style={styles.errorBox}>{error}</div>}
          
          <button type="submit" style={styles.button}>
            {isSignup ? 'Initialize System Access' : 'Authenticate'}
          </button>
        </form>
        
        <p style={styles.footerText}>
          {isSignup ? 'Already registered?' : "New operator?"}
          <button 
            onClick={() => setIsSignup(!isSignup)} 
            style={styles.switchButton}
          >
            {isSignup ? 'Login' : 'Sign Up'}
          </button>
        </p>
      </div>
    </div>
  )
}

const styles = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: '#0f172a', // Deep slate background
    fontFamily: "'Inter', system-ui, sans-serif",
    padding: '20px',
  },
  card: {
    width: '100%',
    maxWidth: '440px',
    background: '#1e293b',
    padding: '40px',
    borderRadius: '16px',
    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
    border: '1px solid #334155',
  },
  header: {
    textAlign: 'center',
    marginBottom: '30px',
  },
  badgeRow: {
    display: 'flex',
    justifyContent: 'center',
    gap: '10px',
    marginBottom: '15px',
  },
  versionBadge: {
    background: '#3b82f6',
    color: '#fff',
    padding: '4px 10px',
    borderRadius: '20px',
    fontSize: '0.75rem',
    fontWeight: 'bold',
  },
  oasBadge: {
    background: '#10b981',
    color: '#fff',
    padding: '4px 10px',
    borderRadius: '20px',
    fontSize: '0.75rem',
    fontWeight: 'bold',
  },
  mainTitle: {
    color: '#f8fafc',
    fontSize: '1.5rem',
    margin: '0 0 8px 0',
    letterSpacing: '-0.025em',
  },
  subTitle: {
    color: '#94a3b8',
    fontSize: '0.9rem',
    margin: 0,
    lineHeight: '1.5',
  },
  endpointPath: {
    display: 'inline-block',
    marginTop: '12px',
    background: '#0f172a',
    color: '#60a5fa',
    padding: '4px 12px',
    borderRadius: '6px',
    fontSize: '0.8rem',
  },
  divider: {
    border: '0',
    borderTop: '1px solid #334155',
    margin: '25px 0',
  },
  formTitle: {
    color: '#f8fafc',
    fontSize: '1.1rem',
    marginBottom: '20px',
    textAlign: 'center',
  },
  inputGroup: {
    marginBottom: '20px',
  },
  label: {
    display: 'block',
    color: '#94a3b8',
    fontSize: '0.85rem',
    marginBottom: '8px',
    fontWeight: '500',
  },
  input: {
    width: '100%',
    padding: '12px 16px',
    background: '#0f172a',
    border: '1px solid #334155',
    borderRadius: '8px',
    color: '#f8fafc',
    fontSize: '1rem',
    boxSizing: 'border-box',
    outline: 'none',
    transition: 'border-color 0.2s',
  },
  button: {
    width: '100%',
    padding: '14px',
    background: '#3b82f6',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    fontSize: '1rem',
    fontWeight: '600',
    cursor: 'pointer',
    marginTop: '10px',
    transition: 'background 0.2s',
  },
  errorBox: {
    background: 'rgba(239, 68, 68, 0.1)',
    color: '#f87171',
    padding: '12px',
    borderRadius: '8px',
    fontSize: '0.85rem',
    marginBottom: '20px',
    border: '1px solid rgba(239, 68, 68, 0.2)',
  },
  footerText: {
    marginTop: '25px',
    textAlign: 'center',
    color: '#94a3b8',
    fontSize: '0.9rem',
  },
  switchButton: {
    marginLeft: '8px',
    color: '#3b82f6',
    background: 'none',
    border: 'none',
    cursor: 'pointer',
    fontWeight: '600',
    textDecoration: 'underline',
  }
}

export default LoginPage