import React, { useState } from 'react'
import api from '../utils/api'

const LoginPage = ({ onLogin }) => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isSignup, setIsSignup] = useState(false)
  const [error, setError] = useState(null)
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);
    
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
      setError(err.response?.data?.detail || 'Authentication failed. Please check your credentials.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div style={{ 
      minHeight: '100vh', 
      display: 'flex', 
      alignItems: 'center', 
      justifyContent: 'center', 
      background: 'linear-gradient(135deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%)',
      padding: '1.5rem'
    }}>
      <div className="card animate-slide-up" style={{ 
        maxWidth: '440px', 
        width: '100%', 
        padding: '2.5rem',
        boxShadow: 'var(--shadow-xl)',
        background: 'var(--bg-primary)'
      }}>
        <div style={{ textAlign: 'center', marginBottom: '2rem' }}>
          <div style={{ 
            fontSize: '3rem', 
            marginBottom: '1rem',
            display: 'inline-block',
            padding: '1rem',
            background: 'var(--primary-light)',
            borderRadius: '50%'
          }}>
            🤖
          </div>
          <h1 style={{ 
            fontSize: '1.75rem', 
            fontWeight: '800', 
            color: 'var(--text-primary)',
            letterSpacing: '-0.025em',
            margin: '0 0 0.5rem 0'
          }}>
            {isSignup ? 'Create Account' : 'Welcome Back'}
          </h1>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
            {isSignup ? 'Start your AutoML journey today' : 'Login to manage your ML lifecycle'}
          </p>
        </div>
        
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div>
            <label style={{ 
              display: 'block', 
              fontSize: '0.875rem', 
              fontWeight: '600', 
              marginBottom: '0.5rem',
              color: 'var(--text-primary)'
            }}>
              Email Address
            </label>
            <input
              type="email"
              placeholder="name@example.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              style={{ 
                width: '100%', 
                padding: '0.75rem 1rem', 
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-color)',
                background: 'var(--bg-primary)',
                color: 'var(--text-primary)',
                outline: 'none',
                transition: 'all var(--transition-base)'
              }}
              onFocus={(e) => e.target.style.borderColor = 'var(--primary)'}
              onBlur={(e) => e.target.style.borderColor = 'var(--border-color)'}
            />
          </div>
          
          <div>
            <label style={{ 
              display: 'block', 
              fontSize: '0.875rem', 
              fontWeight: '600', 
              marginBottom: '0.5rem',
              color: 'var(--text-primary)'
            }}>
              Password
            </label>
            <input
              type="password"
              placeholder="••••••••"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              style={{ 
                width: '100%', 
                padding: '0.75rem 1rem', 
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-color)',
                background: 'var(--bg-primary)',
                color: 'var(--text-primary)',
                outline: 'none',
                transition: 'all var(--transition-base)'
              }}
              onFocus={(e) => e.target.style.borderColor = 'var(--primary)'}
              onBlur={(e) => e.target.style.borderColor = 'var(--border-color)'}
            />
          </div>
          
          {error && (
            <div style={{ 
              padding: '0.75rem 1rem', 
              background: 'var(--danger-light)', 
              color: 'var(--danger)', 
              borderRadius: 'var(--radius-md)',
              fontSize: '0.875rem',
              fontWeight: '500',
              border: '1px solid rgba(239, 68, 68, 0.1)'
            }}>
              ⚠️ {error}
            </div>
          )}
          
          <button 
            type="submit" 
            disabled={isLoading}
            style={{ 
              width: '100%', 
              padding: '0.875rem', 
              background: 'var(--primary)', 
              color: 'white', 
              border: 'none',
              borderRadius: 'var(--radius-md)',
              fontWeight: '700',
              fontSize: '1rem',
              cursor: isLoading ? 'not-allowed' : 'pointer',
              transition: 'all var(--transition-base)',
              boxShadow: '0 4px 6px rgba(59, 130, 246, 0.2)',
              marginTop: '0.5rem',
              opacity: isLoading ? 0.7 : 1
            }}
          >
            {isLoading ? 'Processing...' : (isSignup ? 'Sign Up' : 'Sign In')}
          </button>
        </form>
        
        <div style={{ marginTop: '2rem', textAlign: 'center', borderTop: '1px solid var(--border-color)', paddingTop: '1.5rem' }}>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem' }}>
            {isSignup ? 'Already have an account?' : "Don't have an account?"}
            <button 
              onClick={() => {
                setIsSignup(!isSignup);
                setError(null);
              }} 
              style={{ 
                marginLeft: '0.5rem', 
                color: 'var(--primary)', 
                background: 'none', 
                border: 'none', 
                cursor: 'pointer',
                fontWeight: '700',
                fontSize: '0.875rem'
              }}
            >
              {isSignup ? 'Login' : 'Create one'}
            </button>
          </p>
        </div>
      </div>
    </div>
  )
}

export default LoginPage

