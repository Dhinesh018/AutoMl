import React, { useState } from 'react'
import api from '../utils/api'

const LoginPage = ({ onLogin }) => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isSignup, setIsSignup] = useState(false) // You used 'isSignup' here...
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    try {
      // FIX: Use 'isSignup' and flip the logic
      // If isSignup is true, go to signup. Otherwise, go to login.
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
      // Backend usually sends error in err.response.data.detail
      setError(err.response?.data?.detail || 'Authentication failed');
    }
  };

  return (
    <div style={{ maxWidth: '400px', margin: '100px auto', padding: '2rem' }}>
      <h1>{isSignup ? 'Sign Up' : 'Login'}</h1>
      
      <form onSubmit={handleSubmit}>
        <input
          type="email"
          placeholder="Email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          style={{ width: '100%', padding: '10px', marginBottom: '10px' }}
        />
        
        <input
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          style={{ width: '100%', padding: '10px', marginBottom: '10px' }}
        />
        
        {error && <p style={{ color: 'red' }}>{error}</p>}
        
        <button type="submit" style={{ width: '100%', padding: '12px', background: '#3b82f6', color: 'white', border: 'none' }}>
          {isSignup ? 'Sign Up' : 'Login'}
        </button>
      </form>
      
      <p style={{ marginTop: '20px', textAlign: 'center' }}>
        {isSignup ? 'Already have an account?' : "Don't have an account?"}
        <button onClick={() => setIsSignup(!isSignup)} style={{ marginLeft: '5px', color: '#3b82f6', background: 'none', border: 'none', cursor: 'pointer' }}>
          {isSignup ? 'Login' : 'Sign Up'}
        </button>
      </p>
    </div>
  )
}

export default LoginPage