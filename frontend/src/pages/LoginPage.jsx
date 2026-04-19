import React, { useState } from 'react'
import  api  from '../utils/api'

const LoginPage = ({ onLogin }) => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [isSignup, setIsSignup] = useState(false)
  const [error, setError] = useState(null)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError(null)
    
    try {
      const endpoint = isSignup ? '/auth/signup' : '/auth/login'
      const response = await api.post(endpoint, { email, password })
      
      // Save token
      localStorage.setItem('token', response.data.access_token)
      
      // Set auth header
      api.defaults.headers.common['Authorization'] = `Bearer ${response.data.access_token}`
      
      onLogin()
    } catch (err) {
      setError(err.response?.data?.detail || 'Authentication failed')
    }
  }

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