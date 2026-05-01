import { useState, useEffect } from 'react'
import UploadPage from './pages/UploadPage'
import TrainingPage from './pages/TrainingPage'
import ModelsPage from './pages/ModelsPage'
import PredictionPage from './pages/PredictionPage'
import LoginPage from './pages/LoginPage'
import api from './utils/api';
import APIPage from './pages/APIPage'; // 1. Import the APIPage

function App() {
  const [currentPage, setCurrentPage] = useState('upload')
  const [uploadedDataset, setUploadedDataset] = useState(null)
  const [isAuthenticated, setIsAuthenticated] = useState(false)
  const [theme, setTheme] = useState(localStorage.getItem('theme') || 'light')

  const handleLogout = () => {
    localStorage.removeItem('token')
    delete api.defaults.headers.common['Authorization']
    setIsAuthenticated(false)
  }

  const handleUploadSuccess = (result) => {
    setUploadedDataset(result)
    console.log('Dataset uploaded:', result)
    setCurrentPage('training')
  }

  useEffect(() => {
    const token = localStorage.getItem('token')
    if (token) {
      api.defaults.headers.common['Authorization'] = `Bearer ${token}`
      setIsAuthenticated(true)
    }
    
    document.documentElement.setAttribute('data-theme', theme);
  }, [theme])

  const toggleTheme = () => {
    const newTheme = theme === 'light' ? 'dark' : 'light';
    setTheme(newTheme);
    localStorage.setItem('theme', newTheme);
  }

  if (!isAuthenticated) {
    return <LoginPage onLogin={() => setIsAuthenticated(true)} />
  }

  const navItems = [
    { id: 'upload', label: 'Upload', icon: '📁' },
    { id: 'training', label: 'Training', icon: '🚀' },
    { id: 'models', label: 'Models', icon: '📦' },
    { id: 'prediction', label: 'Predict', icon: '✨' },
    { id: 'api', label: 'API Keys', icon: '🔑' },
  ];

  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const toggleMobileMenu = () => setMobileMenuOpen(!mobileMenuOpen);

  return (
    <div style={{
      minHeight: '100vh',
      background: 'var(--bg-secondary)',
      display: 'flex',
      flexDirection: 'column',
      transition: 'background-color var(--transition-base)'
    }}>
      
      <nav style={{
        background: 'var(--bg-primary)',
        borderBottom: '1px solid var(--border-color)',
        padding: '0 2rem',
        height: '70px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        position: 'sticky',
        top: 0,
        zIndex: 1000,
        boxShadow: 'var(--shadow-sm)',
        transition: 'background-color var(--transition-base), border-color var(--transition-base)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '2rem' }}>
          <h2 style={{
            fontSize: '1.25rem',
            fontWeight: '900',
            color: 'var(--primary)',
            margin: 0,
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            letterSpacing: '-0.025em',
            cursor: 'pointer'
          }} onClick={() => setCurrentPage('upload')}>
            <span style={{ fontSize: '1.5rem' }}>🤖</span>
            <span className="nav-logo-text">
              <span style={{ color: 'var(--text-primary)' }}>AutoML</span>
              <span style={{ color: 'var(--primary)' }}>Assistant</span>
            </span>
          </h2>
          
          <div className="nav-links-desktop" style={{ display: 'flex', gap: '0.25rem' }}>
            {navItems.map((item) => (
              <button 
                key={item.id}
                onClick={() => setCurrentPage(item.id)} 
                style={{
                  padding: '0.5rem 0.875rem',
                  fontSize: '0.875rem',
                  fontWeight: '700',
                  color: currentPage === item.id ? 'var(--primary)' : 'var(--text-secondary)',
                  background: currentPage === item.id ? 'var(--primary-light)' : 'transparent',
                  border: 'none',
                  borderRadius: 'var(--radius-md)',
                  cursor: 'pointer',
                  transition: 'all var(--transition-base)',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.5rem'
                }}
              >
                <span>{item.icon}</span>
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div className="nav-actions-desktop" style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <button 
              onClick={toggleTheme}
              style={{
                width: '38px',
                height: '38px',
                borderRadius: '50%',
                border: '1px solid var(--border-color)',
                background: 'var(--bg-primary)',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontSize: '1.125rem',
                transition: 'all var(--transition-base)',
              }}
            >
              {theme === 'light' ? '🌙' : '☀️'}
            </button>

            <button 
              onClick={handleLogout} 
              style={{
                padding: '0.5rem 1rem',
                backgroundColor: 'var(--danger)',
                color: 'white',
                border: 'none',
                borderRadius: 'var(--radius-md)',
                cursor: 'pointer',
                fontWeight: '700',
                fontSize: '0.8125rem',
                boxShadow: '0 4px 12px rgba(239, 68, 68, 0.2)'
              }}
            >
              Logout
            </button>
          </div>

          <button 
            className="mobile-menu-toggle"
            onClick={toggleMobileMenu}
            style={{
              display: 'none',
              background: 'none',
              border: 'none',
              fontSize: '1.5rem',
              color: 'var(--text-primary)',
              cursor: 'pointer'
            }}
          >
            {mobileMenuOpen ? '✕' : '☰'}
          </button>
        </div>
      </nav>

      {/* Mobile Menu */}
      {mobileMenuOpen && (
        <div className="mobile-menu" style={{
          position: 'fixed',
          top: '70px',
          left: 0,
          right: 0,
          background: 'var(--bg-primary)',
          borderBottom: '1px solid var(--border-color)',
          zIndex: 999,
          padding: '1rem',
          display: 'flex',
          flexDirection: 'column',
          gap: '0.5rem',
          boxShadow: 'var(--shadow-lg)',
          animation: 'slideDown 0.3s ease'
        }}>
          {navItems.map((item) => (
            <button 
              key={item.id}
              onClick={() => {
                setCurrentPage(item.id);
                setMobileMenuOpen(false);
              }} 
              style={{
                padding: '1rem',
                textAlign: 'left',
                background: currentPage === item.id ? 'var(--primary-light)' : 'transparent',
                border: 'none',
                borderRadius: 'var(--radius-md)',
                color: currentPage === item.id ? 'var(--primary)' : 'var(--text-primary)',
                fontWeight: '700',
                display: 'flex',
                alignItems: 'center',
                gap: '1rem'
              }}
            >
              <span>{item.icon}</span>
              {item.label}
            </button>
          ))}
          <div style={{ height: '1px', background: 'var(--border-color)', margin: '0.5rem 0' }}></div>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.5rem 1rem' }}>
            <button onClick={toggleTheme} style={{ background: 'none', border: 'none', fontSize: '1.25rem' }}>
              {theme === 'light' ? '🌙 Dark Mode' : '☀️ Light Mode'}
            </button>
            <button onClick={handleLogout} style={{ color: 'var(--danger)', fontWeight: '700', background: 'none', border: 'none' }}>Logout</button>
          </div>
        </div>
      )}

      <main style={{ 
        flex: 1, 
        padding: '2rem',
        maxWidth: '1200px',
        margin: '0 auto',
        width: '100%',
        animation: 'fadeIn var(--transition-slow)'
      }}>
        {currentPage === 'upload' && <UploadPage onUploadSuccess={handleUploadSuccess} />}
        {currentPage === 'training' && (
          <TrainingPage 
            datasetId={uploadedDataset?.dataset_id}
            targetColumn={uploadedDataset?.target_column}
          />
        )}
        {currentPage === 'models' && <ModelsPage />}
        {currentPage === 'prediction' && <PredictionPage />}
        {currentPage === 'api' && <APIPage />}
      </main>
      
      <footer style={{
        padding: '1.5rem',
        textAlign: 'center',
        color: 'var(--text-tertiary)',
        fontSize: '0.875rem',
        borderTop: '1px solid var(--border-color)',
        background: 'var(--bg-primary)'
      }}>
        &copy; 2026 AutoML Assistant • End-to-End ML Lifecycle
      </footer>
    </div>
  )
}

export default App