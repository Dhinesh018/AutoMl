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
  }, [])

  if (!isAuthenticated) {
    return <LoginPage onLogin={() => setIsAuthenticated(true)} />
  }

  // Helper function to create consistent button styles
  const getNavButtonStyle = (pageName) => ({
    padding: '8px 16px',
    fontSize: '14px',
    fontWeight: '500',
    color: currentPage === pageName ? 'var(--color-primary)' : 'var(--color-text-secondary)',
    background: currentPage === pageName ? 'var(--color-primary-light)' : 'transparent',
    border: 'none',
    borderRadius: 'var(--radius-md)',
    cursor: 'pointer',
    transition: 'all var(--transition-fast)'
  });

  return (
    <div style={{
      minHeight: '100vh',
      background: 'var(--color-bg-secondary)',
      position: 'relative'
    }}>
      
      {/* Navigation Header */}
      <nav style={{
        background: 'var(--color-bg-primary)',
        borderBottom: '1px solid var(--color-border)',
        padding: '1rem 2rem',
        display: 'flex',
        alignItems: 'center',
        gap: '2rem'
      }}>
        <h2 style={{
          fontSize: '20px',
          fontWeight: '700',
          color: 'var(--color-text-primary)',
          margin: 0
        }}>
          🤖 AutoML Assistant
        </h2>
        
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button onClick={() => setCurrentPage('upload')} style={getNavButtonStyle('upload')}>
            📁 Upload
          </button>
          
          <button onClick={() => setCurrentPage('training')} style={getNavButtonStyle('training')}>
            🚀 Training
          </button>

          <button onClick={() => setCurrentPage('models')} style={getNavButtonStyle('models')}>
            📦 Models
          </button>

          <button onClick={() => setCurrentPage('prediction')} style={getNavButtonStyle('prediction')}>
            ✨ Predict
          </button>

          {/* 2. Added API Navigation Button */}
          <button onClick={() => setCurrentPage('api')} style={getNavButtonStyle('api')}>
            🔑 API
          </button>
        </div>

        <button 
          onClick={handleLogout} 
          style={{
            position: 'absolute', 
            top: 20, 
            right: 20,
            padding: '8px 16px',
            backgroundColor: '#ff4d4f',
            color: 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: 'pointer',
            fontWeight: '600'
          }}
        >
          Logout
        </button>
      </nav>

      {/* Page Content */}
      {currentPage === 'upload' && (
        <UploadPage onUploadSuccess={handleUploadSuccess} />
      )}
      
      {currentPage === 'training' && (
        <TrainingPage 
          datasetId={uploadedDataset?.dataset_id}
          targetColumn={uploadedDataset?.target_column}
        />
      )}

      {currentPage === 'models' && (
        <ModelsPage />
      )}

      {currentPage === 'prediction' && (
        <PredictionPage />
      )}

      {/* 3. Added APIPage Content Section */}
      {currentPage === 'api' && (
        <APIPage />
      )}
      
    </div>
  )
}

export default App