import { useState } from 'react'
import UploadPage from './pages/UploadPage'
import TrainingPage from './pages/TrainingPage'
import ModelsPage from './pages/ModelsPage'

function App() {
  const [currentPage, setCurrentPage] = useState('upload')
  const [uploadedDataset, setUploadedDataset] = useState(null)

  const handleUploadSuccess = (result) => {
    setUploadedDataset(result)
    console.log('Dataset uploaded:', result)
    setCurrentPage('training')
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: 'var(--color-bg-secondary)'
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
          <button
            onClick={() => setCurrentPage('upload')}
            style={{
              padding: '8px 16px',
              fontSize: '14px',
              fontWeight: '500',
              color: currentPage === 'upload' ? 'var(--color-primary)' : 'var(--color-text-secondary)',
              background: currentPage === 'upload' ? 'var(--color-primary-light)' : 'transparent',
              border: 'none',
              borderRadius: 'var(--radius-md)',
              cursor: 'pointer',
              transition: 'all var(--transition-fast)'
            }}
          >
            📁 Upload
          </button>
          
          <button
            onClick={() => setCurrentPage('training')}
            style={{
              padding: '8px 16px',
              fontSize: '14px',
              fontWeight: '500',
              color: currentPage === 'training' ? 'var(--color-primary)' : 'var(--color-text-secondary)',
              background: currentPage === 'training' ? 'var(--color-primary-light)' : 'transparent',
              border: 'none',
              borderRadius: 'var(--radius-md)',
              cursor: 'pointer',
              transition: 'all var(--transition-fast)'
            }}
          >
            🚀 Training
          </button>

          <button
            onClick={() => setCurrentPage('models')}
            style={{
              padding: '8px 16px',
              fontSize: '14px',
              fontWeight: '500',
              color: currentPage === 'models' ? 'var(--color-primary)' : 'var(--color-text-secondary)',
              background: currentPage === 'models' ? 'var(--color-primary-light)' : 'transparent',
              border: 'none',
              borderRadius: 'var(--radius-md)',
              cursor: 'pointer',
              transition: 'all var(--transition-fast)'
            }}
          >
            📦 Models
          </button>
        </div>
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
      
    </div>
  )
}

export default App