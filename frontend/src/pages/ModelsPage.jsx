import React, { useState, useEffect } from 'react'
import { 
  Package, 
  TrendingUp, 
  ArrowUp, 
  RotateCcw,
  GitCompare,
  CheckCircle,
  Archive,
  AlertCircle
} from 'lucide-react'
import { 
  listModelVersions, 
  promoteModel, 
  rollbackModel, 
  compareModels 
} from '../utils/api'

const ModelsPage = () => {
  const [models, setModels] = useState([])
  const [productionVersion, setProductionVersion] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [promoting, setPromoting] = useState(null)
  const [comparing, setComparing] = useState({ show: false, v1: null, v2: null, result: null })

  useEffect(() => {
    fetchModels()
  }, [])

  const fetchModels = async () => {
    setLoading(true)
    setError(null)
    try {
        const data = await listModelVersions()
        console.log('Models API response:', data)  // Debug log
        setModels(data.models || data.versions || [])  // Try both field names
        setProductionVersion(data.production_version || null)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to fetch models')
    } finally {
      setLoading(false)
    }
  }

  const handlePromote = async (version, stage) => {
    setPromoting(version)
    setError(null)
    try {
      await promoteModel(version, stage)
      await fetchModels()
      alert(`Model v${version} promoted to ${stage}!\n\n⚠️ Restart the API to load the new Production model.`)
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to promote model')
    } finally {
      setPromoting(null)
    }
  }

  const handleRollback = async () => {
    if (!confirm('Are you sure you want to rollback to the previous Production model?')) {
      return
    }
    
    setError(null)
    try {
      await rollbackModel()
      await fetchModels()
      alert('Rollback successful!\n\n⚠️ Restart the API to load the restored model.')
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to rollback model')
    }
  }

  const handleCompare = async (v1, v2) => {
    if (!v1 || !v2) {
      setError('Please select two models to compare')
      return
    }
    
    setError(null)
    try {
      const result = await compareModels(v1, v2)
      setComparing({ show: true, v1, v2, result })
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to compare models')
    }
  }

  const getStageColor = (stage) => {
    switch (stage) {
      case 'Production':
        return 'var(--color-success)'
      case 'Staging':
        return 'var(--color-warning)'
      case 'Archived':
        return 'var(--color-text-secondary)'
      default:
        return 'var(--color-text-secondary)'
    }
  }

  const getStageIcon = (stage) => {
    switch (stage) {
      case 'Production':
        return <CheckCircle size={16} />
      case 'Staging':
        return <TrendingUp size={16} />
      case 'Archived':
        return <Archive size={16} />
      default:
        return <Package size={16} />
    }
  }

  if (loading) {
    return (
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '2rem',
        textAlign: 'center'
      }}>
        <div style={{
          fontSize: '18px',
          color: 'var(--color-text-secondary)',
          marginTop: '4rem'
        }}>
          Loading models...
        </div>
      </div>
    )
  }

  return (
    <div style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem',
      animation: 'slideUp 0.3s ease'
    }}>
      
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{
          fontSize: '32px',
          fontWeight: '700',
          marginBottom: '0.5rem',
          color: 'var(--color-text-primary)'
        }}>
          Model Registry
        </h1>
        <p style={{
          fontSize: '16px',
          color: 'var(--color-text-secondary)'
        }}>
          Manage model versions, promote to production, and compare performance
        </p>
      </div>

      {/* Error Message */}
      {error && (
        <div style={{
          background: 'var(--color-danger-light)',
          border: '1px solid var(--color-danger)',
          borderRadius: 'var(--radius-md)',
          padding: '12px 16px',
          marginBottom: '2rem',
          display: 'flex',
          alignItems: 'center',
          gap: '12px',
          animation: 'slideUp 0.3s ease'
        }}>
          <AlertCircle size={20} style={{ color: 'var(--color-danger)' }} />
          <p style={{ fontSize: '14px', color: 'var(--color-danger)' }}>{error}</p>
        </div>
      )}

      {/* Quick Actions */}
      <div style={{
        display: 'flex',
        gap: '12px',
        marginBottom: '2rem',
        flexWrap: 'wrap'
      }}>
        <button
          onClick={handleRollback}
          disabled={!productionVersion}
          style={{
            padding: '10px 20px',
            fontSize: '14px',
            fontWeight: '500',
            color: 'white',
            background: !productionVersion ? 'var(--color-text-tertiary)' : 'var(--color-danger)',
            border: 'none',
            borderRadius: 'var(--radius-md)',
            cursor: !productionVersion ? 'not-allowed' : 'pointer',
            transition: 'all var(--transition-fast)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
        >
          <RotateCcw size={16} />
          Rollback Production
        </button>

        <button
          onClick={() => {
            const v1 = prompt('Enter first version number:')
            const v2 = prompt('Enter second version number:')
            if (v1 && v2) handleCompare(v1, v2)
          }}
          style={{
            padding: '10px 20px',
            fontSize: '14px',
            fontWeight: '500',
            color: 'var(--color-primary)',
            background: 'var(--color-primary-light)',
            border: 'none',
            borderRadius: 'var(--radius-md)',
            cursor: 'pointer',
            transition: 'all var(--transition-fast)',
            display: 'flex',
            alignItems: 'center',
            gap: '8px'
          }}
        >
          <GitCompare size={16} />
          Compare Models
        </button>
      </div>

      {/* Models Table */}
      <div style={{
        background: 'var(--color-bg-primary)',
        borderRadius: 'var(--radius-lg)',
        border: '1px solid var(--color-border)',
        overflow: 'hidden'
      }}>
        
        {models.length === 0 ? (
          <div style={{
            padding: '4rem 2rem',
            textAlign: 'center',
            color: 'var(--color-text-secondary)'
          }}>
            <Package size={48} style={{ marginBottom: '1rem', opacity: 0.3 }} />
            <p style={{ fontSize: '16px' }}>No models registered yet</p>
            <p style={{ fontSize: '14px', marginTop: '0.5rem' }}>
              Train a model to see it here
            </p>
          </div>
        ) : (
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: '14px'
          }}>
            <thead>
              <tr style={{
                background: 'var(--color-bg-secondary)',
                borderBottom: '1px solid var(--color-border)'
              }}>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: '600' }}>Version</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: '600' }}>Algorithm</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: '600' }}>R² Score</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: '600' }}>RMSE</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: '600' }}>Stage</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: '600' }}>Created</th>
                <th style={{ padding: '12px 16px', textAlign: 'left', fontWeight: '600' }}>Actions</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model, index) => (
                <tr
                  key={model.version}
                  style={{
                    borderBottom: index < models.length - 1 ? '1px solid var(--color-border)' : 'none',
                    transition: 'background var(--transition-fast)'
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.background = 'var(--color-bg-secondary)'}
                  onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                >
                  <td style={{ padding: '12px 16px', fontWeight: '500' }}>v{model.version}</td>
                  <td style={{ padding: '12px 16px' }}>{model.algorithm}</td>
                  <td style={{ padding: '12px 16px', color: 'var(--color-success)' }}>
                    {model.r2_score?.toFixed(3) || 'N/A'}
                  </td>
                  <td style={{ padding: '12px 16px' }}>
                    {model.rmse?.toFixed(2) || 'N/A'}
                  </td>
                  <td style={{ padding: '12px 16px' }}>
                    <span style={{
                      display: 'inline-flex',
                      alignItems: 'center',
                      gap: '6px',
                      padding: '4px 12px',
                      borderRadius: 'var(--radius-md)',
                      background: `${getStageColor(model.stage)}15`,
                      color: getStageColor(model.stage),
                      fontSize: '13px',
                      fontWeight: '500'
                    }}>
                      {getStageIcon(model.stage)}
                      {model.stage}
                    </span>
                  </td>
                  <td style={{ padding: '12px 16px', color: 'var(--color-text-secondary)' }}>
                    {new Date(model.created_at).toLocaleDateString()}
                  </td>
                  <td style={{ padding: '12px 16px' }}>
                    {model.stage !== 'Production' && (
                      <button
                        onClick={() => handlePromote(model.version, 'Production')}
                        disabled={promoting === model.version}
                        style={{
                          padding: '4px 12px',
                          fontSize: '12px',
                          fontWeight: '500',
                          color: 'var(--color-success)',
                          background: 'var(--color-success-light)',
                          border: 'none',
                          borderRadius: 'var(--radius-md)',
                          cursor: promoting === model.version ? 'not-allowed' : 'pointer',
                          display: 'inline-flex',
                          alignItems: 'center',
                          gap: '4px',
                          opacity: promoting === model.version ? 0.5 : 1
                        }}
                      >
                        <ArrowUp size={12} />
                        {promoting === model.version ? 'Promoting...' : 'Promote'}
                      </button>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Comparison Modal */}
      {comparing.show && comparing.result && (
        <ComparisonModal 
          result={comparing.result} 
          onClose={() => setComparing({ show: false, v1: null, v2: null, result: null })}
        />
      )}

    </div>
  )
}

// Comparison Modal Component
const ComparisonModal = ({ result, onClose }) => (
  <div style={{
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(0, 0, 0, 0.5)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
    animation: 'fadeIn 0.2s ease'
  }}
  onClick={onClose}
  >
    <div style={{
      background: 'var(--color-bg-primary)',
      borderRadius: 'var(--radius-lg)',
      padding: '2rem',
      maxWidth: '600px',
      width: '90%',
      maxHeight: '80vh',
      overflow: 'auto',
      animation: 'slideUp 0.3s ease'
    }}
    onClick={(e) => e.stopPropagation()}
    >
      <h2 style={{
        fontSize: '24px',
        fontWeight: '600',
        marginBottom: '1.5rem',
        color: 'var(--color-text-primary)'
      }}>
        Model Comparison
      </h2>

      <div style={{
        display: 'grid',
        gridTemplateColumns: '1fr 1fr',
        gap: '1rem',
        marginBottom: '1.5rem'
      }}>
        <div style={{
          background: 'var(--color-bg-secondary)',
          padding: '1rem',
          borderRadius: 'var(--radius-md)'
        }}>
          <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginBottom: '0.5rem' }}>
            Version {result.version_a?.version}
          </div>
          <div style={{ fontSize: '18px', fontWeight: '600', marginBottom: '0.5rem' }}>
            {result.version_a?.algorithm}
          </div>
          <div style={{ fontSize: '14px', color: 'var(--color-success)' }}>
            R²: {result.version_a?.r2_score?.toFixed(3)}
          </div>
          <div style={{ fontSize: '14px', color: 'var(--color-text-secondary)' }}>
            RMSE: {result.version_a?.rmse?.toFixed(2)}
          </div>
        </div>

        <div style={{
          background: 'var(--color-bg-secondary)',
          padding: '1rem',
          borderRadius: 'var(--radius-md)'
        }}>
          <div style={{ fontSize: '12px', color: 'var(--color-text-secondary)', marginBottom: '0.5rem' }}>
            Version {result.version_b?.version}
          </div>
          <div style={{ fontSize: '18px', fontWeight: '600', marginBottom: '0.5rem' }}>
            {result.version_b?.algorithm}
          </div>
          <div style={{ fontSize: '14px', color: 'var(--color-success)' }}>
            R²: {result.version_b?.r2_score?.toFixed(3)}
          </div>
          <div style={{ fontSize: '14px', color: 'var(--color-text-secondary)' }}>
            RMSE: {result.version_b?.rmse?.toFixed(2)}
          </div>
        </div>
      </div>

      {result.winner && (
        <div style={{
          background: 'var(--color-success-light)',
          border: '1px solid var(--color-success)',
          borderRadius: 'var(--radius-md)',
          padding: '1rem',
          marginBottom: '1.5rem'
        }}>
          <div style={{ fontSize: '14px', fontWeight: '500', color: 'var(--color-success)' }}>
            🏆 Winner: Version {result.winner}
          </div>
        </div>
      )}

      <button
        onClick={onClose}
        style={{
          width: '100%',
          padding: '10px',
          fontSize: '14px',
          fontWeight: '500',
          color: 'var(--color-text-primary)',
          background: 'var(--color-bg-secondary)',
          border: '1px solid var(--color-border)',
          borderRadius: 'var(--radius-md)',
          cursor: 'pointer'
        }}
      >
        Close
      </button>
    </div>
  </div>
)

export default ModelsPage