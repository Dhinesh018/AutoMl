import React, { useState, useEffect } from 'react'
import { 
  Package, 
  TrendingUp, 
  ArrowUp, 
  RotateCcw,
  GitCompare,
  CheckCircle,
  Archive,
  AlertCircle,
  Award
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
    <div className="animate-fade-in" style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem 1rem',
    }}>
      
      {/* Header */}
      <div style={{ marginBottom: '3rem', display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end' }}>
        <div>
          <h1 style={{
            fontSize: '2.5rem',
            fontWeight: '900',
            marginBottom: '0.75rem',
            color: 'var(--text-primary)',
            letterSpacing: '-0.025em'
          }}>
            Model <span style={{ color: 'var(--primary)' }}>Registry</span>
          </h1>
          <p style={{
            fontSize: '1.125rem',
            color: 'var(--text-secondary)',
            maxWidth: '600px'
          }}>
            Version control for your machine learning assets. Compare, promote, and manage production-ready models.
          </p>
        </div>

        {/* Quick Actions */}
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button
            onClick={() => {
              const v1 = prompt('Enter first version number:')
              const v2 = prompt('Enter second version number:')
              if (v1 && v2) handleCompare(v1, v2)
            }}
            style={{
              padding: '0.75rem 1.5rem',
              fontSize: '0.9375rem',
              fontWeight: '700',
              color: 'var(--primary)',
              background: 'var(--bg-primary)',
              border: '1px solid var(--primary)',
              borderRadius: 'var(--radius-lg)',
              cursor: 'pointer',
              transition: 'all var(--transition-base)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              boxShadow: 'var(--shadow-sm)'
            }}
          >
            <GitCompare size={18} />
            Compare Versions
          </button>

          <button
            onClick={handleRollback}
            disabled={!productionVersion}
            style={{
              padding: '0.75rem 1.5rem',
              fontSize: '0.9375rem',
              fontWeight: '700',
              color: 'white',
              background: !productionVersion ? 'var(--text-tertiary)' : 'var(--danger)',
              border: 'none',
              borderRadius: 'var(--radius-lg)',
              cursor: !productionVersion ? 'not-allowed' : 'pointer',
              transition: 'all var(--transition-base)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              boxShadow: !productionVersion ? 'none' : '0 4px 12px rgba(239, 68, 68, 0.2)'
            }}
          >
            <RotateCcw size={18} />
            Rollback Prod
          </button>
        </div>
      </div>

      {error && (
        <div className="card animate-slide-up" style={{
          background: 'var(--danger-light)',
          padding: '1rem 1.5rem',
          marginBottom: '2rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
          border: '1px solid rgba(239, 68, 68, 0.1)'
        }}>
          <AlertCircle size={20} style={{ color: 'var(--danger)' }} />
          <p style={{ fontSize: '0.9375rem', color: 'var(--danger)', margin: 0, fontWeight: '500' }}>{error}</p>
        </div>
      )}

      {/* Models Table */}
      <div className="card" style={{
        background: 'var(--bg-primary)',
        overflow: 'hidden',
        border: '1px solid var(--border-color)',
        boxShadow: 'var(--shadow-lg)'
      }}>
        {models.length === 0 ? (
          <div style={{ padding: '6rem 2rem', textAlign: 'center' }}>
            <div style={{ width: '80px', height: '80px', background: 'var(--bg-secondary)', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', margin: '0 auto 1.5rem', color: 'var(--text-tertiary)' }}>
              <Package size={40} />
            </div>
            <h3 style={{ fontSize: '1.25rem', fontWeight: '800', color: 'var(--text-primary)' }}>No Models Registered</h3>
            <p style={{ color: 'var(--text-tertiary)', marginTop: '0.5rem' }}>Complete a training job to populate the registry.</p>
          </div>
        ) : (
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ background: 'var(--bg-secondary)', borderBottom: '1px solid var(--border-color)' }}>
                  <th style={{ padding: '1.25rem 1.5rem', fontSize: '0.75rem', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-tertiary)' }}>Version</th>
                  <th style={{ padding: '1.25rem 1.5rem', fontSize: '0.75rem', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-tertiary)' }}>Algorithm</th>
                  <th style={{ padding: '1.25rem 1.5rem', fontSize: '0.75rem', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-tertiary)' }}>R² Score</th>
                  <th style={{ padding: '1.25rem 1.5rem', fontSize: '0.75rem', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-tertiary)' }}>RMSE</th>
                  <th style={{ padding: '1.25rem 1.5rem', fontSize: '0.75rem', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-tertiary)' }}>Stage</th>
                  <th style={{ padding: '1.25rem 1.5rem', fontSize: '0.75rem', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.05em', color: 'var(--text-tertiary)' }}>Deployment Date</th>
                  <th style={{ padding: '1.25rem 1.5rem', textAlign: 'right' }}></th>
                </tr>
              </thead>
              <tbody>
                {models.map((model, index) => (
                  <tr
                    key={model.version}
                    style={{
                      borderBottom: index < models.length - 1 ? '1px solid var(--border-color)' : 'none',
                      transition: 'background var(--transition-base)'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'var(--bg-secondary)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                  >
                    <td style={{ padding: '1.25rem 1.5rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <div style={{ width: '32px', height: '32px', background: 'var(--primary-light)', color: 'var(--primary)', borderRadius: 'var(--radius-sm)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: '800', fontSize: '0.75rem' }}>
                          v{model.version}
                        </div>
                      </div>
                    </td>
                    <td style={{ padding: '1.25rem 1.5rem', fontWeight: '700', color: 'var(--text-primary)' }}>{model.algorithm}</td>
                    <td style={{ padding: '1.25rem 1.5rem' }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span style={{ fontWeight: '800', color: 'var(--success)' }}>{model.r2_score?.toFixed(4) || '—'}</span>
                        <TrendingUp size={14} style={{ color: 'var(--success)' }} />
                      </div>
                    </td>
                    <td style={{ padding: '1.25rem 1.5rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>{model.rmse?.toFixed(2) || '—'}</td>
                    <td style={{ padding: '1.25rem 1.5rem' }}>
                      <span style={{
                        display: 'inline-flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        padding: '0.375rem 0.875rem',
                        borderRadius: '100px',
                        background: `${getStageColor(model.stage)}15`,
                        color: getStageColor(model.stage),
                        fontSize: '0.75rem',
                        fontWeight: '800',
                        textTransform: 'uppercase',
                        letterSpacing: '0.025em'
                      }}>
                        {getStageIcon(model.stage)}
                        {model.stage}
                      </span>
                    </td>
                    <td style={{ padding: '1.25rem 1.5rem', color: 'var(--text-tertiary)', fontSize: '0.875rem' }}>
                      {new Date(model.created_at).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}
                    </td>
                    <td style={{ padding: '1.25rem 1.5rem', textAlign: 'right' }}>
                      {model.stage !== 'Production' && (
                        <button
                          onClick={() => handlePromote(model.version, 'Production')}
                          disabled={promoting === model.version}
                          style={{
                            padding: '0.5rem 1rem',
                            fontSize: '0.8125rem',
                            fontWeight: '700',
                            color: 'white',
                            background: 'var(--primary)',
                            border: 'none',
                            borderRadius: 'var(--radius-md)',
                            cursor: promoting === model.version ? 'not-allowed' : 'pointer',
                            display: 'inline-flex',
                            alignItems: 'center',
                            gap: '0.375rem',
                            transition: 'all var(--transition-base)',
                            opacity: promoting === model.version ? 0.5 : 1,
                            boxShadow: '0 2px 8px rgba(59, 130, 246, 0.2)'
                          }}
                        >
                          <ArrowUp size={14} />
                          {promoting === model.version ? 'Promoting...' : 'Promote'}
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
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

const ComparisonModal = ({ result, onClose }) => (
  <div style={{
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    background: 'rgba(15, 23, 42, 0.8)',
    backdropFilter: 'blur(8px)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
    padding: '1.5rem'
  }}
  onClick={onClose}
  >
    <div className="card animate-slide-up" style={{
      background: 'var(--bg-primary)',
      padding: '2.5rem',
      maxWidth: '800px',
      width: '100%',
      maxHeight: '90vh',
      overflow: 'auto',
      boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)'
    }}
    onClick={(e) => e.stopPropagation()}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: '2.5rem' }}>
        <div>
          <h2 style={{ fontSize: '1.75rem', fontWeight: '900', color: 'var(--text-primary)', margin: 0, letterSpacing: '-0.025em' }}>Head-to-Head Comparison</h2>
          <p style={{ color: 'var(--text-secondary)', marginTop: '0.25rem' }}>Analyzing performance metrics between v{result.version_a?.version} and v{result.version_b?.version}</p>
        </div>
        <button onClick={onClose} style={{ background: 'var(--bg-secondary)', border: 'none', width: '36px', height: '36px', borderRadius: '50%', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--text-secondary)' }}>✕</button>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr auto 1fr', gap: '2rem', alignItems: 'center', marginBottom: '2.5rem' }}>
        {/* Version A */}
        <div style={{ 
          padding: '2rem', 
          background: 'var(--bg-secondary)', 
          borderRadius: 'var(--radius-xl)',
          border: result.winner === result.version_a?.version ? '2px solid var(--success)' : '1px solid var(--border-color)',
          textAlign: 'center',
          position: 'relative'
        }}>
          {result.winner === result.version_a?.version && <div style={{ position: 'absolute', top: '-12px', left: '50%', transform: 'translateX(-50%)', background: 'var(--success)', color: 'white', padding: '0.25rem 0.75rem', borderRadius: '100px', fontSize: '0.625rem', fontWeight: '900', textTransform: 'uppercase' }}>Best Performer</div>}
          <div style={{ fontSize: '0.75rem', fontWeight: '800', color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Version {result.version_a?.version}</div>
          <div style={{ fontSize: '1.25rem', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '1.5rem' }}>{result.version_a?.algorithm}</div>
          
          <div style={{ display: 'grid', gap: '1rem' }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', fontWeight: '600' }}>R² SCORE</div>
              <div style={{ fontSize: '1.5rem', fontWeight: '900', color: 'var(--success)' }}>{result.version_a?.r2_score?.toFixed(4)}</div>
            </div>
            <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', fontWeight: '600' }}>RMSE</div>
              <div style={{ fontSize: '1.125rem', fontWeight: '800', color: 'var(--text-primary)' }}>{result.version_a?.rmse?.toFixed(2)}</div>
            </div>
          </div>
        </div>

        <div style={{ color: 'var(--text-tertiary)', fontWeight: '900', fontSize: '1.5rem' }}>VS</div>

        {/* Version B */}
        <div style={{ 
          padding: '2rem', 
          background: 'var(--bg-secondary)', 
          borderRadius: 'var(--radius-xl)',
          border: result.winner === result.version_b?.version ? '2px solid var(--success)' : '1px solid var(--border-color)',
          textAlign: 'center',
          position: 'relative'
        }}>
          {result.winner === result.version_b?.version && <div style={{ position: 'absolute', top: '-12px', left: '50%', transform: 'translateX(-50%)', background: 'var(--success)', color: 'white', padding: '0.25rem 0.75rem', borderRadius: '100px', fontSize: '0.625rem', fontWeight: '900', textTransform: 'uppercase' }}>Best Performer</div>}
          <div style={{ fontSize: '0.75rem', fontWeight: '800', color: 'var(--text-tertiary)', textTransform: 'uppercase', marginBottom: '0.5rem' }}>Version {result.version_b?.version}</div>
          <div style={{ fontSize: '1.25rem', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '1.5rem' }}>{result.version_b?.algorithm}</div>
          
          <div style={{ display: 'grid', gap: '1rem' }}>
            <div>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', fontWeight: '600' }}>R² SCORE</div>
              <div style={{ fontSize: '1.5rem', fontWeight: '900', color: 'var(--success)' }}>{result.version_b?.r2_score?.toFixed(4)}</div>
            </div>
            <div style={{ borderTop: '1px solid var(--border-color)', paddingTop: '1rem' }}>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', fontWeight: '600' }}>RMSE</div>
              <div style={{ fontSize: '1.125rem', fontWeight: '800', color: 'var(--text-primary)' }}>{result.version_b?.rmse?.toFixed(2)}</div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ background: 'var(--bg-secondary)', borderRadius: 'var(--radius-lg)', padding: '1.5rem', display: 'flex', alignItems: 'center', gap: '1rem', border: '1px solid var(--border-color)' }}>
        <div style={{ width: '40px', height: '40px', background: 'var(--success)', color: 'white', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}><Award size={24} /></div>
        <p style={{ margin: 0, fontSize: '0.9375rem', color: 'var(--text-primary)' }}>
          <strong>Recommendation:</strong> Promoting <strong>Version {result.winner}</strong> will likely improve production performance by <strong>{Math.abs(((result.version_a?.r2_score - result.version_b?.r2_score) / result.version_a?.r2_score) * 100).toFixed(1)}%</strong>.
        </p>
      </div>
    </div>
  </div>
)

export default ModelsPage