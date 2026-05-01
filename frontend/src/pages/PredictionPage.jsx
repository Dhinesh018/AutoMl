import React, { useState, useEffect } from 'react'
import { 
  Sparkles, 
  CheckCircle, 
  AlertCircle,
  Package,
  RefreshCw,
  Loader
} from 'lucide-react'
import { makePrediction, getProductionFeatures } from '../utils/api'

const PredictionPage = () => {
  const [featureMetadata, setFeatureMetadata] = useState(null)
  const [features, setFeatures] = useState({})
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingMetadata, setLoadingMetadata] = useState(true)
  const [error, setError] = useState(null)
  const [history, setHistory] = useState([])

  useEffect(() => {
    loadFeatureSchema()
    loadHistory()
  }, [])

  const loadFeatureSchema = async () => {
  setLoadingMetadata(true)
  setError(null)
  
  try {
    const metadata = await getProductionFeatures()
    console.log('Feature metadata loaded:', metadata)
    setFeatureMetadata(metadata)
    
    // Initialize empty feature values
    const initialFeatures = {}
    metadata.features.forEach(feat => {
      initialFeatures[feat] = ''
    })
    setFeatures(initialFeatures)
    
  } catch (err) {
    console.error('Failed to load features:', err)
    
    // FIX: Extract error properly - NEVER pass objects to setError
    let errorMessage = 'Failed to load model features. Train a model first.'
    
    if (err.response?.data?.detail) {
      const detail = err.response.data.detail
      
      if (typeof detail === 'string') {
        errorMessage = detail
      } else if (Array.isArray(detail)) {
        errorMessage = detail.map(e => e.msg || JSON.stringify(e)).join('; ')
      } else if (typeof detail === 'object') {
        errorMessage = JSON.stringify(detail)
      }
    } else if (err.message) {
      errorMessage = err.message
    }
    
    setError(errorMessage)
    
  } finally {
    setLoadingMetadata(false)
  }
}

  const loadHistory = () => {
    const saved = localStorage.getItem('prediction_history')
    if (saved) {
      setHistory(JSON.parse(saved))
    }
  }

  const saveToHistory = (pred) => {
    const newHistory = [pred, ...history].slice(0, 10)
    setHistory(newHistory)
    localStorage.setItem('prediction_history', JSON.stringify(newHistory))
  }

  const handleFeatureChange = (name, value) => {
    setFeatures({ ...features, [name]: value })
  }

const handlePredict = async () => {
  setLoading(true)
  setError(null)
  setPrediction(null)

  try {
    // Prepare features with smart type handling
    const processedFeatures = {}
    const emptyFields = []
    
    for (const [key, value] of Object.entries(features)) {
      // Check if field is empty
      if (value === '' || value === null || value === undefined) {
        emptyFields.push(key)
        continue
      }
      
      // 🔥 CRITICAL: Check if feature is numeric or categorical
      if (featureMetadata.numeric_features?.includes(key)) {
        // NUMERIC: Convert to number
        const num = parseFloat(value)
        if (isNaN(num)) {
          throw new Error(`Invalid number for ${key}: ${value}`)
        }
        processedFeatures[key] = num
      } else {
        // CATEGORICAL: Keep as string
        processedFeatures[key] = value
      }
    }

    // Check for empty fields
    if (emptyFields.length > 0) {
      throw new Error(`Please fill in all fields. Missing: ${emptyFields.slice(0, 5).join(', ')}${emptyFields.length > 5 ? '...' : ''}`)
    }

    console.log('🔹 Sending to backend:', processedFeatures)

    // Make prediction
    const result = await makePrediction(processedFeatures)
    
    const predictionData = {
      ...result,
      timestamp: new Date().toISOString()
    }
    
    setPrediction(predictionData)
    saveToHistory(predictionData)
    
  } catch (err) {
    console.error('Prediction error:', err)
    
    // Extract error message properly
    const backendError = err.response?.data?.detail
    
    if (Array.isArray(backendError)) {
      setError(backendError.map(e => `${e.loc?.join('.')}: ${e.msg}`).join('; '))
    } else if (typeof backendError === 'object' && backendError !== null) {
      if (backendError.error) {
        const details = backendError.missing || backendError.unexpected || []
        setError(`${backendError.error}${details.length > 0 ? ': ' + details.join(', ') : ''}`)
      } else {
        setError(JSON.stringify(backendError))
      }
    } else if (typeof backendError === 'string') {
      setError(backendError)
    } else {
      setError(err.message || 'Prediction failed')
    }
  } finally {
    setLoading(false)
  }
}

  const fillWithSampleValues = () => {
    if (!featureMetadata) return
    
    const sampleFeatures = {}
    featureMetadata.features.forEach((feat, idx) => {
      // Generate reasonable sample values
      // Numeric features: use index-based values
      sampleFeatures[feat] = String((idx + 1) * 100)
    })
    setFeatures(sampleFeatures)
  }

  const clearForm = () => {
    const emptyFeatures = {}
    featureMetadata?.features.forEach(feat => {
      emptyFeatures[feat] = ''
    })
    setFeatures(emptyFeatures)
    setPrediction(null)
    setError(null)
  }

  // Loading state
  if (loadingMetadata) {
    return (
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '4rem 2rem',
        textAlign: 'center'
      }}>
        <Loader size={48} style={{ 
          color: 'var(--color-primary)',
          animation: 'spin 1s linear infinite'
        }} />
        <p style={{ 
          marginTop: '1rem',
          fontSize: '16px',
          color: 'var(--color-text-secondary)'
        }}>
          Loading model features...
        </p>
      </div>
    )
  }

  // Error state - no Production model
  if (!featureMetadata && error) {
    return (
      <div style={{
        maxWidth: '1200px',
        margin: '0 auto',
        padding: '4rem 2rem',
        textAlign: 'center'
      }}>
        <AlertCircle size={64} style={{ color: 'var(--color-danger)', marginBottom: '1rem' }} />
        <h2 style={{ fontSize: '24px', marginBottom: '1rem', color: 'var(--color-text-primary)' }}>
          No Production Model Found
        </h2>
        <p style={{ fontSize: '16px', color: 'var(--color-text-secondary)', marginBottom: '2rem' }}>
          {error}
        </p>
        <button
          onClick={loadFeatureSchema}
          style={{
            padding: '12px 24px',
            fontSize: '16px',
            fontWeight: '500',
            color: 'white',
            background: 'var(--color-primary)',
            border: 'none',
            borderRadius: 'var(--radius-md)',
            cursor: 'pointer',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '8px'
          }}
        >
          <RefreshCw size={20} />
          Retry
        </button>
      </div>
    )
  }

  // Main UI
  return (
    <div className="animate-fade-in" style={{
      maxWidth: '1300px',
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
            Model <span style={{ color: 'var(--primary)' }}>Inference</span>
          </h1>
          <p style={{
            fontSize: '1.125rem',
            color: 'var(--text-secondary)',
            maxWidth: '600px'
          }}>
            Interact with your production-grade model. Enter features below to receive instant, LLM-optimized predictions.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <button
            onClick={fillWithSampleValues}
            style={{
              padding: '0.625rem 1.25rem',
              fontSize: '0.875rem',
              fontWeight: '700',
              color: 'var(--primary)',
              background: 'var(--primary-light)',
              border: 'none',
              borderRadius: 'var(--radius-md)',
              cursor: 'pointer',
              transition: 'all var(--transition-base)'
            }}
          >
            Fill Samples
          </button>
          <button
            onClick={clearForm}
            style={{
              padding: '0.625rem 1.25rem',
              fontSize: '0.875rem',
              fontWeight: '700',
              color: 'var(--danger)',
              background: 'var(--danger-light)',
              border: 'none',
              borderRadius: 'var(--radius-md)',
              cursor: 'pointer',
              transition: 'all var(--transition-base)'
            }}
          >
            Clear Form
          </button>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 400px', gap: '2rem', alignItems: 'start' }}>
        
        {/* Input Area */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          <div className="card" style={{ padding: '2.5rem', background: 'var(--bg-primary)' }}>
            <h3 style={{ 
              fontSize: '1.25rem', 
              fontWeight: '800', 
              color: 'var(--text-primary)', 
              marginBottom: '2rem',
              display: 'flex',
              alignItems: 'center',
              gap: '0.75rem'
            }}>
              <Package size={24} style={{ color: 'var(--primary)' }} />
              Feature Vector Input
            </h3>

            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))',
              gap: '1.5rem 2rem'
            }}>
              {featureMetadata?.features.map((featureName) => (
                <div key={featureName} style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                  <label style={{
                    fontSize: '0.8125rem',
                    fontWeight: '700',
                    color: 'var(--text-secondary)',
                    textTransform: 'uppercase',
                    letterSpacing: '0.025em',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between'
                  }}>
                    {featureName}
                    <span style={{ 
                      fontSize: '0.625rem', 
                      background: 'var(--bg-secondary)', 
                      padding: '0.125rem 0.375rem', 
                      borderRadius: '4px',
                      color: 'var(--text-tertiary)'
                    }}>
                      {featureMetadata.numeric_features?.includes(featureName) ? 'NUM' : 'CAT'}
                    </span>
                  </label>
                  <input
                    type="text"
                    value={features[featureName] || ''}
                    onChange={(e) => handleFeatureChange(featureName, e.target.value)}
                    placeholder="Enter value..."
                    style={{
                      width: '100%',
                      padding: '0.875rem 1rem',
                      fontSize: '0.9375rem',
                      border: '1px solid var(--border-color)',
                      borderRadius: 'var(--radius-md)',
                      background: 'var(--bg-secondary)',
                      color: 'var(--text-primary)',
                      outline: 'none',
                      transition: 'all var(--transition-base)',
                    }}
                    onFocus={(e) => {
                      e.target.style.borderColor = 'var(--primary)';
                      e.target.style.background = 'var(--bg-primary)';
                      e.target.style.boxShadow = '0 0 0 4px var(--primary-light)';
                    }}
                    onBlur={(e) => {
                      e.target.style.borderColor = 'var(--border-color)';
                      e.target.style.background = 'var(--bg-secondary)';
                      e.target.style.boxShadow = 'none';
                    }}
                  />
                </div>
              ))}
            </div>

            {error && (
              <div className="animate-slide-up" style={{
                marginTop: '2rem',
                background: 'var(--danger-light)',
                padding: '1rem 1.25rem',
                borderRadius: 'var(--radius-lg)',
                border: '1px solid rgba(239, 68, 68, 0.1)',
                display: 'flex',
                gap: '0.75rem'
              }}>
                <AlertCircle size={20} style={{ color: 'var(--danger)', flexShrink: 0 }} />
                <p style={{ fontSize: '0.875rem', color: 'var(--danger)', margin: 0, fontWeight: '500' }}>{error}</p>
              </div>
            )}

            <button
              onClick={handlePredict}
              disabled={loading}
              style={{
                width: '100%',
                marginTop: '2.5rem',
                padding: '1.25rem',
                fontSize: '1.125rem',
                fontWeight: '800',
                color: 'white',
                background: loading ? 'var(--text-tertiary)' : 'var(--primary)',
                border: 'none',
                borderRadius: 'var(--radius-lg)',
                cursor: loading ? 'not-allowed' : 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.75rem',
                transition: 'all var(--transition-base)',
                boxShadow: '0 10px 15px -3px rgba(59, 130, 246, 0.3)'
              }}
            >
              {loading ? (
                <>
                  <div style={{ width: '20px', height: '20px', border: '3px solid rgba(255,255,255,0.3)', borderTopColor: 'white', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }}></div>
                  Processing...
                </>
              ) : (
                <>
                  <Sparkles size={20} />
                  Generate Prediction
                </>
              )}
            </button>
          </div>
        </div>

        {/* Results Column */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {/* Active Result */}
          {prediction ? (
            <div className="card animate-slide-up" style={{ 
              background: 'linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%)',
              padding: '2.5rem',
              color: 'white',
              border: 'none',
              textAlign: 'center',
              boxShadow: '0 20px 40px -10px rgba(59, 130, 246, 0.4)'
            }}>
              <CheckCircle size={48} style={{ marginBottom: '1.5rem', opacity: 0.9 }} />
              <div style={{ fontSize: '0.875rem', fontWeight: '800', textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '0.5rem', opacity: 0.8 }}>
                Predicted Result
              </div>
              <div style={{ fontSize: '3.5rem', fontWeight: '900', marginBottom: '1rem', letterSpacing: '-0.02em' }}>
                {prediction.prediction?.toLocaleString(undefined, {maximumFractionDigits: 2})}
              </div>
              <div style={{ 
                background: 'rgba(255,255,255,0.1)', 
                backdropFilter: 'blur(10px)', 
                padding: '1rem', 
                borderRadius: 'var(--radius-lg)',
                fontSize: '0.8125rem',
                textAlign: 'left'
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ opacity: 0.7 }}>Algorithm</span>
                  <span style={{ fontWeight: '700' }}>{prediction.model_algorithm}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                  <span style={{ opacity: 0.7 }}>Confidence (R²)</span>
                  <span style={{ fontWeight: '700' }}>{prediction.model_r2_score?.toFixed(4)}</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <span style={{ opacity: 0.7 }}>Version</span>
                  <span style={{ fontWeight: '700' }}>v{prediction.model_version}</span>
                </div>
              </div>
            </div>
          ) : (
            <div className="card" style={{ padding: '2.5rem', textAlign: 'center', background: 'var(--bg-primary)', border: '1px dashed var(--border-color)' }}>
              <Sparkles size={48} style={{ color: 'var(--text-tertiary)', marginBottom: '1.5rem', opacity: 0.5 }} />
              <h4 style={{ color: 'var(--text-secondary)', fontWeight: '700' }}>Awaiting Input</h4>
              <p style={{ color: 'var(--text-tertiary)', fontSize: '0.875rem', marginTop: '0.5rem' }}>Predictions will appear here once you submit the form.</p>
            </div>
          )}

          {/* History */}
          <div className="card" style={{ padding: '2rem', background: 'var(--bg-primary)' }}>
            <h4 style={{ fontSize: '1rem', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '1.5rem' }}>Recent History</h4>
            {history.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                {history.slice(0, 5).map((pred, idx) => (
                  <div key={idx} style={{ 
                    padding: '1rem', 
                    background: 'var(--bg-secondary)', 
                    borderRadius: 'var(--radius-md)',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center'
                  }}>
                    <div>
                      <div style={{ fontSize: '1.125rem', fontWeight: '900', color: 'var(--primary)' }}>{pred.prediction?.toFixed(2)}</div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-tertiary)', marginTop: '0.25rem' }}>{new Date(pred.timestamp).toLocaleTimeString()}</div>
                    </div>
                    <div style={{ fontSize: '0.75rem', fontWeight: '700', color: 'var(--text-secondary)' }}>v{pred.model_version}</div>
                  </div>
                ))}
              </div>
            ) : (
              <p style={{ textAlign: 'center', color: 'var(--text-tertiary)', fontSize: '0.875rem', padding: '1rem 0' }}>No history available</p>
            )}
          </div>
        </div>
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}} />
    </div>
  )
}

// Add CSS animation
const style = document.createElement('style')
style.textContent = `
  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }
`
document.head.appendChild(style)

export default PredictionPage