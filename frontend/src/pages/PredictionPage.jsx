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
    // Prepare features: convert only numeric ones to numbers
    const processedFeatures = {}
    const emptyFields = []
    
    for (const [key, value] of Object.entries(features)) {
      // Check if field is empty
      if (value === '' || value === null || value === undefined) {
        emptyFields.push(key)
        continue
      }
      
      // Check if this feature is numeric
      if (featureMetadata.numeric_features?.includes(key)) {
        // Convert to number for numeric features
        const num = parseFloat(value)
        if (isNaN(num)) {
          throw new Error(`Invalid number for ${key}: ${value}`)
        }
        processedFeatures[key] = num
      } else {
        // Keep as string for categorical features
        processedFeatures[key] = value
      }
    }

    // Check for empty fields
    if (emptyFields.length > 0) {
      throw new Error(`Please fill in all fields. Missing: ${emptyFields.slice(0, 5).join(', ')}${emptyFields.length > 5 ? '...' : ''}`)
    }

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
      // FastAPI validation error array
      setError(backendError.map(e => `${e.loc?.join('.')}: ${e.msg}`).join('; '))
    } else if (typeof backendError === 'object' && backendError !== null) {
      // Error object with error/missing/unexpected fields
      if (backendError.error) {
        const details = backendError.missing || backendError.unexpected || []
        setError(`${backendError.error}${details.length > 0 ? ': ' + details.join(', ') : ''}`)
      } else {
        // Generic object - stringify it
        setError(JSON.stringify(backendError))
      }
    } else if (typeof backendError === 'string') {
      // Simple string error
      setError(backendError)
    } else {
      // Fallback to error message
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
    <div style={{
      maxWidth: '1400px',
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
          Make Predictions
        </h1>
        <p style={{
          fontSize: '16px',
          color: 'var(--color-text-secondary)'
        }}>
          Enter values for all features to get predictions from the Production model
        </p>
      </div>

      {/* Model Info Banner */}
      {featureMetadata && (
        <div style={{
          background: 'var(--color-success-light)',
          border: '1px solid var(--color-success)',
          borderRadius: 'var(--radius-md)',
          padding: '16px 20px',
          marginBottom: '2rem',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: '12px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <Package size={24} style={{ color: 'var(--color-success)' }} />
            <div>
              <div style={{ fontSize: '14px', fontWeight: '600', color: 'var(--color-success)' }}>
                Production Model v{featureMetadata.model_version}
              </div>
              <div style={{ fontSize: '13px', color: 'var(--color-success)', marginTop: '4px' }}>
                Target: <strong>{featureMetadata.target}</strong> • 
                Features: <strong>{featureMetadata.num_features}</strong>
              </div>
            </div>
          </div>
          
          <div style={{ display: 'flex', gap: '8px' }}>
            <button
              onClick={fillWithSampleValues}
              style={{
                padding: '8px 16px',
                fontSize: '13px',
                fontWeight: '500',
                color: 'var(--color-success)',
                background: 'white',
                border: '1px solid var(--color-success)',
                borderRadius: 'var(--radius-md)',
                cursor: 'pointer'
              }}
            >
              Fill Sample Values
            </button>
            <button
              onClick={clearForm}
              style={{
                padding: '8px 16px',
                fontSize: '13px',
                fontWeight: '500',
                color: 'var(--color-danger)',
                background: 'white',
                border: '1px solid var(--color-danger)',
                borderRadius: 'var(--radius-md)',
                cursor: 'pointer'
              }}
            >
              Clear All
            </button>
          </div>
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '2.5fr 1fr', gap: '2rem' }}>
        
        {/* Left Column - Dynamic Feature Inputs */}
        <div>
          
          <div style={{
            background: 'var(--color-bg-primary)',
            borderRadius: 'var(--radius-lg)',
            padding: '2rem',
            border: '1px solid var(--color-border)',
            marginBottom: '1.5rem',
            maxHeight: '600px',
            overflowY: 'auto'
          }}>
            
            <h3 style={{
              fontSize: '18px',
              fontWeight: '600',
              marginBottom: '1.5rem',
              color: 'var(--color-text-primary)',
              position: 'sticky',
              top: 0,
              background: 'var(--color-bg-primary)',
              paddingBottom: '1rem',
              borderBottom: '1px solid var(--color-border)'
            }}>
              Feature Values ({featureMetadata?.features.length} required)
            </h3>

            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(auto-fill, minmax(250px, 1fr))',
              gap: '1rem'
            }}>
              {featureMetadata?.features.map((featureName) => (
                <div key={featureName}>
                  <label style={{
                    display: 'block',
                    fontSize: '13px',
                    fontWeight: '500',
                    color: 'var(--color-text-secondary)',
                    marginBottom: '6px'
                  }}>
                    {featureName}
                    {featureMetadata.numeric_features?.includes(featureName) && 
                      <span style={{ fontSize: '11px', marginLeft: '4px', opacity: 0.7 }}>(numeric)</span>
                    }
                    {featureMetadata.categorical_features?.includes(featureName) && 
                      <span style={{ fontSize: '11px', marginLeft: '4px', opacity: 0.7 }}>(categorical)</span>
                    }
                  </label>
                  <input
                    type="text"
                    value={features[featureName] || ''}
                    onChange={(e) => handleFeatureChange(featureName, e.target.value)}
                    placeholder="Enter value"
                    style={{
                      width: '100%',
                      padding: '10px 12px',
                      fontSize: '14px',
                      border: `1px solid ${features[featureName] ? 'var(--color-success)' : 'var(--color-border)'}`,
                      borderRadius: 'var(--radius-md)',
                      outline: 'none',
                      transition: 'border-color 0.2s'
                    }}
                    onFocus={(e) => e.target.style.borderColor = 'var(--color-primary)'}
                    onBlur={(e) => e.target.style.borderColor = features[featureName] ? 'var(--color-success)' : 'var(--color-border)'}
                  />
                </div>
              ))}
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div style={{
              background: 'var(--color-danger-light)',
              border: '1px solid var(--color-danger)',
              borderRadius: 'var(--radius-md)',
              padding: '12px 16px',
              marginBottom: '1.5rem',
              display: 'flex',
              alignItems: 'center',
              gap: '12px'
            }}>
              <AlertCircle size={20} style={{ color: 'var(--color-danger)', flexShrink: 0 }} />
              <p style={{ fontSize: '14px', color: 'var(--color-danger)', margin: 0 }}>
                {error}
              </p>
            </div>
          )}

          {/* Predict Button */}
          <button
            onClick={handlePredict}
            disabled={loading}
            style={{
              width: '100%',
              padding: '16px 24px',
              fontSize: '16px',
              fontWeight: '600',
              color: 'white',
              background: loading ? 'var(--color-text-tertiary)' : 'var(--color-primary)',
              border: 'none',
              borderRadius: 'var(--radius-md)',
              cursor: loading ? 'not-allowed' : 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: '10px',
              transition: 'all 0.2s'
            }}
          >
            {loading ? (
              <>
                <Loader size={20} style={{ animation: 'spin 1s linear infinite' }} />
                Predicting...
              </>
            ) : (
              <>
                <Sparkles size={20} />
                Get Prediction
              </>
            )}
          </button>

        </div>

        {/* Right Column - Results */}
        <div>
          
          {/* Prediction Result */}
          {prediction && (
            <div style={{
              background: 'var(--color-success-light)',
              border: '2px solid var(--color-success)',
              borderRadius: 'var(--radius-lg)',
              padding: '1.5rem',
              marginBottom: '2rem',
              animation: 'slideUp 0.3s ease'
            }}>
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '12px',
                marginBottom: '1rem'
              }}>
                <CheckCircle size={24} style={{ color: 'var(--color-success)' }} />
                <h3 style={{
                  fontSize: '18px',
                  fontWeight: '600',
                  color: 'var(--color-success)',
                  margin: 0
                }}>
                  Prediction Result
                </h3>
              </div>

              <div style={{
                background: 'white',
                borderRadius: 'var(--radius-md)',
                padding: '1.5rem',
                marginBottom: '1rem'
              }}>
                <div style={{
                  fontSize: '13px',
                  color: 'var(--color-text-secondary)',
                  marginBottom: '8px',
                  textTransform: 'uppercase',
                  fontWeight: '600'
                }}>
                  {prediction.target || 'Predicted Value'}
                </div>
                <div style={{
                  fontSize: '36px',
                  fontWeight: '700',
                  color: 'var(--color-success)'
                }}>
                  {prediction.prediction?.toLocaleString(undefined, {maximumFractionDigits: 2})}
                </div>
              </div>

              <div style={{
                fontSize: '12px',
                color: 'var(--color-text-secondary)',
                background: 'white',
                borderRadius: 'var(--radius-md)',
                padding: '12px'
              }}>
                <div style={{ marginBottom: '4px' }}>
                  <strong>Algorithm:</strong> {prediction.model_algorithm || 'N/A'}
                </div>
                <div style={{ marginBottom: '4px' }}>
                  <strong>Model:</strong> v{prediction.model_version}
                </div>
                <div>
                  <strong>R² Score:</strong> {prediction.model_r2_score?.toFixed(3) || 'N/A'}
                </div>
              </div>
            </div>
          )}

          {/* Prediction History */}
          {history.length > 0 && (
            <div style={{
              background: 'var(--color-bg-primary)',
              borderRadius: 'var(--radius-lg)',
              padding: '1.5rem',
              border: '1px solid var(--color-border)'
            }}>
              <h3 style={{
                fontSize: '16px',
                fontWeight: '600',
                marginBottom: '1rem',
                color: 'var(--color-text-primary)'
              }}>
                Recent Predictions
              </h3>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                {history.slice(0, 5).map((pred, idx) => (
                  <div
                    key={idx}
                    style={{
                      background: 'var(--color-bg-secondary)',
                      borderRadius: 'var(--radius-md)',
                      padding: '12px',
                      fontSize: '13px'
                    }}
                  >
                    <div style={{
                      fontWeight: '600',
                      color: 'var(--color-success)',
                      marginBottom: '4px'
                    }}>
                      {pred.prediction?.toLocaleString(undefined, {maximumFractionDigits: 2})}
                    </div>
                    <div style={{ color: 'var(--color-text-secondary)', fontSize: '12px' }}>
                      {new Date(pred.timestamp).toLocaleString()}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

        </div>
      </div>

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