import React, { useState, useRef } from 'react'
import { Upload, FileText, AlertCircle, CheckCircle } from 'lucide-react'
import { uploadDataset } from '../utils/api'

const UploadPage = ({ onUploadSuccess }) => {
  const [file, setFile] = useState(null)
  const [targetColumn, setTargetColumn] = useState('')
  const [dragActive, setDragActive] = useState(false)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState(null)
  const [uploadResult, setUploadResult] = useState(null)
  
  // 1. Added useRef for the file input
  const fileInputRef = useRef(null)

  const handleDrag = (e) => {
    e.preventDefault()
    e.stopPropagation()
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true)
    } else if (e.type === 'dragleave') {
      setDragActive(false)
    }
  }

  const handleDrop = (e) => {
    e.preventDefault()
    e.stopPropagation()
    setDragActive(false)
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0])
    }
  }

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0])
    }
  }

  const handleFile = (selectedFile) => {
    const validExtensions = ['csv', 'xlsx', 'xls']
    const fileExtension = selectedFile.name.split('.').pop().toLowerCase()
    
    if (!validExtensions.includes(fileExtension)) {
      setError('Please upload a CSV or Excel file')
      return
    }
    
    if (selectedFile.size > 100 * 1024 * 1024) {
      setError('File size must be less than 100MB')
      return
    }
    
    setFile(selectedFile)
    setError(null)
    setUploadResult(null)
  }

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file')
      return
    }
    
    if (!targetColumn.trim()) {
      setError('Please enter a target column name')
      return
    }
    
    setUploading(true)
    setError(null)
    
    try {
      const result = await uploadDataset(file, targetColumn)
      setUploadResult(result)
      
      if (onUploadSuccess) {
        onUploadSuccess(result)
      }
    } catch (err) {
      // 2. Safely parse the backend error object into a string
      const data = err.response?.data;
      let errorMessage = 'Upload failed. Please try again.';

      if (data && typeof data === 'object') {
        if (data.message) {
          errorMessage = data.message;
          // Append the helpful suggestion from your backend if it exists
          if (data.suggestion) {
            errorMessage += ` Suggestion: ${data.suggestion}`;
          }
        } else if (data.error) {
          errorMessage = typeof data.error === 'string' ? data.error : 'An error occurred';
        } else if (data.detail) {
          errorMessage = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
        }
      }
      
      setError(errorMessage);
    } finally {
      setUploading(false)
    }
  }

  // 3. Trigger the click using the React Ref safely
  const handleAreaClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  }

  return (
    <div className="animate-fade-in" style={{
      maxWidth: '900px',
      margin: '0 auto',
      padding: '2rem 1rem',
    }}>
      
      {/* Header */}
      <div style={{ marginBottom: '3rem', textAlign: 'center' }}>
        <h1 style={{
          fontSize: '2.5rem',
          fontWeight: '900',
          marginBottom: '0.75rem',
          color: 'var(--text-primary)',
          letterSpacing: '-0.025em'
        }}>
          Upload Your <span style={{ color: 'var(--primary)' }}>Dataset</span>
        </h1>
        <p style={{
          fontSize: '1.125rem',
          color: 'var(--text-secondary)',
          maxWidth: '600px',
          margin: '0 auto'
        }}>
          Bring your data to life. Our LLM-powered assistant will analyze your schema and recommend the best models.
        </p>
      </div>

      <div className="card" style={{ padding: '2.5rem', background: 'var(--bg-primary)', boxShadow: 'var(--shadow-lg)' }}>
        {/* Upload Area */}
        <div
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
          style={{
            background: dragActive ? 'var(--primary-light)' : 'var(--bg-secondary)',
            border: `2px dashed ${dragActive ? 'var(--primary)' : 'var(--border-color)'}`,
            borderRadius: 'var(--radius-xl)',
            padding: '4rem 2rem',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all var(--transition-base)',
            marginBottom: '2rem',
            position: 'relative',
            overflow: 'hidden'
          }}
          onClick={handleAreaClick}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileInput}
            style={{ display: 'none' }}
          />
          
          {!file ? (
            <div className="animate-fade-in">
              <div style={{ 
                width: '80px', 
                height: '80px', 
                background: 'var(--bg-primary)', 
                borderRadius: '50%', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                margin: '0 auto 1.5rem',
                boxShadow: 'var(--shadow-md)',
                color: 'var(--primary)'
              }}>
                <Upload size={32} />
              </div>
              <p style={{
                fontSize: '1.25rem',
                fontWeight: '700',
                color: 'var(--text-primary)',
                marginBottom: '0.5rem'
              }}>
                {dragActive ? 'Drop it like it\'s hot!' : 'Drop your data here'}
              </p>
              <p style={{
                fontSize: '0.9375rem',
                color: 'var(--text-secondary)'
              }}>
                Click to browse or drag and drop CSV, XLSX (max 100MB)
              </p>
            </div>
          ) : (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              animation: 'fadeIn 0.4s ease'
            }}>
              <div style={{ 
                width: '64px', 
                height: '64px', 
                background: 'var(--primary-light)', 
                borderRadius: 'var(--radius-lg)', 
                display: 'flex', 
                alignItems: 'center', 
                justifyContent: 'center',
                marginBottom: '1rem',
                color: 'var(--primary)'
              }}>
                <FileText size={32} />
              </div>
              <div style={{ textAlign: 'center' }}>
                <p style={{
                  fontSize: '1.125rem',
                  fontWeight: '700',
                  color: 'var(--text-primary)',
                  marginBottom: '0.25rem'
                }}>
                  {file.name}
                </p>
                <p style={{
                  fontSize: '0.875rem',
                  color: 'var(--text-secondary)',
                  background: 'var(--bg-tertiary)',
                  padding: '0.25rem 0.75rem',
                  borderRadius: '100px',
                  display: 'inline-block'
                }}>
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
              <button 
                onClick={(e) => { e.stopPropagation(); setFile(null); }}
                style={{
                  marginTop: '1.5rem',
                  background: 'none',
                  border: 'none',
                  color: 'var(--danger)',
                  fontSize: '0.875rem',
                  fontWeight: '600',
                  cursor: 'pointer',
                  textDecoration: 'underline'
                }}
              >
                Change File
              </button>
            </div>
          )}
        </div>

        {/* Form Controls */}
        <div style={{ display: 'grid', gap: '1.5rem' }}>
          {file && !uploadResult && (
            <div className="animate-slide-up">
              <label style={{
                display: 'block',
                fontSize: '0.9375rem',
                fontWeight: '600',
                color: 'var(--text-primary)',
                marginBottom: '0.75rem'
              }}>
                What's the Target Column?
              </label>
              <div style={{ position: 'relative' }}>
                <input
                  type="text"
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                  placeholder="e.g., SalePrice, Churn, Category"
                  style={{
                    width: '100%',
                    padding: '1rem 1.25rem',
                    fontSize: '1rem',
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
              <p style={{
                fontSize: '0.8125rem',
                color: 'var(--text-tertiary)',
                marginTop: '0.5rem',
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem'
              }}>
                <AlertCircle size={14} /> The variable your model will learn to predict
              </p>
            </div>
          )}

          {/* Action Button */}
          {file && !uploadResult && (
            <button
              onClick={(e) => { e.stopPropagation(); handleUpload(); }}
              disabled={uploading}
              className="animate-slide-up"
              style={{
                width: '100%',
                padding: '1.25rem',
                fontSize: '1.125rem',
                fontWeight: '700',
                color: 'white',
                background: uploading ? 'var(--text-tertiary)' : 'var(--primary)',
                border: 'none',
                borderRadius: 'var(--radius-lg)',
                cursor: uploading ? 'not-allowed' : 'pointer',
                transition: 'all var(--transition-base)',
                boxShadow: 'var(--shadow-md)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '0.75rem'
              }}
            >
              {uploading ? (
                <>
                  <div style={{ 
                    width: '20px', 
                    height: '20px', 
                    border: '3px solid rgba(255,255,255,0.3)', 
                    borderTopColor: 'white', 
                    borderRadius: '50%', 
                    animation: 'spin 0.8s linear infinite' 
                  }}></div>
                  Uploading...
                </>
              ) : (
                <>
                  <CheckCircle size={20} />
                  Process Dataset
                </>
              )}
            </button>
          )}
        </div>

        {/* Status Messages */}
        {error && (
          <div className="animate-slide-up" style={{
            marginTop: '2rem',
            background: 'var(--danger-light)',
            border: '1px solid rgba(239, 68, 68, 0.2)',
            borderRadius: 'var(--radius-lg)',
            padding: '1rem 1.25rem',
            display: 'flex',
            alignItems: 'flex-start',
            gap: '0.75rem',
          }}>
            <AlertCircle size={20} style={{ color: 'var(--danger)', flexShrink: 0, marginTop: '2px' }} />
            <p style={{ fontSize: '0.9375rem', color: 'var(--danger)', margin: 0, lineHeight: 1.5 }}>
              <strong>Upload Error:</strong> {error}
            </p>
          </div>
        )}

        {uploadResult && (
          <div className="animate-slide-up" style={{
            marginTop: '2rem',
            background: 'var(--success-light)',
            border: '1px solid rgba(16, 185, 129, 0.2)',
            borderRadius: 'var(--radius-lg)',
            padding: '2rem',
            textAlign: 'center'
          }}>
            <div style={{ 
              width: '56px', 
              height: '56px', 
              background: 'var(--success)', 
              color: 'white', 
              borderRadius: '50%', 
              display: 'flex', 
              alignItems: 'center', 
              justifyContent: 'center',
              margin: '0 auto 1.5rem',
              boxShadow: '0 4px 12px rgba(16, 185, 129, 0.3)'
            }}>
              <CheckCircle size={32} />
            </div>
            <h3 style={{ fontSize: '1.5rem', fontWeight: '800', color: '#065f46', marginBottom: '0.5rem' }}>Success!</h3>
            <p style={{ color: '#065f46', marginBottom: '1.5rem', opacity: 0.8 }}>Your dataset is ready for the training phase.</p>
            
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
              gap: '1rem',
              background: 'rgba(255,255,255,0.5)',
              borderRadius: 'var(--radius-md)',
              padding: '1.25rem',
              border: '1px solid rgba(16, 185, 129, 0.1)'
            }}>
              <div style={{ textAlign: 'center' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: '700', textTransform: 'uppercase', color: 'var(--text-tertiary)', letterSpacing: '0.05em' }}>Rows</span>
                <p style={{ fontSize: '1.25rem', fontWeight: '800', color: 'var(--text-primary)', margin: '0.25rem 0 0' }}>{uploadResult.num_rows.toLocaleString()}</p>
              </div>
              <div style={{ textAlign: 'center' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: '700', textTransform: 'uppercase', color: 'var(--text-tertiary)', letterSpacing: '0.05em' }}>Columns</span>
                <p style={{ fontSize: '1.25rem', fontWeight: '800', color: 'var(--text-primary)', margin: '0.25rem 0 0' }}>{uploadResult.num_columns}</p>
              </div>
              <div style={{ textAlign: 'center' }}>
                <span style={{ fontSize: '0.75rem', fontWeight: '700', textTransform: 'uppercase', color: 'var(--text-tertiary)', letterSpacing: '0.05em' }}>Target</span>
                <p style={{ fontSize: '1.125rem', fontWeight: '800', color: 'var(--primary)', margin: '0.25rem 0 0' }}>{targetColumn}</p>
              </div>
            </div>
          </div>
        )}
      </div>

      <style dangerouslySetInnerHTML={{ __html: `
        @keyframes spin {
          to { transform: rotate(360deg); }
        }
      `}} />
    </div>
  )
}

export default UploadPage