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
    <div style={{
      maxWidth: '800px',
      margin: '0 auto',
      padding: '2rem',
      animation: 'slideUp 0.3s ease'
    }}>
      
      {/* Header */}
      <div style={{ marginBottom: '2rem', textAlign: 'center' }}>
        <h1 style={{
          fontSize: '32px',
          fontWeight: '700',
          marginBottom: '0.5rem',
          color: 'var(--color-text-primary)'
        }}>
          Upload Your Dataset
        </h1>
        <p style={{
          fontSize: '16px',
          color: 'var(--color-text-secondary)'
        }}>
          Upload a CSV or Excel file to start training with LLM-powered model selection
        </p>
      </div>

      {/* Upload Area */}
      <div
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
        style={{
          background: 'var(--color-bg-primary)',
          border: `2px dashed ${dragActive ? 'var(--color-primary)' : 'var(--color-border)'}`,
          borderRadius: 'var(--radius-lg)',
          padding: '3rem 2rem',
          textAlign: 'center',
          cursor: 'pointer',
          transition: 'all var(--transition-base)',
          marginBottom: '1.5rem',
          ...(dragActive && {
            background: 'var(--color-primary-light)',
            transform: 'scale(1.02)'
          })
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
          <>
            <Upload
              size={48}
              style={{
                color: dragActive ? 'var(--color-primary)' : 'var(--color-text-tertiary)',
                marginBottom: '1rem'
              }}
            />
            <p style={{
              fontSize: '18px',
              fontWeight: '500',
              color: 'var(--color-text-primary)',
              marginBottom: '0.5rem'
            }}>
              {dragActive ? 'Drop your file here' : 'Click to upload or drag and drop'}
            </p>
            <p style={{
              fontSize: '14px',
              color: 'var(--color-text-secondary)'
            }}>
              CSV or Excel files (max 100MB)
            </p>
          </>
        ) : (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '12px',
            animation: 'fadeIn 0.3s ease'
          }}>
            <FileText size={24} style={{ color: 'var(--color-primary)' }} />
            <div style={{ textAlign: 'left' }}>
              <p style={{
                fontSize: '16px',
                fontWeight: '500',
                color: 'var(--color-text-primary)'
              }}>
                {file.name}
              </p>
              <p style={{
                fontSize: '14px',
                color: 'var(--color-text-secondary)'
              }}>
                {(file.size / 1024 / 1024).toFixed(2)} MB
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Target Column Input */}
      {file && (
        <div style={{
          background: 'var(--color-bg-primary)',
          borderRadius: 'var(--radius-lg)',
          padding: '1.5rem',
          marginBottom: '1.5rem',
          animation: 'slideUp 0.3s ease'
        }}>
          <label style={{
            display: 'block',
            fontSize: '14px',
            fontWeight: '500',
            color: 'var(--color-text-primary)',
            marginBottom: '0.5rem'
          }}>
            Target Column Name
          </label>
          <input
            type="text"
            value={targetColumn}
            onChange={(e) => setTargetColumn(e.target.value)}
            placeholder="e.g., SalePrice, Price, Churn"
            style={{
              width: '100%',
              padding: '12px 16px',
              fontSize: '15px',
              border: '1px solid var(--color-border)',
              borderRadius: 'var(--radius-md)',
              outline: 'none',
              transition: 'border var(--transition-fast)',
            }}
            onFocus={(e) => e.target.style.borderColor = 'var(--color-primary)'}
            onBlur={(e) => e.target.style.borderColor = 'var(--color-border)'}
          />
          <p style={{
            fontSize: '13px',
            color: 'var(--color-text-secondary)',
            marginTop: '0.5rem'
          }}>
            The column you want to predict
          </p>
        </div>
      )}

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
          gap: '12px',
          animation: 'slideUp 0.3s ease'
        }}>
          <AlertCircle size={20} style={{ color: 'var(--color-danger)', flexShrink: 0 }} />
          <p style={{ fontSize: '14px', color: 'var(--color-danger)', margin: 0 }}>{error}</p>
        </div>
      )}

      {/* Success Message */}
      {uploadResult && (
        <div style={{
          background: 'var(--color-success-light)',
          border: '1px solid var(--color-success)',
          borderRadius: 'var(--radius-md)',
          padding: '1.5rem',
          marginBottom: '1.5rem',
          animation: 'slideUp 0.3s ease'
        }}>
          <div style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            marginBottom: '1rem'
          }}>
            <CheckCircle size={24} style={{ color: 'var(--color-success)' }} />
            <p style={{
              fontSize: '16px',
              fontWeight: '500',
              color: 'var(--color-success)',
              margin: 0
            }}>
              Dataset uploaded successfully!
            </p>
          </div>
          
          <div style={{
            background: 'white',
            borderRadius: 'var(--radius-md)',
            padding: '1rem',
            fontSize: '14px',
            color: 'var(--color-text-secondary)'
          }}>
            <p style={{ margin: '0 0 8px 0' }}><strong>Dataset ID:</strong> {uploadResult.dataset_id}</p>
            <p style={{ margin: '0 0 8px 0' }}><strong>Rows:</strong> {uploadResult.num_rows}</p>
            <p style={{ margin: 0 }}><strong>Columns:</strong> {uploadResult.num_columns}</p>
          </div>
        </div>
      )}

      {/* Upload Button */}
      {file && !uploadResult && (
        <button
          onClick={handleUpload}
          disabled={uploading}
          style={{
            width: '100%',
            padding: '14px 24px',
            fontSize: '16px',
            fontWeight: '500',
            color: 'white',
            background: uploading ? 'var(--color-text-tertiary)' : 'var(--color-primary)',
            border: 'none',
            borderRadius: 'var(--radius-md)',
            cursor: uploading ? 'not-allowed' : 'pointer',
            transition: 'all var(--transition-fast)',
            animation: 'slideUp 0.3s ease'
          }}
          onMouseEnter={(e) => {
            if (!uploading) e.target.style.background = 'var(--color-primary-hover)'
          }}
          onMouseLeave={(e) => {
            if (!uploading) e.target.style.background = 'var(--color-primary)'
          }}
        >
          {uploading ? 'Uploading...' : 'Upload Dataset'}
        </button>
      )}

    </div>
  )
}

export default UploadPage