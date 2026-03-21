import React, { useState, useEffect } from 'react'
import { 
  Activity, 
  Clock, 
  CheckCircle, 
  XCircle, 
  TrendingUp,
  Zap,
  Database,
  Award
} from 'lucide-react'
import { 
  startTraining, 
  getTrainingStatus, 
  listTrainingJobs 
} from '../utils/api'

const TrainingPage = ({ datasetId, targetColumn }) => {
  const [jobs, setJobs] = useState([])
  const [activeJob, setActiveJob] = useState(null)
  const [training, setTraining] = useState(false)
  const [error, setError] = useState(null)

  // Fetch all training jobs on mount
  useEffect(() => {
    fetchJobs()
  }, [])

  // Poll active job status
  useEffect(() => {
    if (activeJob && activeJob.status === 'running') {
      const interval = setInterval(() => {
        pollJobStatus(activeJob.job_id)
      }, 3000) // Poll every 3 seconds

      return () => clearInterval(interval)
    }
  }, [activeJob])

  const fetchJobs = async () => {
    try {
      const data = await listTrainingJobs()
      setJobs(data.jobs || [])
      
      // Find running job
      const running = data.jobs?.find(j => j.status === 'running')
      if (running) {
        setActiveJob(running)
      }
    } catch (err) {
      console.error('Failed to fetch jobs:', err)
    }
  }

  const pollJobStatus = async (jobId) => {
    try {
      const status = await getTrainingStatus(jobId)
      setActiveJob(status)
      
      // Update jobs list
      setJobs(prev => prev.map(j => 
        j.job_id === jobId ? status : j
      ))
      
      // If completed, stop polling
      if (status.status === 'completed' || status.status === 'failed') {
        setActiveJob(null)
        fetchJobs()
      }
    } catch (err) {
      console.error('Failed to poll status:', err)
    }
  }

  const handleStartTraining = async () => {
    if (!datasetId || !targetColumn) {
      setError('No dataset uploaded. Please upload a dataset first.')
      return
    }

    setTraining(true)
    setError(null)

    try {
      const result = await startTraining(datasetId, targetColumn)
      setActiveJob({ ...result, progress: 0, status: 'running' })
      fetchJobs()
    } catch (err) {
      setError(err.response?.data?.detail || 'Failed to start training')
    } finally {
      setTraining(false)
    }
  }

  // Calculate stats
  const stats = {
    total: jobs.length,
    completed: jobs.filter(j => j.status === 'completed').length,
    running: jobs.filter(j => j.status === 'running').length,
    failed: jobs.filter(j => j.status === 'failed').length,
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'completed':
        return 'var(--color-success)'
      case 'running':
        return 'var(--color-primary)'
      case 'failed':
        return 'var(--color-danger)'
      default:
        return 'var(--color-text-secondary)'
    }
  }

  const getStatusIcon = (status) => {
    switch (status) {
      case 'completed':
        return <CheckCircle size={16} />
      case 'running':
        return <Activity size={16} />
      case 'failed':
        return <XCircle size={16} />
      default:
        return <Clock size={16} />
    }
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
          Training Dashboard
        </h1>
        <p style={{
          fontSize: '16px',
          color: 'var(--color-text-secondary)'
        }}>
          Monitor your AutoML training jobs powered by LLM model selection
        </p>
      </div>

      {/* Stats Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
        gap: '1rem',
        marginBottom: '2rem'
      }}>
        
        <StatCard
          icon={<Database size={24} />}
          label="Total Jobs"
          value={stats.total}
          color="var(--color-primary)"
        />
        
        <StatCard
          icon={<CheckCircle size={24} />}
          label="Completed"
          value={stats.completed}
          color="var(--color-success)"
        />
        
        <StatCard
          icon={<Activity size={24} />}
          label="Running"
          value={stats.running}
          color="var(--color-primary)"
        />
        
        <StatCard
          icon={<XCircle size={24} />}
          label="Failed"
          value={stats.failed}
          color="var(--color-danger)"
        />
        
      </div>

      {/* Start Training Button */}
      {!activeJob && (
        <button
          onClick={handleStartTraining}
          disabled={training || !datasetId}
          style={{
            width: '100%',
            padding: '16px 24px',
            fontSize: '16px',
            fontWeight: '500',
            color: 'white',
            background: training || !datasetId 
              ? 'var(--color-text-tertiary)' 
              : 'var(--color-primary)',
            border: 'none',
            borderRadius: 'var(--radius-md)',
            cursor: training || !datasetId ? 'not-allowed' : 'pointer',
            transition: 'all var(--transition-fast)',
            marginBottom: '2rem',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '8px'
          }}
          onMouseEnter={(e) => {
            if (!training && datasetId) {
              e.target.style.background = 'var(--color-primary-hover)'
            }
          }}
          onMouseLeave={(e) => {
            if (!training && datasetId) {
              e.target.style.background = 'var(--color-primary)'
            }
          }}
        >
          <Zap size={20} />
          {training ? 'Starting Training...' : 'Start New Training'}
        </button>
      )}

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
          <XCircle size={20} style={{ color: 'var(--color-danger)' }} />
          <p style={{ fontSize: '14px', color: 'var(--color-danger)' }}>{error}</p>
        </div>
      )}

      {/* Active Training */}
      {activeJob && activeJob.status === 'running' && (
        <div style={{
          background: 'var(--color-bg-primary)',
          borderRadius: 'var(--radius-lg)',
          padding: '1.5rem',
          marginBottom: '2rem',
          border: '2px solid var(--color-primary)',
          animation: 'slideUp 0.3s ease'
        }}>
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            marginBottom: '1rem'
          }}>
            <div>
              <h3 style={{
                fontSize: '18px',
                fontWeight: '600',
                marginBottom: '4px',
                color: 'var(--color-text-primary)'
              }}>
                Training in Progress
              </h3>
              <p style={{
                fontSize: '14px',
                color: 'var(--color-text-secondary)'
              }}>
                Job ID: {activeJob.job_id}
              </p>
            </div>
            <div style={{
              background: 'var(--color-primary-light)',
              color: 'var(--color-primary)',
              padding: '6px 16px',
              borderRadius: 'var(--radius-md)',
              fontSize: '13px',
              fontWeight: '500',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}>
              <Activity size={14} className="animate-pulse" />
              Running
            </div>
          </div>

          {/* Progress Bar */}
          <div style={{
            background: 'var(--color-bg-secondary)',
            height: '12px',
            borderRadius: '6px',
            overflow: 'hidden',
            marginBottom: '12px'
          }}>
            <div style={{
              background: 'var(--color-primary)',
              height: '100%',
              width: `${activeJob.progress || 0}%`,
              transition: 'width 0.5s ease',
              borderRadius: '6px'
            }} />
          </div>

          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            fontSize: '14px',
            color: 'var(--color-text-secondary)'
          }}>
            <span>{activeJob.current_step || 'Initializing...'}</span>
            <span style={{ fontWeight: '500' }}>
              {activeJob.progress || 0}%
            </span>
          </div>
        </div>
      )}

      {/* Training Jobs List */}
      <div style={{
        background: 'var(--color-bg-primary)',
        borderRadius: 'var(--radius-lg)',
        padding: '1.5rem',
        border: '1px solid var(--color-border)'
      }}>
        <h3 style={{
          fontSize: '18px',
          fontWeight: '600',
          marginBottom: '1.5rem',
          color: 'var(--color-text-primary)'
        }}>
          Training History
        </h3>

        {jobs.length === 0 ? (
          <div style={{
            textAlign: 'center',
            padding: '3rem 1rem',
            color: 'var(--color-text-secondary)'
          }}>
            <Database size={48} style={{ marginBottom: '1rem', opacity: 0.3 }} />
            <p style={{ fontSize: '16px' }}>No training jobs yet</p>
            <p style={{ fontSize: '14px', marginTop: '0.5rem' }}>
              Start your first training to see it here
            </p>
          </div>
        ) : (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
            {jobs.map((job) => (
              <JobCard key={job.job_id} job={job} getStatusColor={getStatusColor} getStatusIcon={getStatusIcon} />
            ))}
          </div>
        )}
      </div>

    </div>
  )
}

// Stat Card Component
const StatCard = ({ icon, label, value, color }) => (
  <div style={{
    background: 'var(--color-bg-primary)',
    borderRadius: 'var(--radius-lg)',
    padding: '1.5rem',
    border: '1px solid var(--color-border)',
    transition: 'all var(--transition-base)'
  }}>
    <div style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      marginBottom: '0.5rem'
    }}>
      <span style={{
        fontSize: '14px',
        color: 'var(--color-text-secondary)',
        fontWeight: '500'
      }}>
        {label}
      </span>
      <div style={{ color }}>{icon}</div>
    </div>
    <div style={{
      fontSize: '32px',
      fontWeight: '700',
      color
    }}>
      {value}
    </div>
  </div>
)

// Job Card Component
const JobCard = ({ job, getStatusColor, getStatusIcon }) => (
  <div style={{
    background: 'var(--color-bg-secondary)',
    borderRadius: 'var(--radius-md)',
    padding: '1rem 1.25rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    transition: 'all var(--transition-fast)',
    border: '1px solid transparent'
  }}
  onMouseEnter={(e) => {
    e.currentTarget.style.borderColor = 'var(--color-border)'
    e.currentTarget.style.background = 'var(--color-bg-primary)'
  }}
  onMouseLeave={(e) => {
    e.currentTarget.style.borderColor = 'transparent'
    e.currentTarget.style.background = 'var(--color-bg-secondary)'
  }}
  >
    <div style={{ flex: 1 }}>
      <div style={{
        fontSize: '15px',
        fontWeight: '500',
        color: 'var(--color-text-primary)',
        marginBottom: '4px'
      }}>
        {job.job_id}
      </div>
      <div style={{
        fontSize: '13px',
        color: 'var(--color-text-secondary)'
      }}>
        {job.result?.best_model && `Best: ${job.result.best_model} (R² ${job.result.best_score?.toFixed(3)})`}
      </div>
    </div>
    
    <div style={{
      display: 'flex',
      alignItems: 'center',
      gap: '6px',
      padding: '4px 12px',
      borderRadius: 'var(--radius-md)',
      background: `${getStatusColor(job.status)}15`,
      color: getStatusColor(job.status),
      fontSize: '13px',
      fontWeight: '500'
    }}>
      {getStatusIcon(job.status)}
      {job.status}
    </div>
  </div>
)

export default TrainingPage