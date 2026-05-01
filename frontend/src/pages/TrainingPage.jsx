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
    <div className="animate-fade-in" style={{
      maxWidth: '1200px',
      margin: '0 auto',
      padding: '2rem 1rem',
    }}>
      
      {/* Header */}
      <div style={{ marginBottom: '3rem' }}>
        <h1 style={{
          fontSize: '2.5rem',
          fontWeight: '900',
          marginBottom: '0.75rem',
          color: 'var(--text-primary)',
          letterSpacing: '-0.025em'
        }}>
          Training <span style={{ color: 'var(--primary)' }}>Dashboard</span>
        </h1>
        <p style={{
          fontSize: '1.125rem',
          color: 'var(--text-secondary)',
          maxWidth: '800px'
        }}>
          Harness the power of LLMs to automatically select and optimize the best machine learning architecture for your specific data.
        </p>
      </div>

      {/* Stats Grid */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
        gap: '1.5rem',
        marginBottom: '3rem'
      }}>
        <StatCard
          icon={<Database size={24} />}
          label="Total Experiments"
          value={stats.total}
          color="var(--primary)"
          trend="+12% from last week"
        />
        <StatCard
          icon={<Award size={24} />}
          label="Best Performance"
          value={jobs.find(j => j.status === 'completed')?.result?.best_score?.toFixed(3) || '0.000'}
          color="var(--success)"
          trend="Accuracy Score"
        />
        <StatCard
          icon={<Activity size={24} />}
          label="Active Jobs"
          value={stats.running}
          color="var(--primary)"
          trend="Currently processing"
        />
        <StatCard
          icon={<XCircle size={24} />}
          label="Failures"
          value={stats.failed}
          color="var(--danger)"
          trend="Check logs for info"
        />
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '2rem', alignItems: 'start' }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '2rem' }}>
          {/* Action Card */}
          {!activeJob && (
            <div className="card" style={{ padding: '2rem', background: 'var(--bg-primary)', border: '1px solid var(--primary-light)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.5rem' }}>
                <div>
                  <h3 style={{ fontSize: '1.25rem', fontWeight: '800', color: 'var(--text-primary)', margin: 0 }}>Start New Experiment</h3>
                  <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)', marginTop: '0.25rem' }}>Launch a new AutoML training cycle using LLM-augmented selection.</p>
                </div>
                <Zap size={32} style={{ color: 'var(--warning)' }} />
              </div>
              
              <button
                onClick={handleStartTraining}
                disabled={training || !datasetId}
                style={{
                  width: '100%',
                  padding: '1.25rem',
                  fontSize: '1rem',
                  fontWeight: '700',
                  color: 'white',
                  background: training || !datasetId ? 'var(--text-tertiary)' : 'var(--primary)',
                  border: 'none',
                  borderRadius: 'var(--radius-lg)',
                  cursor: training || !datasetId ? 'not-allowed' : 'pointer',
                  transition: 'all var(--transition-base)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  gap: '0.75rem',
                  boxShadow: '0 4px 12px rgba(59, 130, 246, 0.2)'
                }}
              >
                {training ? (
                  <>
                    <div style={{ width: '20px', height: '20px', border: '3px solid rgba(255,255,255,0.3)', borderTopColor: 'white', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }}></div>
                    Starting Pipeline...
                  </>
                ) : (
                  <>
                    <Zap size={20} />
                    Execute AutoML Pipeline
                  </>
                )}
              </button>
              
              {!datasetId && (
                <p style={{ color: 'var(--danger)', fontSize: '0.8125rem', marginTop: '1rem', textAlign: 'center', fontWeight: '600' }}>
                  ⚠️ Please upload a dataset first to enable training.
                </p>
              )}
            </div>
          )}

          {/* Active Training */}
          {activeJob && activeJob.status === 'running' && (
            <div className="card animate-pulse" style={{
              background: 'var(--bg-primary)',
              padding: '2.5rem',
              border: '2px solid var(--primary)',
              boxShadow: '0 0 20px rgba(59, 130, 246, 0.15)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
                <div>
                  <span style={{ 
                    background: 'var(--primary-light)', 
                    color: 'var(--primary)', 
                    padding: '0.25rem 0.75rem', 
                    borderRadius: '100px', 
                    fontSize: '0.75rem', 
                    fontWeight: '800',
                    textTransform: 'uppercase',
                    letterSpacing: '0.05em'
                  }}>
                    In Progress
                  </span>
                  <h3 style={{ fontSize: '1.5rem', fontWeight: '800', color: 'var(--text-primary)', marginTop: '0.75rem', marginBottom: '0.25rem' }}>Automating ML Lifecycle</h3>
                  <p style={{ fontSize: '0.875rem', color: 'var(--text-tertiary)' }}>Experiment ID: {activeJob.job_id}</p>
                </div>
                <div style={{ color: 'var(--primary)' }}>
                  <Activity size={48} />
                </div>
              </div>

              <div style={{ marginBottom: '1rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                  <span style={{ fontSize: '0.9375rem', fontWeight: '700', color: 'var(--text-secondary)' }}>{activeJob.current_step || 'Processing...'}</span>
                  <span style={{ fontSize: '1.125rem', fontWeight: '900', color: 'var(--primary)' }}>{activeJob.progress || 0}%</span>
                </div>
                <div style={{ background: 'var(--bg-secondary)', height: '12px', borderRadius: '100px', overflow: 'hidden' }}>
                  <div style={{
                    background: 'linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%)',
                    height: '100%',
                    width: `${activeJob.progress || 0}%`,
                    transition: 'width 0.5s cubic-bezier(0.4, 0, 0.2, 1)',
                    boxShadow: '0 0 10px rgba(59, 130, 246, 0.5)'
                  }} />
                </div>
              </div>
            </div>
          )}

          {/* Training History */}
          <div className="card" style={{ padding: '2rem', background: 'var(--bg-primary)' }}>
            <h3 style={{ fontSize: '1.25rem', fontWeight: '800', color: 'var(--text-primary)', marginBottom: '2rem', display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
              <Clock size={24} style={{ color: 'var(--primary)' }} />
              Recent Experiments
            </h3>

            {jobs.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '4rem 1rem', color: 'var(--text-tertiary)' }}>
                <Database size={64} style={{ marginBottom: '1.5rem', opacity: 0.1 }} />
                <p style={{ fontSize: '1.125rem', fontWeight: '600' }}>No history found</p>
                <p style={{ fontSize: '0.875rem' }}>Your training experiments will appear here.</p>
              </div>
            ) : (
              <div style={{ display: 'grid', gap: '1rem' }}>
                {jobs.map((job) => (
                  <JobCard key={job.job_id} job={job} getStatusColor={getStatusColor} getStatusIcon={getStatusIcon} />
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Sidebar Info */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <div className="card" style={{ padding: '1.5rem', background: 'linear-gradient(to bottom right, var(--primary), var(--accent))', color: 'white' }}>
            <Zap size={24} style={{ marginBottom: '1rem' }} />
            <h4 style={{ fontSize: '1.125rem', fontWeight: '800', marginBottom: '0.75rem' }}>How it works?</h4>
            <p style={{ fontSize: '0.875rem', opacity: 0.9, lineHeight: 1.6 }}>
              Our engine uses a Large Language Model (LLM) to analyze your dataset features, target distribution, and task complexity. It then dynamically constructs an optimal set of candidate models to train.
            </p>
          </div>
          
          {error && (
            <div className="card animate-slide-up" style={{ padding: '1.5rem', background: 'var(--danger-light)', border: '1px solid rgba(239, 68, 68, 0.1)' }}>
              <div style={{ display: 'flex', gap: '0.75rem' }}>
                <XCircle size={20} style={{ color: 'var(--danger)', flexShrink: 0 }} />
                <p style={{ fontSize: '0.875rem', color: 'var(--danger)', margin: 0, fontWeight: '500' }}>{error}</p>
              </div>
            </div>
          )}
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

const StatCard = ({ icon, label, value, color, trend }) => (
  <div className="card" style={{
    padding: '1.5rem',
    background: 'var(--bg-primary)',
    display: 'flex',
    flexDirection: 'column',
    gap: '1rem'
  }}>
    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
      <div style={{ 
        width: '48px', 
        height: '48px', 
        background: `${color}15`, 
        color: color, 
        borderRadius: 'var(--radius-md)', 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center' 
      }}>
        {icon}
      </div>
      <span style={{ fontSize: '0.75rem', fontWeight: '700', color: color === 'var(--danger)' ? 'var(--danger)' : 'var(--success)', background: color === 'var(--danger)' ? 'var(--danger-light)' : 'var(--success-light)', padding: '0.25rem 0.5rem', borderRadius: '4px' }}>
        {trend}
      </span>
    </div>
    <div>
      <span style={{ fontSize: '0.875rem', fontWeight: '600', color: 'var(--text-secondary)' }}>{label}</span>
      <div style={{ fontSize: '2rem', fontWeight: '900', color: 'var(--text-primary)', marginTop: '0.25rem', letterSpacing: '-0.02em' }}>{value}</div>
    </div>
  </div>
)

const JobCard = ({ job, getStatusColor, getStatusIcon }) => (
  <div style={{
    background: 'var(--bg-secondary)',
    borderRadius: 'var(--radius-lg)',
    padding: '1.25rem 1.5rem',
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    transition: 'all var(--transition-base)',
    border: '1px solid var(--border-color)',
  }}
  onMouseEnter={(e) => {
    e.currentTarget.style.borderColor = 'var(--primary)';
    e.currentTarget.style.transform = 'translateX(4px)';
  }}
  onMouseLeave={(e) => {
    e.currentTarget.style.borderColor = 'var(--border-color)';
    e.currentTarget.style.transform = 'translateX(0)';
  }}
  >
    <div style={{ display: 'flex', alignItems: 'center', gap: '1.25rem' }}>
      <div style={{ 
        width: '40px', 
        height: '40px', 
        borderRadius: '50%', 
        background: `${getStatusColor(job.status)}15`, 
        color: getStatusColor(job.status),
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center'
      }}>
        {getStatusIcon(job.status)}
      </div>
      <div>
        <div style={{ fontSize: '1rem', fontWeight: '700', color: 'var(--text-primary)' }}>{job.job_id}</div>
        <div style={{ fontSize: '0.8125rem', color: 'var(--text-tertiary)', marginTop: '0.25rem' }}>
          {job.result?.best_model ? (
            <span style={{ color: 'var(--success)', fontWeight: '600' }}>🏆 {job.result.best_model} ({job.result.best_score?.toFixed(4)})</span>
          ) : (
            'Awaiting results...'
          )}
        </div>
      </div>
    </div>
    
    <div style={{
      padding: '0.375rem 1rem',
      borderRadius: '100px',
      background: 'var(--bg-primary)',
      border: `1px solid ${getStatusColor(job.status)}30`,
      color: getStatusColor(job.status),
      fontSize: '0.75rem',
      fontWeight: '800',
      textTransform: 'uppercase',
      letterSpacing: '0.05em'
    }}>
      {job.status}
    </div>
  </div>
)

export default TrainingPage