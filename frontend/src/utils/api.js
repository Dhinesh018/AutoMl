import axios from 'axios'

const API_BASE_URL = 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Dataset API
export const uploadDataset = async (file, targetColumn) => {
  const formData = new FormData()
  formData.append('file', file)
  formData.append('target_column', targetColumn)
  
  const response = await api.post('/datasets/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  })
  
  return response.data
}

// Training API
export const startTraining = async (datasetId, targetColumn) => {
  const response = await api.post(`/train?dataset_id=${datasetId}&target_column=${targetColumn}`)
  return response.data
}

export const getTrainingStatus = async (jobId) => {
  const response = await api.get(`/train/status/${jobId}`)
  return response.data
}

export const listTrainingJobs = async () => {
  const response = await api.get('/train/jobs')
  return response.data
}

// Prediction API
export const getProductionFeatures = async () => {
  const response = await api.get('/models/production/features')
  return response.data
}

export const makePrediction = async (features) => {
  const response = await api.post('/predict', { features })
  return response.data
}

// Models API
export const listModelVersions = async () => {
  const response = await api.get('/models/versions')
  return response.data
}

export const promoteModel = async (version, stage = 'Production') => {
  const response = await api.post(`/models/promote/${version}?stage=${stage}`)
  return response.data
}

export const rollbackModel = async () => {
  const response = await api.post('/models/rollback')
  return response.data
}

export const compareModels = async (version1, version2) => {
  const response = await api.get(`/models/compare?version1=${version1}&version2=${version2}`)
  return response.data
}

// Health API
export const getHealth = async () => {
  const response = await api.get('/health')
  return response.data
}

export default api