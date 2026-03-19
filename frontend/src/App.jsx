import { useState } from 'react'
import UploadPage from './pages/UploadPage'

function App() {
  const [uploadedDataset, setUploadedDataset] = useState(null)

  const handleUploadSuccess = (result) => {
    setUploadedDataset(result)
    console.log('Dataset uploaded:', result)
    // Tomorrow we'll navigate to training page here!
  }

  return (
    <div style={{
      minHeight: '100vh',
      background: 'var(--color-bg-secondary)'
    }}>
      <UploadPage onUploadSuccess={handleUploadSuccess} />
    </div>
  )
}

export default App