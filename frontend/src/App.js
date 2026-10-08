import { useEffect, useState } from 'react';
import axios from 'axios';

function App() {
  const [status, setStatus] = useState('');
  useEffect(() => {
    axios.get('http://localhost:8000/health').then(res => setStatus(res.data.status));
  }, []);
  return <div>Backend status: {status}</div>;
}

export default App;