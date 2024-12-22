import React from 'react';
import { predictImage } from '../api/index';
import { useLocation } from 'react-router-dom';

function ResultPage() {
  const location = useLocation();
  const { file } = location.state || {}; // Get the file passed via state

  const handleProcessFile = () => {
    if (file) {
        // Call my api
        predictImage(file);
      console.log('Processing file:', file);
    }
  };

  return (
    <div>
      <h1>Result Page</h1>
      {file ? (
        <div>
          <p>Uploaded File: {file.name}</p>
          <button onClick={handleProcessFile}>Process File</button>
        </div>
      ) : (
        <p>No file was uploaded!</p>
      )}
    </div>
  );
}

export default ResultPage;