import React, { useState } from 'react';
import DeleteForeverIcon from '@mui/icons-material/DeleteForever';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { useNavigate } from "react-router-dom";
import UploadApi from '../api/upload';
import PredictApi from '../api/predict';
import '../styles/UploadPage.css';

function UploadPage() {

  const [file, setFile] = useState('');
  const [error, setError] = useState(''); 
  const navigate = useNavigate();

  const handleButtonClick = () => {
    document.getElementById('file-upload').click();
  };

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
    }
  };

  const handleDelete = () => {
    setFile(null);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (file) {
      try {
        // Call the upload API
        await UploadApi(file);

        // Call the predict API
        const predictResponse = await PredictApi(file);
        
        // Navigate and pass the result to the ResultPage
        if (predictResponse) {
          // Parse the nested json string
          const parsedPrediction = JSON.parse(predictResponse.prediction);
          navigate('/ResultPage', { state: { result: parsedPrediction } });
        } else {
          setError('No prediction response received.');
        }
      } catch (error) {
        console.error(error);
        setError('An error occurred while processing the file');
      }
    } else {
      setError('Please select a file to upload!');
    }
  }

  return (
    <div className="upload-page">
      <div className="upload-card">
        <div className='upload-dropbox'>
          <input
            type="file"
            id="file-upload"
            className="input-file"
            onChange={handleFileChange}
          />
          <button className="upload-button" onClick={handleButtonClick}>
            <CloudUploadIcon style={{ fontSize: 40 }} />
          </button>
          {file && (
            <div className="selected-file">
              <p>{file.name}</p>
              <button onClick={handleDelete}>
                <DeleteForeverIcon />
              </button>
            </div>
          )}
          <button className='submit-button' onClick={handleSubmit}>
            Submit
          </button>
          {error && (
            <p className="error-message">{error}</p>
          )}
        </div>
      </div>
    </div>
  );
}


export default UploadPage;
