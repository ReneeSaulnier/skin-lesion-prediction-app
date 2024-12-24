import React, { useState } from 'react';
import DeleteForeverIcon from '@mui/icons-material/DeleteForever';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { useNavigate } from "react-router-dom";
import axios from 'axios';
import '../styles/UploadPage.css';

function UploadPage() {

  const [file, setFile] = useState('');
  const [error, setError] = useState(''); 
  const [result, setResult] = useState('');
  const uploadUrl = 'http://localhost:8080/api/upload';
  const predictUrl = 'http://localhost:8080/api/predict';
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
        uploadFile(file);
      } catch (error) {
        console.error(error);
        setError('An error occurred while processing the file');
      }
    } else {
      setError('Please select a file to upload!');
    }
  }


  const uploadFile = async (file) => {  
    try {
      const formData = new FormData();
      formData.append('file', file);

      // Send the FormData
      const response = await axios.post(uploadUrl, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
          'accept': '*/*',
        },
      });
  
      console.log('File uploaded successfully:', response.data);
    } catch (error) {
      console.error('Error uploading file:', error.response || error.message);
    }
  };

  return (
    <div className="upload-page">
      <h1 className="upload-title">Get instant results.</h1>
      <div className="upload-card">
        <div className='upload-dropbox'>
          <input
            type="file"
            id="file-upload"
            className="input-file"
            onChange={handleFileChange}
          />
          <button className="upload-button" onClick={handleButtonClick}>
            <CloudUploadIcon />
            <span>Upload file</span>
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

        {result && (
          <div className="result">
            <h2>Result</h2>
            <p>{result}</p>
          </div>
        )}
      </div>
    </div>
  );
}


export default UploadPage;
