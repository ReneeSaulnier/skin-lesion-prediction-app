import React, { useState } from 'react';
import DeleteForeverIcon from '@mui/icons-material/DeleteForever';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import { useNavigate } from "react-router-dom";
import '../styles/UploadPage.css';

function UploadPage() {

  const [fileName, setFileName] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleButtonClick = () => {
    document.getElementById('file-upload').click();
  };

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setFileName(file.name);
    }
  };

  const handleDelete = () => {
    setFileName('');
  };

  const handleSubmit = () => {
    if (fileName) {
      navigate("/ResultsPage");
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
            <CloudUploadIcon />
            <span>Upload file</span>
          </button>
          {fileName && (
            <div className="selected-file">
              <p>{fileName}</p>
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
