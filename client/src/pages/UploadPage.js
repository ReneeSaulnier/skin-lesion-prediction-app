import React from 'react';
import DeleteForeverIcon from '@mui/icons-material/DeleteForever';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import '../styles/UploadPage.css';

function UploadPage() {

  return (
    <div className="upload-page">
      <div className="upload-card">
        <h1 className="upload-title">Upload Photo</h1>
        <div className='upload-dropbox'>
          <button className="upload-button">
            <CloudUploadIcon />
            <span>Upload file</span>
          </button>

          <div className="selected-file">
            <p>File Name Here</p>
            <button>
              <DeleteForeverIcon />
            </button>
          </div>

        </div>
      </div>
    </div>
  );
}

export default UploadPage;