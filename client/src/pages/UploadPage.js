import React from 'react';
import '../styles/UploadPage.css';

function UploadPage() {
  return (
      <div className="upload-page">
        <div className="hero-section">
          <h1 className="title">Upload a photo of your skin lesion.</h1>
            <p className="description">
                Snap a photo and let AI guide you. Whether it’s just a harmless freckle or an early sign of something serious, checking now can make all the difference.
            </p>  
            <div className="button-group">
                <button className="upload-button">Upload Photo</button>
            </div>
        </div>  
      </div>
  );
}

export default UploadPage;