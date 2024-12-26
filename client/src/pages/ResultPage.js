import React from 'react';
import { useLocation } from 'react-router-dom';
import '../styles/ResultPage.css';

function ResultPage() {
  const location = useLocation();
  const { result } = location.state || {};

  if (!result) {
    return (
      <div className="result-page">
        <div className="result-card">
          <h1 className="result-title">No result available.</h1>
          <button className="btn-go-back" onClick={handleGoBack}>
            Go Back
          </button>
        </div>
      </div>
    );
  }

  const { predicted_class, confidence } = result;

  return (
    <div className="result-page">
      <div className="result-card">
        <h1 className="result-title">Prediction Result</h1>
        
        <div className="prediction-info">
          <p>
            <span className="label">Predicted Class:</span> {predicted_class}
          </p>
          <p>
            <span className="label">Confidence:</span> {(confidence * 100).toFixed(2)}%
          </p>
        </div>
      </div>
    </div>
  );
}

export default ResultPage;