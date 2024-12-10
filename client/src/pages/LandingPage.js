import React from 'react';
import '../styles/LandingPage.css';

function LandingPage() {
  return (
    <div className="landing-page">
      <div className="hero-section">
        <h1 className="title">Technology Meets Dermatology.</h1>
        <p className="description">
          Harness the power of artificial intelligence to detect skin cancer early. 
          Upload your images and get instant analysis.
        </p>
        <div className="button-group">
          <button className="start-button">Get Started</button>
          <button className="learn-button">Learn More</button>
        </div>
      </div>
    </div>
  );
}

export default LandingPage;