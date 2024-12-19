import React from 'react';
import { useNavigate } from "react-router-dom";
import '../styles/LandingPage.css';

function LandingPage() {
  const navigate = useNavigate();

  const handleGetStarted = () => {
    navigate("/UploadPage");
  };

  const handleLearnMore = () => {
    navigate("/LearnMore");
  };

  return (
    <div className="landing-page">
      <div className="hero-section">
        <h1 className="title">Skin Lesion Analysis.</h1>
        <p className="description">
          Snap a photo and let AI guide you. Whether it’s just a harmless freckle or an early sign of something serious, checking now can make all the difference.
        </p>
        <div className="button-group">
          <button className="start-button" onClick={handleGetStarted}>Get Started</button>
          <button className="learn-button" onClick={handleLearnMore}>Learn More</button>
        </div>
      </div>
    </div>
  );
}

export default LandingPage;