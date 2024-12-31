import React from 'react';
import { Link } from 'react-router-dom';
import '../styles/Navbar.css';

function Navbar() {
  return (
    <div className="navbar">
      <div className="company-name">
        <Link to="/" className="logo-text">Dermalytics</Link>
      </div>
      <div className="nav-links">
        <Link to="/UploadPage" className="nav-link">Get Started</Link>
        <Link to="/LearnMore" className="nav-link">Learn More</Link>
        <Link to="/Contact" className="nav-link">Contact</Link>
      </div>
    </div>
  );
}

export default Navbar;