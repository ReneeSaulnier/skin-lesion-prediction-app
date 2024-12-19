import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import UploadPage from './pages/UploadPage';
import LearnMore from './pages/LearnMore';

function App() {
  return (
    <Router>
      <div className="App">
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/UploadPage" element={<UploadPage />} />
          <Route path="/LearnMore" element={<LearnMore />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;