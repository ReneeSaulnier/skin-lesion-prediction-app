import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import UploadPage from './pages/UploadPage';
import ResultPage from './pages/ResultPage';
import LearnMore from './pages/LearnMore';
import Navbar from './components/Navbar';

function App() {
  return (
    <Router>
      <div classNames="App">
        <Navbar />
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path="/UploadPage" element={<UploadPage />} />
          <Route path="/ResultPage" element={<ResultPage />} />
          <Route path="/LearnMore" element={<LearnMore />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;