import React from 'react';
import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';
import LandingPage from './pages/LandingPage';
import UploadPage from './pages/UploadPage';
import ResultPage from './pages/ResultPage';
import LearnMore from './pages/LearnMore';
import Navbar from './components/Navbar';
import Footer from './components/Footer';

function App() {
  const location = useLocation();

  // Conditionally hide navbar and footer on specific pages
  const hideNavbar = location.pathname === '/';
  const hideFooter = location.pathname === '/';

  return (
    <div className="App">
      {!hideNavbar && <Navbar />}
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/UploadPage" element={<UploadPage />} />
        <Route path="/ResultPage" element={<ResultPage />} />
        <Route path="/LearnMore" element={<LearnMore />} />
      </Routes>
      {!hideFooter && <Footer />}
    </div>
  );
}

export default function AppWrapper() {
  return (
    <Router>
      <App />
    </Router>
  );
}