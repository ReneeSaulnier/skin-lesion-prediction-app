import React from 'react';
import '../styles/LearnMore.css';

function LearnMore() {
    return (
        <div className="learn-more">
            <div className="learn-more-section">
                <h1 className="title">Hi there.</h1>
                <p>Please navigate to my <a href="https://github.com/ReneeSaulnier/skin-cancer-prediction-app" target="_blank" rel="noopener noreferrer">GitHub</a> for the source code and more details!</p>
            </div>  
        </div>
    );
    }

export default LearnMore;