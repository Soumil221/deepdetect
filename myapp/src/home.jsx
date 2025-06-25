import React from "react";
import { useNavigate } from "react-router-dom";
import "./home.css";

const Homeh = () => {
  const navigate=useNavigate();

  return (
    <div className="home-container">
      
      <nav className="navbar">
        <h1 className="logo">DEEPFAKE DETECTOR</h1>
        
        <div className="nav-links">
          <a href="#help">Help</a>
          <a href="#contact">Contact Us</a>
          <a href="#about">About Us</a>
        </div>
      </nav>
      
     
      <section className="hero">
        <div className="hero-text">
          <h1>Understanding the <span className="highlight">Deepfake</span> Problem</h1>
          <p>Deepfake technology is an AI-based method of creating synthetic media. It poses risks such as misinformation and identity fraud.</p>
          <button onClick={() => navigate("/scanner")} className="cta-button">
            Try the Scanner
          </button>
        </div>
        <div className="hero-image">
          <img 
            src="https://sosafe-awareness.com/sosafe-files/uploads/2022/08/Comparison_deepfake_blogpost_EN.jpg" 
            alt="Deepfake Example" 
            className="image-preview" 
          />
        </div>
      </section>

     
      <section id="about" className="about-section">
        
        <div className="section-content">
          <h2>What Are Deepfakes?</h2>
          <p>Deepfakes are AI-generated videos that replace someone's face with another person's likeness.</p>
          <img 
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQUSvez3zcbw9NeIYTVqtZCXJZh5gxYTY2uCw&s" 
            alt="What Are Deepfakes?" 
            className="image-preview" 
          />
        </div>
      </section>

      
      <section id="help" className="help-section">
        
        <div className="section-content">
          <h2>How to Detect Deepfakes?</h2>
          <p>Look for unnatural eye movements, mismatched audio, and inconsistent lighting.</p>
          <img 
            src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRwaMz4Vxi6rZ2HNU8W9OFlZMuhw-NqSM6flA&s" 
            alt="Detecting Deepfakes" 
            className="image-preview" 
          />
        </div>
      </section>
    </div>
  );
};

export default Homeh;
