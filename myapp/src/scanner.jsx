import React, { useState } from "react";
import "./scanner.css";


import axios from "axios";
// import { FaGithub, FaTwitter, FaFacebookF, FaInstagram, FaLinkedin, FaYoutube } from "react-icons/fa";

const Scanner = () => {
  const [video, setVideo] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState("");
  const [accur, setAccur] = useState("");

  const handleFileChange = (event) => {
    setVideo(event.target.files[0]);
    setResult(null);
  };

  const handleUpload = async () => {
    setLoading(true);
    if (!video) {
      setResult("Please select a file first.");
      return;
    }
  
    const formData = new FormData();
    formData.append("file", video);
  
    try {
      const response = await axios.post('http://localhost:5000/predict', formData,{
        headers: {
          
        },
      }
        
      );
      setLoading(false);
      console.log(response.data.prediction);
      setAccur(response.data.confidence);
      setResult(response.data.prediction);
      console.log(result);}
     
    catch (error) {
      setResult(`Error uploading file: ${error.response?.data || error.message}`);
    }
  };
  return (
    <div className="container">
      
      <nav className="navbar">
        <div className="logo">DEEPFAKE DETECTOR</div>
        
        <div className="nav-links">
          <a href="#help">Help</a>
          <a href="#contact">Contact Us</a>
          <a href="#about">About Us</a>
        </div>
        
      </nav>

     
      <div className="scanner-section">
        <h1>Scan & Detect Deepfake Videos</h1>
        <p>Place a video link or upload a video</p>

        <div className="input-container">
          <input type="file" accept="video/*" onChange={handleFileChange} className="file-input" />
        </div>

        {/* {video && <video src={URL.createObjectURL(video)} controls className="video-preview" />} */}

       

        <button className="scan-button" onClick={handleUpload} disabled={loading}>
          {loading ? "Analyzing..." : "SCAN"}
        </button>

        {result && <p className={`result ${result === "Fake" ? "fake" : "real"}`}>Result: {result}</p>}
        {accur && <p className={`result ${result === "Fake" ? "fake" : "real"}`}>Confidence: {accur+"%"}</p>}
      </div>

     
      <footer className="footer">
        <div className="footer-logo">Deepfake Detection</div>

        <div className="footer-links">
          <div>
            
            <ul>
              <li>About Us</li>
              <li>Contact Us</li>
              <li>FAQ</li>
            </ul>
          </div>

         
        </div>

       
        <div className="social-icons">
          
        </div>

        
      </footer>
    </div>
  );
};

export default Scanner;
