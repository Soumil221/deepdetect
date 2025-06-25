import { useState } from 'react'

import './App.css'


import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';

import Homeh from './home';
import Scanner from './scanner';





function App() {


  return (
    
    <Router>
    <Routes>
      <Route path="/" element={< Homeh/>}/>
      <Route path="/scanner" element={<Scanner/>} />
    </Routes>
  </Router>

      
  )
}

export default App

