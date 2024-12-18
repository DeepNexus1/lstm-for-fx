# Deep Nexus One (DN1) Research Repository for FX Prediction

## Project Overview

This repository contains a foreign exchange (FX) trading prediction system developed in 2019. The project implements machine learning prediction models using an LSTM architecture with a regression output. Multiple models run in parallel and send signals to MT4.

## Technical Architecture

### Core Components
- **Prediction Models**: 
  - Implemented using TensorFlow and Keras
  - Long Short-Term Memory (LSTM) neural network architecture
  - Developed for 6 different currency pairs
  - Prediction frequency: Every 5 minutes

- **Backend Framework**:
  - Flask web framework for serving models
  - Waitress WSGI server for deployment
  - Oanda Rest API integration for market data and trading execution
  
## Technical Complexity
The project represented a significant technical challenge in:
- Coordinating multiple LSTM neural network models
- Integrating diverse technologies (TensorFlow, Keras, Flask, Waitress)
- Serving multiple prediction models simultaneously
- Executing trades across multiple currency pairs from a platform originally designed for single-pair trading

### Key Technical Challenges Solved

1. **Multi-Model Management**
   - Developed a custom .dll file to manage multiple prediction models and scripts
   - Implemented a timer function to orchestrate script execution across different currency pairs

2. **Platform Integration**
   - Bridged prediction models with MetaTrader 4 (MT4) trading platform
   - Resolved complex compatibility issues between different programming environments
   - Managed communication protocols (largely involving TCP/IP networking)
   - Implemented asynchronous processing to overcome limitations of MT4

## Technology Stack
- Python
- TensorFlow
- Keras
- Flask
- Waitress
- Oanda Rest API
- MetaTrader 4 (mql4)

## Deployment Considerations
- Custom .dll file for script and model management
- Periodic execution of prediction models
- Seamless integration between machine learning predictions and trading platform

## Limitations and Considerations
- Research project from 2019
- Developed for specific trading strategy and currency pairs
- Requires careful review and potential updates for current market conditions
- MT4 trade logic is built from a genetic algorithm and is sample logic only
- Python 3.5 was used to build this project
- Final production models are proprietary

## Disclaimer
This is a research project and should not be considered financial advice. Trading involves significant financial risk.

## License
http://www.apache.org/licenses/LICENSE-2.0

## Contact
web@deepnexus.com

_Repository initialized in 2024, based on research conducted in 2019._
