# Stock Price Prediction using LSTM

This project demonstrates how to predict stock prices using Long Short-Term Memory (LSTM) networks. It includes a Streamlit app to visualize the predictions and other relevant data.

## Prerequisites

Before you start, ensure that you have the following tools installed:

- **Python 3.12 or higher**
- **Git** (for version control)
- **Streamlit** (for app deployment)

## Setting Up the Project in a Codespace

Follow these steps to set up the project and run it in a Codespace:

### 1. **Create a Codespace**

   - Navigate to your GitHub repository page.
   - Click the **Code** button and select **Open with Codespaces**.
   - Click **Create codespace on main**.

### 2. **Set Up a Virtual Environment**

   Once the Codespace is ready, open the terminal and execute the following commands:

   ```bash
   # Create a virtual environment
   a.python3 -m venv venv

   # Activate the virtual environment
   b.source venv/bin/activate  # For Linux/MacOS
   # or
   venv\Scripts\activate  # For Windows
   Upgrade pip
   It's always a good idea to make sure pip is up to date:
   c.pip install --upgrade pip
   Install Required Dependencies
Install the necessary libraries for this project:
d.pip install tensorflow==2.19.0 numpy==1.26.3 yfinance
  pip install streamlit
  pip install matplotlib
  pip install scikit-learn

To install Streamlit, use the following command:
pip install streamlit
This will install Streamlit in your current Python environment. Once the installation is complete, you can run your Streamlit app using:
streamlit run app.py

