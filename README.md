# AI Stock Predictor

An intelligent stock analysis and prediction application that leverages machine learning to forecast future stock prices for the top 50 S&P 500 companies. This tool provides a user-friendly graphical interface for training models, generating predictions, and visualizing historical stock data with AI-driven insights.

## About The Project

This project is designed for individuals interested in the stock market who want to explore the application of machine learning in financial forecasting. The application simplifies the complex process of data gathering, feature engineering, model training, and prediction into a seamless user experience. By focusing on the top 50 S&P 500 stocks, it offers a manageable yet powerful tool for analyzing market leaders.

The core of the application is a Random Forest Regressor, a robust machine learning model, which is trained on five years of historical data for each stock. The model uses technical indicators like moving averages and the Relative Strength Index (RSI) to inform its predictions. The intuitive GUI, built with CustomTkinter, allows users to easily train the models and then visualize the predictions on a dashboard and on detailed individual stock charts.

## Features

*   **Automated Ticker Discovery**: Automatically fetches the top 50 S&P 500 tickers by market capitalization.
*   **Individual Model Training**: Trains a unique Random Forest Regressor model for each stock to capture its specific historical patterns.
*   **Dynamic Prediction Dashboard**: Displays predictions for all stocks in an easy-to-read card format, showing current price, predicted price, and expected change.
*   **Detailed Stock Analysis**: Offers an in-depth view for each stock, including a historical candlestick chart with 50 and 200-day moving averages.
*   **AI Prediction Overlay**: Visualizes the AI's predicted price on the historical chart for easy comparison and analysis.
*   **Responsive User Interface**: Utilizes multithreading to perform long-running tasks in the background, ensuring the UI remains responsive.
*   **Company Information**: Displays company logos and full names for easy identification.

## Getting Started

To get a local copy up and running, follow these simple steps.

### Prerequisites

You will need to have Python 3 installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

### Installation

1.  **Clone the repo**
    ```sh
    git clone https://github.com/your_username/ai-stock-predictor.git
    cd ai-stock-predictor
    ```
2.  **Install Python packages**
    Create a virtual environment (optional but recommended):
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
    Install the required libraries:
    ```sh
    pip install -r requirements.txt
    ```
    You will need a `requirements.txt` file with the following content:
    ```
    customtkinter
    yfinance
    pandas
    scikit-learn
    joblib
    Pillow
    requests
    mplfinance
    matplotlib
    ```

## Usage

1.  **Run the application**
    ```sh
    python ai_studio_code.py
    ```
2.  **Train the Models**
    Upon launching the application, it will begin fetching the list of top 50 S&P 500 stocks. Once this is complete, the "Train Models" button will become active. Click it to begin the training process for all 50 stocks. This may take some time, and you can monitor the progress in the log window.

3.  **Predict Stock Prices**
    After the models have been trained, click the "Predict All Stocks" button. The application will then fetch the latest market data and generate predictions for each stock, displaying the results on the main dashboard.

4.  **Analyze Individual Stocks**
    Click on any stock card in the dashboard to open a detailed view. This view will show a historical price chart for the selected stock, along with the AI's predicted price for the next day.

## Technologies Used

*   **Python**: The core programming language.
*   **CustomTkinter**: For creating the modern and responsive graphical user interface.
*   **yfinance**: To download historical market data from Yahoo Finance.
*   **pandas**: For data manipulation and analysis.
*   **scikit-learn**: For implementing the Random Forest Regressor machine learning model.
*   **joblib**: For saving and loading the trained machine learning models.
*   **Pillow (PIL)**: For image processing, particularly for handling company logos.
*   **requests**: To fetch company profile information and logos.
*   **Matplotlib & mplfinance**: For creating detailed and aesthetically pleasing financial charts.
