# 📈 AI Stock Market Analyzer

A full-stack machine learning application that predicts daily stock price movements using technical analysis and Random Forest classification.

## 🚀 Features
* **Real-time Data:** Fetches live historical data using `yfinance`.
* **Technical Analysis:** Automatically calculates RSI, MACD, and SMA indicators.
* **Machine Learning:** Uses a Random Forest Classifier to predict if a stock will go **UP** or **DOWN** the next day.
* **Interactive UI:** Built with Streamlit and Plotly for zoomable, interactive financial charts.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Frontend:** Streamlit, Plotly
* **Backend:** Scikit-Learn, Pandas-TA, Yfinance
* **Testing:** Pytest

## 💻 How to Run Locally

1.  **Clone the repository**
    ```bash
    git clone https://github.com/YOUR_USERNAME/stock-market-analyzer.git
    cd stock-market-analyzer
    ```

2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the app**
    ```bash
    streamlit run app.py
    ```

## 🧪 Running Tests
To verify the data fetching and model logic:
```bash
pytest tests/
