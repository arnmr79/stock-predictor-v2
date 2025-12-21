import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from datetime import date, timedelta
from src.data_loader import fetch_data, add_indicators
from src.model import prepare_data, train_model, evaluate_model

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("Financial Stock Predictor")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
start_date = st.sidebar.date_input("Start Date", value=date.today() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", value=date.today())

if st.sidebar.button("Load Data"):
    try:
        with st.spinner("Fetching Data..."):
            df = fetch_data(ticker)
            df = add_indicators(df)
            
            # Filter by date
            df = df[(df.index.date >= start_date) & (df.index.date <= end_date)]
            
            st.session_state['data'] = df
            st.success("Data Loaded Successfully!")
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Main Content
if 'data' in st.session_state:
    df = st.session_state['data']
    
    # Visualization
    st.subheader(f"Price Chart: {ticker}")
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'])])
    
    # Add SMA trace
    if 'SMA_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20'))
        
    fig.update_layout(xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Prediction Section
    st.divider()
    st.subheader("Model Training & Prediction")
    
    if st.button("Train & Predict"):
        with st.spinner("Training Model..."):
            # Prepare data for training
            train_df = prepare_data(df)
            
            # Define Features
            features = ['Close', 'RSI', 'SMA_20']
            # Ensure features exist (handle potential NaNs from manual calc if needed)
            features = [f for f in features if f in train_df.columns]
            
            if not features:
                st.error("Not enough features generated.")
            else:
                X = train_df[features]
                y = train_df['Target']
                
                # Split (simple split for demo)
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
                y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
                
                model = train_model(X_train, y_train)
                
                # Evaluate
                acc = evaluate_model(model, X_test, y_test)
                st.metric("Model Accuracy (Test Set)", f"{acc:.2%}")
                
                # Predict Future
                # Get the last available data point (Today)
                last_row = df.iloc[[-1]] 
                current_features = last_row[features]
                
                prediction = model.predict(current_features)[0]
                prob = model.predict_proba(current_features)[0]
                
                res = "UP" if prediction == 1 else "DOWN"
                color = "green" if prediction == 1 else "red"
                
                st.markdown(f"### Prediction for Tomorrow: <span style='color:{color}'>{res}</span>", unsafe_allow_html=True)
                st.write(f"Confidence: {max(prob):.2f}")
