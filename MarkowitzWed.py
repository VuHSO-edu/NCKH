import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from vnstock import stock_historical_data
from gurobipy import Model, GRB, quicksum
import concurrent.futures

# Mapping of Vietnam stock tickers
ticker_mapping = {
    'VIC': 'VIC',
    'VNM': 'VNM',
    'MSN': 'MSN',
    'HPG': 'HPG',
    'BID': 'BID',
    'VPB': 'VPB',
    'VCB': 'VCB',
}


# Fetch stock data using vnstock
@st.cache_data
def get_data(tickers, start_date, end_date):
    def fetch_data(ticker):
        try:
            # Use vnstock to get stock data
            df = stock_historical_data(ticker=ticker, start_date=start_date.strftime('%Y-%m-%d'),
                                       end_date=end_date.strftime('%Y-%m-%d'))
            data_stock = df[['close']]
            return data_stock
        except Exception as e:
            st.warning(f"Không tìm thấy dữ liệu cho mã cổ phiếu {ticker}: {e}")
            return None

    data = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {ticker: executor.submit(fetch_data, ticker) for ticker in tickers}
        for ticker, future in futures.items():
            result = future.result()
            if result is not None:
                data[ticker] = result
            else:
                st.warning(f"Dữ liệu cho mã {ticker} không khả dụng.")

    # Return data as a DataFrame
    return pd.DataFrame(data)


# Calculate covariance matrix and expected returns
@st.cache_data
def calculate_covariance_matrix(data):
    returns = data.pct_change().dropna()
    return returns.cov(), returns.mean()


# Portfolio optimization using Markowitz model
def optimize_portfolio(covariance_matrix, mean_returns, target_return, tickers):
    model = Model("Portfolio Optimization")

    available_tickers = list(covariance_matrix.index)
    valid_tickers = [ticker for ticker in tickers if ticker in available_tickers]

    if not valid_tickers:
        st.warning("Không có dữ liệu đủ để tối ưu hóa danh mục.")
        return None

    n = len(valid_tickers)
    weights = model.addVars(valid_tickers, lb=0, ub=1, name="weights")

    portfolio_variance = quicksum(weights[i] * covariance_matrix.loc[i, j] * weights[j]
                                  for i in valid_tickers for j in valid_tickers)
    model.setObjective(portfolio_variance, GRB.MINIMIZE)

    model.addConstr(quicksum(weights[i] for i in valid_tickers) == 1, "SumWeights")
    model.addConstr(quicksum(weights[i] * mean_returns[i] for i in valid_tickers) >= target_return / 100,
                    "TargetReturn")

    model.optimize()

    if model.status == GRB.OPTIMAL:
        return {ticker: weights[ticker].X for ticker in valid_tickers}
    else:
        st.warning("Không tìm thấy giải pháp tối ưu.")
        return None


# Main Streamlit app function
def main():
    st.title("Phân tích danh mục đầu tư chứng khoán Việt Nam")

    tickers = st.multiselect("Chọn mã cổ phiếu", list(ticker_mapping.keys()))
    start_date = st.date_input("Ngày bắt đầu", pd.to_datetime('today') - pd.DateOffset(years=5))
    end_date = st.date_input("Ngày kết thúc", pd.to_datetime('today'))
    target_return = st.slider("Mục tiêu lợi nhuận (%)", 0, 20, 10)

    if len(tickers) > 0:
        data = get_data(tickers, start_date, end_date)

        if not data.empty:
            covariance_matrix, mean_returns = calculate_covariance_matrix(data)
            portfolio = optimize_portfolio(covariance_matrix, mean_returns, target_return, tickers)

            if portfolio:
                st.subheader("Kết quả phân tích danh mục")
                st.write(f"Danh mục tối ưu với mục tiêu lợi nhuận {target_return}%:")
                for ticker, weight in portfolio.items():
                    st.write(f"{ticker}: {weight * 100:.2f}%")
        else:
            st.warning("Không có dữ liệu cổ phiếu để phân tích.")
    else:
        st.warning("Vui lòng chọn mã cổ phiếu!")


if __name__ == "__main__":
    main()
