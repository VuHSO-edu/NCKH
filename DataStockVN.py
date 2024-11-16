import vnstock
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# Hàm lấy dữ liệu cổ phiếu
def get_data_stock(ticker: str, start_date: str, end_date: str):
    data = vnstock.stock_historical_data(ticker, start_date, end_date)
    data_dict = data[['time', 'close', 'ticker']]
    return pd.DataFrame(data_dict)


# Tính toán tỷ suất lợi nhuận và ma trận hiệp phương sai
def calculate_portfolio_metrics(data):
    returns = data.pct_change().dropna()  # Tính tỷ suất sinh lợi hàng ngày
    mean_returns = returns.mean()  # Lợi nhuận trung bình
    cov_matrix = returns.cov()  # Ma trận hiệp phương sai
    return returns, mean_returns, cov_matrix


# Hàm tối ưu hóa danh mục theo mô hình Markowitz
def optimize_portfolio(mean_returns, cov_matrix, num_assets, target_return):
    def portfolio_performance(weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights)
        volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return returns, volatility

    def minimize_volatility(weights, mean_returns, cov_matrix):
        return portfolio_performance(weights, mean_returns, cov_matrix)[1]  # Giảm thiểu độ lệch chuẩn (rủi ro)

    # Các ràng buộc và điều kiện biên
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Tổng tỷ trọng = 1
                   {'type': 'eq', 'fun': lambda x: portfolio_performance(x, mean_returns, cov_matrix)[
                                                       0] - target_return})  # Đảm bảo lợi nhuận kỳ vọng
    bounds = tuple((0, 1) for asset in range(num_assets))
    initial_guess = num_assets * [1. / num_assets]  # Phân bổ đều ban đầu

    result = minimize(minimize_volatility, initial_guess, args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)
    return result


# Vẽ Đường biên hiệu quả (Efficient Frontier)
def plot_efficient_frontier(mean_returns, cov_matrix, num_portfolios=10000):
    results = np.zeros((3, num_portfolios))
    num_assets = len(mean_returns)

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        portfolio_return = np.sum(mean_returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results[0, i] = portfolio_return
        results[1, i] = portfolio_volatility
        results[2, i] = results[0, i] / results[1, i]  # Tỷ lệ Sharpe

    plt.scatter(results[1, :], results[0, :], c=results[2, :], cmap='viridis')
    plt.xlabel('Rủi ro (Độ lệch chuẩn)')
    plt.ylabel('Lợi nhuận kỳ vọng')
    plt.title('Đường biên hiệu quả (Efficient Frontier)')
    plt.colorbar(label='Tỷ lệ Sharpe')
    st.pyplot(plt)


# Danh sách các mã cổ phiếu
ticker_mapping = {
    'VIC': 'VIC',
    'VNM': 'VNM',
    'MSN': 'MSN',
    'HPG': 'HPG',
    'BID': 'BID',
    'VPB': 'VPB',
    'VCB': 'VCB',
    'FPT': 'FPT',    # Công nghệ thông tin
    'MWG': 'MWG',    # Bán lẻ
    'PNJ': 'PNJ',    # Trang sức
    'GVR': 'GVR',    # Cao su
    'VRE': 'VRE',    # Bất động sản bán lẻ
    'VHM': 'VHM',    # Bất động sản nhà ở
    'NVL': 'NVL',    # Bất động sản
    'KDH': 'KDH',    # Bất động sản
    'DIG': 'DIG',    # Bất động sản
    'DXG': 'DXG',    # Bất động sản
    'CTG': 'CTG',    # Ngân hàng
    'TPB': 'TPB',    # Ngân hàng
    'HDB': 'HDB',    # Ngân hàng
    'MBB': 'MBB',    # Ngân hàng
    'EIB': 'EIB',    # Ngân hàng
    'ACB': 'ACB',    # Ngân hàng
    'SSB': 'SSB',    # Ngân hàng
    'STB': 'STB',    # Ngân hàng
    'SHB': 'SHB',    # Ngân hàng
    'PLX': 'PLX',    # Xăng dầu
    'PVD': 'PVD',    # Dịch vụ khoan dầu khí
    'PVS': 'PVS',    # Dầu khí
    'BSR': 'BSR',    # Lọc hóa dầu
    'GAS': 'GAS',    # Khí đốt
    'VJC': 'VJC',    # Hàng không
    'HVN': 'HVN',    # Hàng không
    'VHC': 'VHC',    # Thủy sản
    'ANV': 'ANV',    # Thủy sản
    'DBC': 'DBC',    # Chăn nuôi
    'DPM': 'DPM',    # Phân bón
    'DGC': 'DGC',    # Hóa chất
    'HSG': 'HSG',    # Thép
    'NKG': 'NKG',    # Thép
    'VCS': 'VCS',    # Vật liệu xây dựng
    'FRT': 'FRT',    # Bán lẻ
    'CMG': 'CMG',    # Công nghệ thông tin
    'MSB': 'MSB',    # Ngân hàng
    'HBC': 'HBC',    # Xây dựng
    'REE': 'REE',    # Năng lượng và hạ tầng
    'VCI': 'VCI',    # Chứng khoán
    'SSI': 'SSI',    # Chứng khoán
    'VND': 'VND',    # Chứng khoán
    'HCM': 'HCM',    # Chứng khoán
    'MBS': 'MBS',    # Chứng khoán
    'TCH': 'TCH',    # Bất động sản
    'FCN': 'FCN',    # Xây dựng
    'HAH': 'HAH',    # Vận tải biển
    'TCB': 'TCB',    # Ngân hàng
    'BSI': 'BSI',    # Chứng khoán
    'FTS': 'FTS',    # Chứng khoán
    'SVC': 'SVC',    # Ô tô
    'HRC': 'HRC',    # Cao su
    'COM': 'COM',    # Thương mại
    'VCS': 'VCS',    # Vật liệu xây dựng
    'HAH': 'HAH',
    'TCB': 'TCB',
    'BSI': 'BSI',
    'FTS': 'FTS',
    'SVC': 'SVC',
    'HRC': 'HRC',
    'COM': 'COM',
}



# Hàm chính của ứng dụng
def main():
    st.title("Mô hình Markowitz - Tối ưu hóa danh mục cổ phiếu")

    # Hiển thị công thức của mô hình Markowitz
    st.subheader("Công thức của mô hình Markowitz")
    st.latex(r'''
        \text{Minimize} \quad \sigma_p^2 = w^T \Sigma w
        ''')
    st.latex(r'''
        \text{Với ràng buộc:} \quad \mu_p = w^T \mu \geq r_{\text{target}} \quad \text{và} \quad \sum w_i = 1
        ''')

    # Lựa chọn mã cổ phiếu
    tickers = st.multiselect("Chọn mã cổ phiếu", list(ticker_mapping.keys()))

    # Nhập ngày bắt đầu và ngày kết thúc
    start_date = st.date_input("Ngày bắt đầu", pd.to_datetime('2021-01-01')).strftime('%Y-%m-%d')
    end_date = st.date_input("Ngày kết thúc", pd.to_datetime("today")).strftime('%Y-%m-%d')

    # Hiển thị dữ liệu đóng cửa các mã chứng khoán
    if tickers:
        st.subheader("Giá đóng cửa của các cổ phiếu đã chọn")

        # Lấy dữ liệu và hiển thị giá đóng cửa cho mỗi mã cổ phiếu
        all_data = pd.DataFrame()
        for ticker in tickers:
            data = get_data_stock(ticker, start_date, end_date)
            st.write(f"Dữ liệu giá đóng cửa của mã {ticker}:")
            st.write(data[['time', 'close']])
            all_data[ticker] = data['close']

        # Tính tỷ suất lợi nhuận và ma trận hiệp phương sai
        returns, mean_returns, cov_matrix = calculate_portfolio_metrics(all_data)

        # Hiển thị ma trận hiệp phương sai
        st.subheader("Ma trận hiệp phương sai")
        st.write(cov_matrix)

        # Nhập lợi nhuận kỳ vọng từ người dùng
        target_return = st.slider("Chọn lợi nhuận kỳ vọng (%)", 0.0, 0.3, 0.1, step = 0.001)

        # Tối ưu hóa danh mục đầu tư
        optimized_result = optimize_portfolio(mean_returns, cov_matrix, len(tickers), target_return)

        # Hiển thị tỷ trọng tối ưu cho mỗi cổ phiếu
        st.subheader("Tỷ trọng tối ưu cho mỗi cổ phiếu:")
        for i, ticker in enumerate(tickers):
            st.write(f"{ticker}: {optimized_result.x[i]:.2%}")

        # Vẽ Đường biên hiệu quả
        st.subheader("Đường biên hiệu quả")
        plot_efficient_frontier(mean_returns, cov_matrix)

    else:
        st.write("Vui lòng chọn ít nhất một mã cổ phiếu để hiển thị dữ liệu.")


# Chạy ứng dụng
if __name__ == "__main__":
    main()
