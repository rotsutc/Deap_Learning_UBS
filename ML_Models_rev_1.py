import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor
import seaborn as sns
import random
import os
import optuna
import lightgbm as lgb
from sklearn.tree import DecisionTreeRegressor
from scipy.stats import skew
from tabulate import tabulate
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import TheilSenRegressor
from sklearn.isotonic import IsotonicRegression


def load_data(file_name):
    # Đường dẫn tương đối dựa trên vị trí file .py
    current_dir = os.path.dirname(__file__)  # Lấy đường dẫn thư mục chứa file .py
    
    file_path = os.path.join(current_dir, file_name)
    
    # Đọc file CSV
    data = pd.read_csv(file_path)
    
    # Loại bỏ các cột không có tiêu đề
    df = data.loc[:, ~data.columns.str.contains('^Unnamed')]
    
    # Loại bỏ các cột không có giá trị
    df = df.dropna(axis=1, how='all')
    
    # Loại bỏ các dòng không có giá trị
    df = df.dropna(axis=0, how='all')
    
    # Hàm kiểm tra giá trị có phải số không
    def is_number(value):
        try:
            float(value)  # Thử chuyển đổi sang số
            return True
        except ValueError:
            return False
    
    # Áp dụng hàm kiểm tra cho toàn bộ DataFrame
    mask = df.map(is_number)
    
    # Xác định dòng và cột bị lỗi
    invalid_entries = np.where(mask == False)  # Các vị trí chứa giá trị không phải số
    rows_with_errors, cols_with_errors = invalid_entries
    
    if len(rows_with_errors) > 0:
        print("Vị trí dữ liệu bị lỗi (dòng, cột):")
        for row, col in zip(rows_with_errors, cols_with_errors):
            print(f"Dòng {row}, Cột '{df.columns[col]}': {df.iloc[row, col]}")
    #else:
        # print("Toàn bộ dữ liệu không bị lỗi.")
    
    return df

def dataset_split_Xy(df):
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
            
    return X, y

def model_data_split (X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size, random_state)
    
    return X_train, y_train, X_test, y_test
    
def database_normalisation(X_train, y_train, X_test, y_test):
    # Chuẩn hóa các đặc trưng X
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)

    # Chuẩn hóa target y
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()  # Chuyển Series thành NumPy array
    y_test = scaler_y.transform(y_test.values.reshape(-1, 1)).flatten()

    return X_train, y_train, X_test, y_test
    
def plot_graph(model_name, y_train, y_train_pred, y_test, y_pred, train_r2, train_rmse, train_mae, train_a2o, 
                   test_r2, test_rmse, test_mae, test_a2o,  error_margin):
        
    # Vẽ biểu đồ
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 2, 1)
    plt.scatter(y_train_pred, y_train, label="Training Data", color="red", alpha=0.6)
        
    # Plot perfect regression line
    # plt.plot(y_test, y_test, color="red", label="Test Perfect Fit", linewidth=2)
    plt.plot(y_train, y_train, color="black", label="Perfect Fit", linewidth=2)
    
    # Plot +/- error margin lines
        
    plt.plot(y_train, y_train * (1 - error_margin), color="red", linestyle=(0, (3, 5)), label=f"+{error_margin:.0%} Error Margin")
    plt.plot(y_train, y_train * (1 + error_margin), color="green", linestyle=(0, (3, 5)), label=f"-{error_margin:.0%} Error Margin")
    
    # Chú thích thông số trên biểu đồ
    test_metrics_text = (
        f"$R^2$: {train_r2:.3f}\n"
        f"RMSE: {train_rmse:.3f}\n"
        f"MAE: {train_mae:.3f}\n"
        f"A2O: {train_a2o:.3f}"
        )
    plt.text(0.05, 0.95, test_metrics_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Cài đặt biểu đồ
    plt.title(f"Comparison of Exact and Predicted Values with {model_name} Model", fontsize=12)
    plt.xlabel("Predicted Data", fontsize=10)
    plt.ylabel("Exact Data", fontsize=10)
    plt.legend()
    plt.grid(True)
    
    
    plt.subplot(1, 2, 2)
    
    plt.scatter(y_pred, y_test, label="Testing Data", color="blue", alpha=0.6)
    
        # Plot perfect regression line
    # plt.plot(y_test, y_test, color="red", label="Test Perfect Fit", linewidth=2)
    plt.plot(y_test, y_test, color="black", label="Perfect Fit", linewidth=2)
    
    # Plot +/- error margin lines
        
    plt.plot(y_test, y_test * (1 - error_margin), color="red", linestyle=(0, (3, 5)), label=f"+{error_margin:.0%} Error Margin")
    plt.plot(y_test, y_test * (1 + error_margin), color="green", linestyle=(0, (3, 5)), label=f"-{error_margin:.0%} Error Margin")
    
    # Chú thích thông số trên biểu đồ
    test_metrics_text = (
        f"$R^2$: {test_r2:.3f}\n"
        f"RMSE: {test_rmse:.3f}\n"
        f"MAE: {test_mae:.3f}\n"
        f"A2O: {test_a2o:.3f}"
        )
    plt.text(0.05, 0.95, test_metrics_text, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    # Cài đặt biểu đồ
    plt.title(f"Comparison of Exact and Predicted Values with {model_name} Model", fontsize=12)
    plt.xlabel("Predicted Data", fontsize=10)
    plt.ylabel("Exact Data", fontsize=10)
    plt.legend()
    plt.grid(True)
    
    
    plt.tight_layout()
    
    # Maximize the plot window (not fullscreen)
    manager = plt.get_current_fig_manager()
    manager.window.state('zoomed')  # For maximized window in Windows
    # Hiển thị biểu đồ
    plt.show()

# Hàm đánh giá mô hình
def evaluate_model(model_name, y_train, y_train_pred, y_test, y_test_pred):
    
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = mean_squared_error(y_train, y_train_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_a2o = np.mean(np.abs(np.array(y_train) - np.array(y_train_pred)))
    
    test_r2 = r2_score(y_test, y_test_pred)  
    test_rmse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_a2o = np.mean(np.abs(np.array(y_test) - np.array(y_test_pred)))
        
    return {"Model": model_name, "Training R2": train_r2, "Tesing R2": test_r2, "Training RMSE": train_rmse, 
            "Tesing RMSE": test_rmse, "Training MAE": train_mae, "Tesing MAE": test_mae, 
            "Training A2O": train_a2o, "Tesing A2O": test_a2o
            }

def run_name_model(name, model, X, y, t_size=0.2, n_iter=30):
    # Lưu trữ kết quả
    data_results = []
    for i in range(n_iter):
        
        # Tách dữ liệu thành 2 tập huấn luyện và kiểm tra
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t_size, random_state=i)
        
        # Chuẩn hóa dữ liệu
        X_train_norm, y_train_norm, X_test_norm, y_test_norm = database_normalisation(X_train, y_train, X_test, y_test)
        
        model.fit(X_train_norm, y_train)  # Huấn luyện
        
        y_test_pred = model.predict(X_test_norm)  # Dự đoán tập kiểm tra
        y_train_pred = model.predict(X_train_norm)  # Dự đoán tập huấn luyện
        
        data_results.append(evaluate_model(name, y_train, y_train_pred, y_test, y_test_pred))
    
    #print(data_results)
    
    # Lấy tên các cột trừ cột đầu tiên
    columns = list(data_results[0].keys())[1:]
    
    # # Tạo thống kê cho từng cột
    # statistics = {}
    # for col in columns:
    #     values = [entry[col] for entry in data_results]
    #     statistics[col] = {
    #         'min': np.min(values),
    #         'avg': np.mean(values),
    #         'std': np.std(values),
    #         'max': np.max(values)
    #         }
    # # In kết quả thống kê
    # for col, stats in statistics.items():
    #     print(f"\nThống kê cho {col}:")
    #     print(f"  Min: {stats['min']:.4f}")
    #     print(f"  Avg: {stats['avg']:.4f}")
    #     print(f"  Std: {stats['std']:.4f}")
    #     print(f"  Max: {stats['max']:.4f}")
    
    # Tạo kết quả dưới dạng yêu cầu
    result = {"Model": name}
    for col in columns:
        values = [e[col] for e in data_results]
        result[f"Min {col}"] = np.min(values)
        result[f"Avg {col}"] = np.mean(values)
        result[f"Std {col}"] = np.std(values)
        result[f"Max {col}"] = np.max(values)
                
    # In kết quả
    # for res in results:
    #     print(res)    
    
    return result

def run_models(X, y, test_size=0.2, number_of_iterations=30):
    
    models = {'Random Forest': RandomForestRegressor(random_state=42), 
              "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
              'XGBoost': XGBRegressor(random_state=42, verbosity=0),
              'Decision Tree': DecisionTreeRegressor(random_state=42),
              'LightGBM': lgb.LGBMRegressor(random_state=42),
              }
    results = []
    for name, model in models.items():
        result_i = run_name_model(name, model, X, y, test_size, number_of_iterations)
        
        results.append(result_i)
        
    #results_df = pd.DataFrame(results).T # Đảo hàng và cột
    
    results_df = pd.DataFrame(results)
    # Hiển thị bảng
    print(results_df)
    #print(tabulate(results_df, headers='keys', tablefmt='grid', floatfmt=".2f", showindex=True))
    
    # Xuất DataFrame thành file CSV
    #results_df.to_csv('models_results.csv', index=False)
    results_df.to_csv('models_results2.csv', index=True)

if __name__ == "__main__":
    # Đọc dư liệu từ file  
    Xy = load_data("Dataset.csv")
    
    X, y = dataset_split_Xy(Xy)
    
    # Chạy các mô hình 
    run_models(X, y, test_size=0.2, number_of_iterations=30)
    
    
        
    