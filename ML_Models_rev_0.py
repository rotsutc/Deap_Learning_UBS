import pandas as pd
import numpy as np
from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, accuracy_score
from sklearn.svm import SVR, SVC
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
#from catboost import CatBoostRegressor
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

def data_split(df):
    
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Tách dữ liệu thành tập huấn luyện 80% và kiểm tra 20%
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_test, y_test, X, df
    
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
    
    # # Maximize the plot window (not fullscreen)
    # manager = plt.get_current_fig_manager()
    # manager.window.state('zoomed')  # For maximized window in Windows
    # Hiển thị biểu đồ
    plt.show()
    

def best_RF_model(X_train, y_train):
    # Tìm tham số tốt nhất với GridSearchCV cho mô hình Random Forest
    
    param_grid_rf = {'n_estimators': [50, 100, 200],'max_depth': [None, 5, 10],'min_samples_split': [2, 5, 10]} 
    rf = RandomForestRegressor(random_state=42)
    grid_rf = GridSearchCV(rf, param_grid_rf, cv=3, scoring='r2', verbose=1)
    grid_rf.fit(X_train, y_train)
    best_rf = grid_rf.best_estimator_
    
    return best_rf, dict(grid_rf.best_params_.items())

def best_XGBoost_model(X_train, y_train):
    # Cải tiến mô hình với GridSearchCV cho XGBoost
       
    param_grid_XGB = {'n_estimators': [100, 200, 300], 
                  'max_depth': [3, 5, 7],
                  'learning_rate': [0.01, 0.1, 0.2],
                  'subsample': [0.7, 0.8, 1.0],
                  'colsample_bytree': [0.7, 0.8, 1.0]
                  }
    
    XGB = XGBRegressor()
    
    grid_XGB = GridSearchCV(XGB, param_grid_XGB, cv=3, scoring='r2', verbose=1)
    grid_XGB.fit(X_train, y_train)
    best_XGB = grid_XGB.best_estimator_
    
    return best_XGB, dict(grid_XGB.best_params_.items())

def best_SVR_model(X_train, y_train):
    # Khởi tạo SVR
    svr = SVR()
    
    # Định nghĩa lưới tham số
    param_grid = {'kernel': ['linear', 'rbf', 'poly'],  # Các loại kernel
                  'C': [0.1, 1, 10, 100],               # Giá trị C
                  'epsilon': [0.01, 0.1, 0.5, 1],       # Giá trị epsilon
                  'gamma': ['scale', 'auto', 0.1, 1]    # Giá trị gamma
                  }
        
    # GridSearchCV
    grid_search = GridSearchCV(estimator=svr,
                               param_grid=param_grid,
                               cv=3, # Cross-validation với 3 folds
                               scoring='neg_mean_squared_error',  # Sử dụng MSE làm tiêu chí đánh giá
                               verbose=1) #v erbose=1 show information
    # Huấn luyện
    grid_search.fit(X_train, y_train)
    # Dự đoán với mô hình tốt nhất
    best_SVR = grid_search.best_estimator_
    
    return best_SVR, dict(grid_search.best_params_.items())

def tpe_best_SVR_model(X_train, y_train):
    # Hàm mục tiêu cho TPE
    def objective(params):
        model = SVR(C=params['C'], epsilon=params['epsilon'], kernel=params['kernel'])
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error').mean()
        return {'loss': -score, 'status': STATUS_OK}
    
    # Không gian tìm kiếm tham số
    space = {
        'C': hp.loguniform('C', np.log(0.1), np.log(100)),  # Thu hẹp từ [1e-3, 1e3] xuống [0.1, 100]
        'epsilon': hp.uniform('epsilon', 0.01, 0.3),       # Thu hẹp từ [0.01, 0.5] xuống [0.01, 0.3]
        'kernel': hp.choice('kernel', ['linear', 'rbf'])   # Tập trung vào kernel phổ biến
        }
    
    # Tìm tham số tốt nhất
    trials = Trials()
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)
    # Huấn luyện mô hình tốt nhất
    best_model = SVR(C=best['C'], epsilon=best['epsilon'], kernel=['linear', 'poly', 'rbf', 'sigmoid'][best['kernel']])
    
    return best_model, best

def tpe_best_SVM_model(X_train, y_train):
    # Hàm mục tiêu cho TPE
    def objective(params):
        model = SVR(
            kernel=params['kernel'],
            C=params['C'],
            epsilon=params['epsilon'],
            gamma=params['gamma']
            )
        
        score = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=3).mean()
        return -score
    
    # Không gian tìm kiếm tham số
    space = {
        'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
        'C': hp.uniform('C', 100, 200),  
        'epsilon': hp.uniform('epsilon', 0.7, 0.8), 
        'gamma': hp.uniform('gamma', 0.1, 0.2)  #hp.choice('gamma', ['scale', 'auto', 0.1, 0.2])  
        }
    
    # Khởi tạo Trials để theo dõi quá trình tối ưu
    trials = Trials()
    
    # Tối ưu hóa tham số
    best_params = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
        )
    
    # Chuyển đổi kernel về dạng tên
    
    best_params['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid'][best_params['kernel']]
    
    # # In kết quả tham số tốt nhất
    # print("Best Parameters:")
    # print(best_params)
    
    # Huấn luyện SVR với tham số tối ưu
    best_model = SVR(
        kernel=best_params['kernel'],
        C=best_params['C'],
        epsilon=best_params['epsilon'],
        gamma=best_params['gamma']
        )
    
    return best_model, best_params


def tpe_best_RF_model(X_train, y_train):
    # Hàm mục tiêu để tối ưu hóa
    def objective(params):
        """Hàm tối ưu hóa hyperparameter cho RandomForestRegressor"""
        # Tạo mô hình RandomForestRegressor với các tham số từ TPE
        model = RandomForestRegressor(
            n_estimators=int(params['n_estimators']),
            max_depth=int(params['max_depth']),
            min_samples_split=int(params['min_samples_split']),
            min_samples_leaf=int(params['min_samples_leaf']),
            random_state=42,
            n_jobs=-1
            )
        
        # Đánh giá bằng cross-validation
        score = cross_val_score(model, X_train, y_train, cv=3, scoring='neg_mean_squared_error').mean()
        return -score  # Trả về giá trị lỗi (MSE) để minimization
    
    # Không gian tham số cho TPE
    space = {
        'n_estimators': hp.quniform('n_estimators', 50, 300, 10),
        'max_depth': hp.quniform('max_depth', 5, 30, 1),
        'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1),
        'min_samples_leaf': hp.quniform('min_samples_leaf', 1, 5, 1),
        }
    
    # Khởi tạo TPE và tìm tham số tối ưu
    trials = Trials()
    best = fmin(
        fn=objective,         # Hàm mục tiêu
        space=space,          # Không gian tham số
        algo=tpe.suggest,     # Thuật toán tối ưu hóa (TPE)
        max_evals=50,         # Số lần đánh giá
        trials=trials         # Lưu thông tin các lần thử
        )
    #In kết quả
    #print("Best hyperparameters:", best)
    
    # Huấn luyện mô hình với tham số tối ưu
    optimized_model = RandomForestRegressor(
        n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),
        min_samples_split=int(best['min_samples_split']),
        min_samples_leaf=int(best['min_samples_leaf']),
        random_state=42,
        n_jobs=-1
        )
    
    return optimized_model, best

def print_best_params(model_name, params):
    # In tham số tốt nhất và giá trị
    print(f"Best parameters of {model_name} Model:")
    for param_name, param_value in params.items():
        print(f"{param_name} = {param_value}") 

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

def tpe_svr(X_train, y_train, X_test, y_test, max_trials=50):
    #best_loss = float('inf')
    best_r2 = float('-inf')
    best_params = None
    trial_number = 0

    # Lưu kết quả từng lần thử
    results = []

    for _ in range(max_trials):
        trial_number += 1
        # Random hóa tham số
        kernel = 'rbf' #random.choice(['linear', 'poly', 'rbf', 'sigmoid'])
        C = random.uniform(1, 1000)
        epsilon = random.uniform(0.1, 1)
        gamma = random.uniform(0.1, 1) #random.choice(['scale', 'auto']
        
        # Huấn luyện mô hình
        model = SVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        #loss = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Lưu kết quả từng lần thử
        results.append({
            'Trial': trial_number,
            'Kernel': kernel,
            'C': C,
            'Epsilon': epsilon,
            'Gamma': gamma,
            'R2': r2
        })

        #print(f"Trial {trial_number}: Kernel={kernel}, C={C:.6f}, Epsilon={epsilon:.6f}, Gamma={gamma}, R2={r2:.6f}")
        print(f"Trial {trial_number}...Please wait...")

        # Cập nhật giá trị tốt nhất
        if r2 > best_r2:
            best_r2 = r2
            best_params = {
                'Kernel': kernel,
                'C': C,
                'Epsilon': epsilon,
                'Gamma': gamma
            }
            print(f"Trial {trial_number}: Kernel={kernel}, C={C:.6f}, Epsilon={epsilon:.6f}, Gamma={gamma}, R2={r2:.6f}")

    print(f"\nBest Parameters: {best_params}")
    print(f"Best R2: {best_r2:.6f}")

    # Trả về kết quả tất cả các thử nghiệm
    return results

def optuna_RFR_TPE(X_train, y_train, n_trials=50, timeout=60): #Thử n_trials trong thời gian timeout giây
    # Hàm mục tiêu để tối ưu
    def objective(trial):
        # Tạo các tham số cần tối ưu
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 10, 50)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        
        
        # Khởi tạo mô hình với tham số
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            bootstrap=bootstrap,
            random_state=42,
            n_jobs=-1
        )
        
        # Tính điểm thông qua cross-validation
        score = cross_val_score(model, X_train, y_train, scoring="neg_mean_squared_error", cv=5, n_jobs=-1)
        return -score.mean()  # Trả về MSE (cần giá trị nhỏ nhất)
    
    # Tối ưu tham số bằng Optuna
    study = optuna.create_study(direction="minimize")  # Tối ưu để giảm thiểu MSE
    study.optimize(objective, n_trials, timeout)
    
    # Kết quả tốt nhất
    print("Best trial:")
    print("  Value: ", study.best_value)  # MSE tốt nhất
    print("  Params: ", study.best_params)  # Tham số tốt nhất
 
def LightGBM_model(X_train, y_train, X_test, y_test):
    # Tạo DataSet cho LightGBM
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    # Tham số mô hình
    params = {
        'objective': 'regression',        # Bài toán hồi quy (ví dụ: 'binary', 'multiclass', 'regression').
        'boosting_type': 'gbdt',          # Gradient Boosting Decision Tree ('gbdt', 'dart', 'goss', 'rf')
        'metric': 'rmse',                 # Đánh giá bằng RMSE (ví dụ: 'binary_logloss', 'auc', 'rmse')
        'num_leaves': 31,                 # Số lượng lá tối đa của mỗi cây (lớn hơn sẽ tăng độ phức tạp)
        'learning_rate': 0.05,            # Tốc độ học (quá nhỏ sẽ cần nhiều vòng lặp, quá lớn dễ overfit)  
        'feature_fraction': 0.9           # Tỷ lệ các đặc trưng được sử dụng trong mỗi cây (ví dụ: 0.9 nghĩa là sử dụng 90% đặc trưng)
        }
    # Huấn luyện mô hình
    lgb_model = lgb.train(params, train_data, valid_sets=[test_data], num_boost_round=100, early_stopping_rounds=10)
    
    # Dự đoán và đánh giá
    y_pred = lgb_model.predict(X_test)  # Dự đoán tập kiểm tra
    y_train_pred = lgb_model.predict(X_train)  # Dự đoán tập huấn luyện
                
    test_r2 = r2_score(y_test, y_pred)  # Tính R2
    test_rmse = mean_squared_error(y_test, y_pred) # Tính toán lỗi trung bình bình phương RMSE (Root Mean Squared Error)
    test_mae = mean_absolute_error(y_test, y_pred) # Tính MAE (Mean Absolute Error)
    test_a2o = np.mean(np.abs(np.array(y_test) - np.array(y_pred))) # Tính A2O (Mean Absolute Error)
    
    train_r2 = r2_score(y_train, y_train_pred)  # Tính R2
    train_rmse = mean_squared_error(y_train, y_train_pred) # Tính toán lỗi trung bình bình phương RMSE (Root Mean Squared Error)
    train_mae = mean_absolute_error(y_train, y_train_pred) # Tính MAE (Mean Absolute Error)
    train_a2o = np.mean(np.abs(np.array(y_train) - np.array(y_train_pred))) # Tính A2O (Mean Absolute Error)    
    
    # Vẽ biểu đồ các mô hình
    plot_graph("LightGBM", y_train, y_train_pred, y_test, y_pred, train_r2, train_rmse, train_mae, train_a2o, 
                   test_r2, test_rmse, test_mae, test_a2o, 0.2)
   
def run_models(X_train, y_train, X_test, y_test):
    
    # # Huấn luyện và đánh giá các mô hình
    # models = {"Linear Regression": LinearRegression(), 
    #            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    #            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    #            "ANN": MLPRegressor(hidden_layer_sizes=(64, 32),  activation='relu', solver='adam', max_iter=500, random_state=42),
    #            "XGBoost": XGBRegressor(),
    #            "LightGBM": lgb.LGBMRegressor(boosting_type='gbdt', n_estimators=100, learning_rate=0.1, random_state=42),
    #            "SVR": SVR(kernel='rbf', C=100, epsilon=0.1)               
    #            }
    
    # models = {'Random Forest': RandomForestRegressor(random_state=42),
    #           'XGBoost': XGBRegressor(random_state=42, verbosity=0),
    #           'LightGBM': lgb.LGBMRegressor(random_state=42),
    #           'Decision Tree': DecisionTreeRegressor(random_state=42),
    #           'ANN': MLPRegressor(random_state=42, max_iter=1000),
    #           'SVR': SVR(),
    #           'AdaBoost': AdaBoostRegressor(n_estimators=100),
    #           'CatBoost': CatBoostRegressor(iterations=100),
    #           'Bayesian Ridge': BayesianRidge(),
    #           'Gradient Boosting': GradientBoostingRegressor(n_estimators=100),
    #           'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
    #           'Gaussian Process': GaussianProcessRegressor(),
    #           'Quantile': QuantileRegressor(alpha=0.5),
    #           'Partial Least Squares': PLSRegression(n_components=2),
    #           'Locally Weighted': RANSACRegressor(),
    #           'Generalized Linear': PoissonRegressor(),
    #           'Huber': HuberRegressor(),
    #           'Orthogonal Matching Pursuit': OrthogonalMatchingPursuit(),
    #           'Theil-Sen Estimator': TheilSenRegressor(),
    #           'Isotonic': IsotonicRegression()                         
    #           }
    
    models = {'Random Forest': RandomForestRegressor(random_state=42),
              'XGBoost': XGBRegressor(random_state=42, verbosity=0),
              'LightGBM': lgb.LGBMRegressor(random_state=42),
              'Decision Tree': DecisionTreeRegressor(random_state=42),
              'ANN': MLPRegressor(random_state=42, max_iter=1000),
              'SVR': SVR(),
              'Gradient Boosting': GradientBoostingRegressor(n_estimators=100),
              }
    
    # models = {"LightGBM": lgb.LGBMRegressor(boosting_type='gbdt',         # Loại boosting
    #                                         num_leaves=31,                # Số lá (leaf nodes)
    #                                         max_depth=-1,                 # Độ sâu tối đa của cây
    #                                         learning_rate=0.1,            # Tốc độ học
    #                                         n_estimators=100,             # Số lượng cây
    #                                         subsample=1.0,                # Tỷ lệ mẫu khi xây dựng mỗi cây
    #                                         colsample_bytree=1.0,         # Tỷ lệ cột được sử dụng khi xây dựng mỗi cây
    #                                         reg_alpha=0.0,                # Hệ số điều chỉnh L1 (regularization)
    #                                         reg_lambda=0.0,               # Hệ số điều chỉnh L2 (regularization)
    #                                         random_state=42,              # Seed cho kết quả tái lập
    #                                         importance_type='split',      # Kiểu tính tầm quan trọng của đặc trưng
    #                                         n_jobs=-1
    #                                         )
    #           }
    
    # models = {"SVM": SVR(C= 4658.553823226561, epsilon= 0.7786207614697529, kernel= 'rbf', gamma= 0.1596953484086365)               
    #            }
    
    # models = {"SVM": SVR(C= 100, epsilon= 0.78, kernel= 'rbf', gamma= 0.16)               
    #            }
    
    # models = {"ANN": MLPRegressor(hidden_layer_sizes=(64, 32),  # 2 hidden layers với 64 và 32 nodes
    #                               activation='relu',            # Hàm kích hoạt 'relu', 'tanh', 'logistic', hoặc 'identity'
    #                               solver='adam',                # Optimizer
    #                               max_iter=500,                 # Số vòng lặp tối đa
    #                               random_state=42)
    #           }
    
    # Lưu trữ kết quả
    results = {}    
    for name, model in models.items():
        model.fit(X_train, y_train)  # Huấn luyện
        
        y_pred = model.predict(X_test)  # Dự đoán tập kiểm tra
        y_train_pred = model.predict(X_train)  # Dự đoán tập huấn luyện
                
        test_r2 = r2_score(y_test, y_pred)  # Tính R2
        test_rmse = mean_squared_error(y_test, y_pred) # Tính toán lỗi trung bình bình phương RMSE (Root Mean Squared Error)
        test_mae = mean_absolute_error(y_test, y_pred) # Tính MAE (Mean Absolute Error)
        test_a2o = np.mean(np.abs(np.array(y_test) - np.array(y_pred))) # Tính A2O (Mean Absolute Error)
        
        train_r2 = r2_score(y_train, y_train_pred)  # Tính R2
        train_rmse = mean_squared_error(y_train, y_train_pred) # Tính toán lỗi trung bình bình phương RMSE (Root Mean Squared Error)
        train_mae = mean_absolute_error(y_train, y_train_pred) # Tính MAE (Mean Absolute Error)
        train_a2o = np.mean(np.abs(np.array(y_train) - np.array(y_train_pred))) # Tính A2O (Mean Absolute Error)
        
        metrics = {
            'Training RMSE': train_rmse,
            'Testing RMSE': test_rmse,
            'Training MAE': train_mae,
            'Testing MAE': test_mae,
            'Training R2': train_r2,
            'Testing R2': test_r2,
            'Training A2O': train_a2o,
            'Testing A2O': test_a2o,
            }        
        results[name] = metrics
        # print(f"Mô hình: {name}, R2: {r2:.4f} RMSE =  {rmse:.4f} MAE =  {mae:.4f} A2O =  {a2o:.4f}")
        # Vẽ biểu đồ các mô hình
        plot_graph(name, y_train, y_train_pred, y_test, y_pred, train_r2, train_rmse, train_mae, train_a2o, 
                   test_r2, test_rmse, test_mae, test_a2o, 0.2)
    
    # Xuất kết quả đánh giá giống Table 6
    #results_df = pd.DataFrame(results).T
    # results_df = results_df[['Training MSE', 'Testing MSE', 'Training MAE', 'Testing MAE', 
    #                          'Training R2', 'Testing R2', 'Training A2O', 'Testing A2O']]
    #print("Model Performance:\n", results_df)

def show_correlation_matrix(X, Xy):
    # Tính ma trận tương quan
    correlation_matrix_Xy = Xy.corr()
    correlation_matrix_X = X.corr()
    
    
    # Hiển thị ma trận tương quan
    print("Ma trận tương quan X và y")
    print(correlation_matrix_Xy)
    
    print("Ma trận tương quan X")
    print(correlation_matrix_X)
    
    # Vẽ biểu đồ
    plt.figure(figsize=(15, 10))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(correlation_matrix_Xy, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of X and y")
    
    plt.subplot(1, 2, 2)
    sns.heatmap(correlation_matrix_X, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix of X")
    
    plt.tight_layout()
    
    # # Maximize the plot window (not fullscreen)
    # manager = plt.get_current_fig_manager()
    # manager.window.state('zoomed')  # For maximized window in Windows
    # Hiển thị biểu đồ
    plt.show()

def calculate_statistical(data):
    def calculate_statistics(df):
        stats = {
            "Min": df.min(),
            "Mean": df.mean(),
            "Std": df.std(),
            "Skewness": df.apply(lambda col: skew(col, nan_policy='omit')),
            "Max": df.max()
            }
        stats_df = pd.DataFrame(stats)
        return stats_df.round(4)  # Làm tròn 4 chữ số thập phân
    
    # # Gộp X_train và y_train
    # data_combined = pd.concat([X_train, y_train], axis=1)
    
    # Tính toán các thống kê
    statistics_df = calculate_statistics(data)
    
    # Định dạng bảng
    headers = statistics_df.columns.insert(0, "Variables")  # Thêm cột đầu tiên
    table = tabulate(
        statistics_df, 
        headers=headers, 
        tablefmt="grid", 
        colalign=("left", *["right"] * (len(headers) - 1))  # Cột đầu trái, các cột còn lại phải
        )
    
    # Hiển thị bảng
    print("Statistics Table:")
    print(table)
    
    # Hiển thị kết quả
    #print("Statistics Table:")
    #print(statistics_df)
    #print(tabulate(statistics_df, headers='keys', tablefmt='grid'))

if __name__ == "__main__":
    # Đọc dư liệu từ file    
    X_train, y_train, X_test, y_test, X, Xy = data_split(load_data("UBS_X1_X6_output_remove_outliers_20250109_215259.csv"))
    
    # Hiển thị giá trị và biểu đồ ma trận tương quan
    show_correlation_matrix (X, Xy)
    
    # Hiển thị các kết quả thống kê
    calculate_statistical(Xy)
    
    # Chạy các mô hình chưa chuẩn hóa
    run_models(X_train, y_train, X_test, y_test)
    
    # Chuẩn hóa dữ liệu
    #X_train_norm, y_train_norm, X_test_norm, y_test_norm = database_normalisation(X_train, y_train, X_test, y_test)
    
    #optuna_RFR_TPE(X_train_norm, y_train)
    
    # Chạy các mô hình chuẩn hóa
    #run_models(X_train_norm, y_train_norm, X_test_norm, y_test_norm)
    # run_models(X_train_norm, y_train, X_test_norm, y_test)
    #LightGBM_model(X_train_norm, y_train, X_test_norm, y_test)
    # best, params = best_RF_model(X_train_norm, y_train)
    # print_best_params("Random Forest Regressor", params)
    
    # best, params = tpe_best_SVR_model(X_train_norm, y_train)
    # print_best_params("SVR", params)
    
    # best, params = tpe_best_SVM_model(X_train_norm, y_train)
    # print_best_params("SVM", params)
    
    # Gọi hàm TPE với SVM
    # results = tpe_svr(X_train_norm, y_train, X_test_norm, y_test, max_trials=5000)
    
    # # Chuyển kết quả thành DataFrame để dễ dàng phân tích
    # results_df = pd.DataFrame(results)
    # print("\nTất cả kết quả thử nghiệm:")
    # print(results_df)
    
        
    