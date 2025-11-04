import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import joblib

# === 1. Đọc dữ liệu ===
data = pd.read_csv("Processed_Data.csv")

# Đảm bảo tên cột đúng theo yêu cầu
expected_cols = [
    "Compressive strength [MPa]",
    "Concrete cover [mm]",
    "Rebar type [1=Plain bar; 2=Deformed bar]",
    "Diameter of rebar [mm]",
    "Bond length [mm]",
    "Corrosion level [%]",
    "Ultimate bond strength [MPa]"
]
data.columns = expected_cols  # Gán lại tên cột (phòng trường hợp khác tên)

# === 2. Tách X, Y ===
X = data.iloc[:, 0:6].values
y = data.iloc[:, 6].values

# === 3. Chia train/test (80/20) ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Chuẩn hóa dữ liệu ===
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# === 5. Hàm đánh giá ===
def evaluate_model(model, X_train, y_train, X_test, y_test):
    y_train_pred = model.predict(X_train).flatten()
    y_test_pred = model.predict(X_test).flatten()

    metrics = {
        "Train R2": r2_score(y_train, y_train_pred),
        "Test R2": r2_score(y_test, y_test_pred),
        "Train MSE": mean_squared_error(y_train, y_train_pred),
        "Test MSE": mean_squared_error(y_test, y_test_pred),
        "Train MAE": mean_absolute_error(y_train, y_train_pred),
        "Test MAE": mean_absolute_error(y_test, y_test_pred)
    }
    return metrics

# === 6. Xây dựng mô hình ANN (1 hidden layer, 10 neurons) ===
def build_ann():
    model = Sequential([
        Input(shape=(6,)),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# === 7. Xây dựng mô hình DNN (5 hidden layers, 10 neurons mỗi lớp) ===
def build_dnn():
    model = Sequential([
        Input(shape=(6,)),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_dnn_dropout_20():
    model = Sequential([
        Input(shape=(6,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def build_dnn_BatchNormalization():
    model = Sequential([
        Input(shape=(6,)),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# === 8. Huấn luyện ===
ann = build_ann()
dnn = build_dnn()
#dnn_dropout = build_dnn_dropout_20()
dnn_BatchNormalization = build_dnn_BatchNormalization()

print("\nTraining ANN...")
ann.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1)

print("Training DNN...")
dnn.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1)

# print("Training DNN with Dropout 20%...")
# dnn_dropout.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1, validation_split=0.1)
print("Training DNN with Batch Normalization...")
dnn_BatchNormalization.fit(X_train, y_train, epochs=200, batch_size=16, verbose=1, validation_split=0.1)

# === 9. Đánh giá ===
ann_metrics = evaluate_model(ann, X_train, y_train, X_test, y_test)
dnn_metrics = evaluate_model(dnn, X_train, y_train, X_test, y_test)
#dnn_dropout_metrics = evaluate_model(dnn_dropout, X_train, y_train, X_test, y_test)
dnn_BatchNormalization_metrics = evaluate_model(dnn_BatchNormalization, X_train, y_train, X_test, y_test)

# === 10. Hiển thị kết quả ===
print("\n=== ANN Performance ===")
for k, v in ann_metrics.items():
    print(f"{k}: {v:.4f}")

print("\n=== DNN Performance ===")
for k, v in dnn_metrics.items():
    print(f"{k}: {v:.4f}")

# print("\n=== DNN with Dropout 20% Performance ===")
# for k, v in dnn_dropout_metrics.items():
#     print(f"{k}: {v:.4f}")

print("\n=== DNN with Batch Normalization Performance ===")
for k, v in dnn_BatchNormalization_metrics.items():
    print(f"{k}: {v:.4f}")
    
# === 11. Lưu mô hình DNN ===

dnn.save('DNN_bond_strength_model.keras')
print("\n✅ DNN model saved successfully as 'DNN_bond_strength_model.keras'")
# dnn_dropout.save('DNN_Dropout20_bond_strength_model.keras')
# print("✅ DNN with Dropout 20% model saved successfully as 'DNN_Dropout20_bond_strength_model.keras'")

dnn_BatchNormalization.save('DNN_BatchNormalization_bond_strength_model.keras')
print("✅ DNN with Batch Normalization model saved successfully as 'DNN_BatchNormalization_bond_strength_model.keras'")

# Lưu scaler sau khi huấn luyện
joblib.dump(scaler, "scaler.pkl")
print("✅ Scaler saved successfully as 'scaler.pkl'")