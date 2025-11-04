import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# =========================
# ⚙️ 1. Cấu hình giao diện
# =========================
st.set_page_config(page_title="Dự đoán cường độ bám dính – DNN", layout="centered")

# CSS tuỳ chỉnh
st.markdown("""
    <style>
    * { font-family: Arial; }
    .title-text { font-size: 18px; font-weight: bold; text-align: center; }
    .subtitle-text { font-size: 15px; font-weight: bold; text-align: center; }
    .info-text { font-size: 14px; text-align: left; margin-left: 20mm; }
    .footer { font-size: 13px; color: gray; text-align: center; margin-top: 30px; }
    </style>
""", unsafe_allow_html=True)

# =========================
# 🧩 2. Load mô hình & scaler
# =========================
try:
    # model = load_model("DNN_bond_strength_model.keras")
    model = load_model("DNN_BatchNormalization_bond_strength_model.keras")
    scaler = joblib.load("scaler.pkl")
    model_loaded = True
except Exception as e:
    st.error("❌ Không thể tải mô hình hoặc scaler. Vui lòng kiểm tra file 'DNN_BatchNormalization_bond_strength_model.keras' và 'scaler.pkl'. Chi tiết lỗi: {e}")
    model_loaded = False

# =========================
# 🖼️ 3. Panel trên cùng (logo)
# =========================
st.image("HCMUTE-fit.png", width='stretch')

st.markdown("---")

# =========================
# 📘 4. Thông tin đồ án
# =========================
st.markdown("""
<div class="subtitle-text">ĐỒ ÁN CUỐI KỲ MÔN HỌC SÂU</div>
<div class="title-text">ỨNG DỤNG DNN TRONG BÀI TOÁN DỰ ĐOÁN CƯỜNG ĐỘ BÁM DÍNH CỦA CỐT THÉP TRONG BÊ TÔNG</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-text">
GVHD: PGS. TS Hoàng Văn Dũng<br>
Nhóm: 1<br>
Học viên: NGUYỄN THÀNH QUÍ – MSHV: 2591320<br>
Học viên: TRẦN THỊ BẢO MY – MSHV: 2591314
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# =========================
# 📊 5. Nhập dữ liệu
# =========================
st.subheader("🔹 Nhập thông số đầu vào")

col_left, col_right = st.columns(2)

with col_left:
    X1 = st.number_input("X1 = Cường độ bê tông [MPa]", min_value=10.0, max_value=100.0, value=40.0, step=1.0)
    X2 = st.number_input("X2 = Lớp bê tông bảo vệ [mm]", min_value=10.0, max_value=100.0, value=30.0, step=1.0)
    X3 = st.selectbox("X3 = Loại thép", options=[1, 2],
                      format_func=lambda x: "1 = Thép trơn" if x == 1 else "2 = Thép gân")

with col_right:
    X4 = st.number_input("X4 = Đường kính thanh thép [mm]", min_value=6.0, max_value=40.0, value=16.0, step=1.0)
    X5 = st.number_input("X5 = Chiều dài neo [mm]", min_value=50.0, max_value=500.0, value=150.0, step=5.0)
    X6 = st.number_input("X6 = Mức độ ăn mòn [%]", min_value=0.0, max_value=20.0, value=2.0, step=0.1)

# =========================
# 🔮 6. Dự đoán
# =========================
if st.button("🔹 Dự đoán"):
    if model_loaded:
        X_input = np.array([[X1, X2, X3, X4, X5, X6]])
        X_scaled = scaler.transform(X_input)
        y_pred = model.predict(X_scaled)
        y_pre = float(y_pred.flatten()[0])

        st.markdown(f"<h3 style='text-align:center; color:blue;'>Cường độ bám dính = {y_pre:.2f} MPa</h3>",
                    unsafe_allow_html=True)
    else:
        st.warning("⚠️ Mô hình chưa sẵn sàng để dự đoán.")

# =========================
# 📜 7. Footer
# =========================
st.markdown("""
<div class="footer">
Toàn bộ dữ liệu chỉ sử dụng cho mục đích học tập và nghiên cứu.
</div>
""", unsafe_allow_html=True)


