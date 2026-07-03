

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import os
# nenyembunyikan log informasi dan peringatan dari TensorFlow di terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
st.set_page_config(
    page_title="NASA Airfoil Noise Predictor",
    layout="wide"
)

st.title("NASA Airfoil Noise Predictor (NASA Self-Noise)")
st.markdown("""
Aplikasi ini memanfaatkan arsitektur **Deep Learning (1D-CNN & SimpleRNN)** untuk memprediksi tingkat tekanan bunyi 
(*Scaled Sound Pressure Level* - SSPL) berdasarkan parameter aliran fisis udara di terowongan angin.
""")

@st.cache_resource
def load_ml_components():
    try:
        model_cnn = tf.keras.models.load_model('model_cnn_air_reg.keras')
        model_rnn = tf.keras.models.load_model('model_rnn_air_reg.keras')
        scaler_X = joblib.load('scaler_X_air_reg.pkl')
        scaler_y = joblib.load('scaler_y_air_reg.pkl')
        return model_cnn, model_rnn, scaler_X, scaler_y
    except:
        return None, None, None, None

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv('AirfoilSelfNoise.csv')
    except:
        return None

# Memuat komponen ML dan dataset
model_cnn, model_rnn, scaler_X, scaler_y = load_ml_components()
df = load_dataset()

st.sidebar.header("parameter Fisis Airfoil")

f = st.sidebar.number_input("Frekuensi Bunyi (Hz)", min_value=200, max_value=20000, value=1200, step=100)
alpha = st.sidebar.slider("Sudut Serang / Angle of Attack (deg)", min_value=0.0, max_value=22.2, value=0.0, step=0.1)
c = st.sidebar.slider("Panjang Profil Sayap / Chord Length (m)", min_value=0.025, max_value=0.305, value=0.10, step=0.001)
U_infinity = st.sidebar.slider("Kecepatan Aliran / Free-stream velocity (m/s)", min_value=31.7, max_value=71.3, value=39.5, step=0.1)
delta = st.sidebar.number_input("Tebal Lapisan Batas / Displacement Thickness (m)", min_value=0.0004, max_value=0.0584, value=0.00160, format="%.5f")

chosen_model = st.sidebar.selectbox("Pilih Arsitektur Deep Learning", ["1D-CNN (Convolutional)", "SimpleRNN"])
predict_btn = st.sidebar.button("kalkulasi Prediksi Kebisingan")
tab1, tab2 = st.tabs(["Predictive Calculator", "Dataset Analytics & Visualizations"])

with tab1:#tab 1 for calculating the predicted SSPL based on user input parameters
    if predict_btn:
        if model_cnn is not None and scaler_X is not None:
            input_raw = np.array([[f, alpha, c, U_infinity, delta]])
            input_scaled = scaler_X.transform(input_raw)
            input_dl = input_scaled.reshape(input_scaled.shape[0], 1, 5)
            
            if chosen_model == "1D-CNN (Convolutional)":
                prediction_scaled = model_cnn.predict(input_dl)
            else:
                prediction_scaled = model_rnn.predict(input_dl)
                
            final_sspl = scaler_y.inverse_transform(prediction_scaled)[0][0]
            
            st.header("hasil Analisis Prediksi Aeroakustik")
            
            col1, col2 = st.columns([2, 3])
            with col1:
                st.markdown(f"**Predicted Sound Pressure Level (SSPL)**")
                st.markdown(f"<h1 style='font-size: 56px; margin-top: -10px; color: #E91E63;'>{final_sspl:.2f} dB</h1>", unsafe_allow_html=True)
            with col2:
                st.write("")
                if final_sspl < 115:
                    st.success(f"status: Tingkat Kebisingan Rendah (Aman)")
                elif 115 <= final_sspl <= 130:
                    st.warning(f"status: Tingkat Kebisingan Sedang (Perlu Peredam)")
                else:
                    st.error(f"status: Tingkat Kebisingan Tinggi (Bahaya Resonansi)")
                    
            st.info(f"Kombinasi Parameter Pengujian: Frekuensi **{f} Hz** dengan Sudut Serang **{alpha}°** menghasilkan estimasi gelombang suara reflektif pada profil sayap NACA 0012 sebesar **{final_sspl:.2f} dB**.")
            
            if df is not None:
                avg_sspl = df['SSPL'].mean()
                diff = final_sspl - avg_sspl
                if diff > 0:
                    st.markdown(f"🔺 Nilai prediksi ini **{abs(diff):.2f} dB lebih tinggi** dari rata-rata empiris laboratorium ({avg_sspl:.2f} dB).")
                else:
                    st.markdown(f"🔻 Nilai prediksi ini **{abs(diff):.2f} dB lebih rendah** dari rata-rata empiris laboratorium ({avg_sspl:.2f} dB).")
        else:
            st.error("Komponen model Deep Learning gagal dimuat. Pastikan file .keras dan .pkl ada di folder.")
    else:
        st.write("Silakan sesuaikan parameter pada panel kiri dan klik tombol **'Kalkulasi Prediksi Kebisingan'** untuk melihat hasil analisis.")

with tab2:#tab2 for dataset analytics and visualizations from jupyter notebook
    if df is not None:
        st.header("analisis Karakteristik Aeroakustik Eksperimental")
        st.markdown("Berikut adalah ringkasan statistik dan distribusi sebaran data empiris terowongan angin NASA NACA 0012:")
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Total Sampel Eksperimen", f"{df.shape[0]}")
        col_m2.metric("Rata-Rata SSPL", f"{df['SSPL'].mean():.2f} dB")
        col_m3.metric("Kebisingan Minimum", f"{df['SSPL'].min():.2f} dB")
        col_m4.metric("Kebisingan Maksimum", f"{df['SSPL'].max():.2f} dB")
        
        st.subheader("Sampel Data Mentah (5 Baris Pertama)")
        st.dataframe(df.head(5), use_container_width=True)
        
        st.subheader("analisis Sebaran Frekuensi & Korelasi")
        col_c1, col_col2 = st.columns(2)
        
        with col_c1:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.histplot(df['SSPL'], bins=30, kde=True, color='#E91E63', ax=ax)
            ax.set_title('Distribusi Variabel Target Kebisingan (SSPL)', fontweight='bold')
            ax.set_xlabel('Sound Pressure Level (dB)')
            ax.set_ylabel('Frekuensi Kemunculan')
            st.pyplot(fig)
            
        with col_col2:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax, annot_kws={"size": 9})
            ax.set_title('Matriks Korelasi Parameter Fisis Airfoil', fontweight='bold')
            st.pyplot(fig)
            
        st.subheader("Tren Urutan Sekuensial Tingkat Tekanan Suara (Target)")
        fig, ax = plt.subplots(figsize=(14, 3))
        ax.plot(df['SSPL'].values, color='#2196F3', alpha=0.8, linewidth=0.8)
        ax.set_title('Deret Indeks Sekuensial Kebisingan — Airfoil Self-Noise', fontweight='bold')
        ax.set_xlabel('Indeks Sampel Pengujian')
        ax.set_ylabel('SSPL (dB)')
        st.pyplot(fig)
        
    else:
        st.warning("File 'AirfoilSelfNoise.csv' tidak ditemukan. Pastikan file diletakkan selevel dengan app.py untuk membuka fitur analitik data.")