import streamlit as st
import pandas as pd
import pickle
import os

st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    .Button button {
        background-color: #333333;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 18px;
        color: #ffffff;
    }
    .stTextInput, .stNumberInput input {
        background-color: #333333;
        color: white;
        border: 2px solid #ffffff;
    }
    .stSidebar {
        background-color: #000000;
        color: #ffffff;
    }
    .stSidebar h3, .stSidebar p, .stSidebar label, .stSidebar .header-text {
        color: #ffffff;
    }
    h1 {
        font-family: 'Courier New', monospace;
        color: #ffffff;
    }
    h3, p {
        font-family: 'Arial', sans-serif;
        color: #ffffff;
    }
    .uploadedFile {
        font-family: 'Arial', sans-serif;
        color: #cccccc;
    }
    .center {
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .center img {
        width: 150px;
        height: 150px;
        object-fit: contain;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="center">
        <img src="https://media.tenor.com/oqJo9GcbfjYAAAAi/welcome-images-server.gif" alt="Welcome Image">
    </div>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center; color: #4CAF50;">Prediksi IPK - 1965</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #00736e;">Aplikasi ini berguna untuk memprediksi IPK berdasarkan nilai Matematika, Bahasa Inggris, dan Bahasa Indonesia</p>', unsafe_allow_html=True)

st.sidebar.markdown('<h3 class="header-text">Upload File dan Input Nilai</h3>', unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader('Upload file dataset_regresi_IPK.csv', type="csv")

if uploaded_file is not None:
    input_data = pd.read_csv(uploaded_file)
    st.markdown('<h3 style="text-align: center; color: #00736e;">Data yang diupload:</h3>', unsafe_allow_html=True)
    st.dataframe(input_data)

    model_path = r'D:\Documents Eliandani\Tugas\Semester 5\Machine Learning\Tugas3_A_11965\SVR_IPK_model.pkl'
    
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            loaded_model = pickle.load(f)
        
        scaler = loaded_model[0]
        feature_selector = loaded_model[1]
        SVR_model = loaded_model[2]

        st.sidebar.subheader("Masukkan Nilai")
        mk1 = st.sidebar.number_input("Nilai Matematika Semester 1.1", 0.0, 100.0)
        mk2 = st.sidebar.number_input("Nilai Matematika Semester 1.2", 60.0, 100.0)
        mk3 = st.sidebar.number_input("Nilai Matematika Semester 2.1", 60.0, 100.0)
        mk4 = st.sidebar.number_input("Nilai Matematika Semester 2.2", 60.0, 100.0)

        ing1 = st.sidebar.number_input("Nilai Bahasa Inggris Semester 1.1", 62.0, 99.0)
        ing2 = st.sidebar.number_input("Nilai Bahasa Inggris Semester 1.2", 64.0, 99.0)
        ing3 = st.sidebar.number_input("Nilai Bahasa Inggris Semester 2.1", 50.0, 99.0)
        ing4 = st.sidebar.number_input("Nilai Bahasa Inggris Semester 2.2", 60.0, 99.0)

        ind1 = st.sidebar.number_input("Nilai Bahasa Indonesia Semester 1.1", 67.0, 99.0)
        ind2 = st.sidebar.number_input("Nilai Bahasa Indonesia Semester 1.2", 70.0, 99.0)
        ind3 = st.sidebar.number_input("Nilai Bahasa Indonesia Semester 2.1", 60.0, 99.0)
        ind4 = st.sidebar.number_input("Nilai Bahasa Indonesia Semester 2.2", 70.0, 100.0)

        st.sidebar.subheader("Masukkan KKN Nilai")
        kkn_ind1 = st.sidebar.number_input("KKN Nilai Bahasa Indonesia Semester 1.1", 0.0, 100.0)
        kkn_ind2 = st.sidebar.number_input("KKN Nilai Bahasa Indonesia Semester 1.2", 0.0, 100.0)
        kkn_ind3 = st.sidebar.number_input("KKN Nilai Bahasa Indonesia Semester 2.1", 0.0, 100.0)
        kkn_ind4 = st.sidebar.number_input("KKN Nilai Bahasa Indonesia Semester 2.2", 0.0, 100.0)

        kkn_ing1 = st.sidebar.number_input("KKN Nilai Bahasa Inggris Semester 1.1", 0.0, 100.0)
        kkn_ing2 = st.sidebar.number_input("KKN Nilai Bahasa Inggris Semester 1.2", 0.0, 100.0)
        kkn_ing3 = st.sidebar.number_input("KKN Nilai Bahasa Inggris Semester 2.1", 0.0, 100.0)
        kkn_ing4 = st.sidebar.number_input("KKN Nilai Bahasa Inggris Semester 2.2", 0.0, 100.0)

        kkn_mtk1 = st.sidebar.number_input("KKN Nilai Matematika Semester 1.1", 0.0, 100.0)
        kkn_mtk2 = st.sidebar.number_input("KKN Nilai Matematika Semester 1.2", 0.0, 100.0)
        kkn_mtk3 = st.sidebar.number_input("KKN Nilai Matematika Semester 2.1", 0.0, 100.0)
        kkn_mtk4 = st.sidebar.number_input("KKN Nilai Matematika Semester 2.2", 0.0, 100.0)

        input_data = [[mk1, mk2, mk3, mk4, ing1, ing2, ing3, ing4, ind1, ind2, ind3, ind4,
                       kkn_mtk1, kkn_mtk2, kkn_mtk3, kkn_mtk4, kkn_ing1, kkn_ing2, kkn_ing3, kkn_ing4,
                       kkn_ind1, kkn_ind2, kkn_ind3, kkn_ind4]]

        input_data_scaled = scaler.transform(input_data)
        input_data_selected = feature_selector.transform(input_data_scaled)

        if st.sidebar.button("Prediksi"):
            SVR_model_predict = SVR_model.predict(input_data_selected)
            st.markdown(f'<h3 style="text-align: center; color: #4CAF50;">Prediksi IPK adalah: {SVR_model_predict[0]:.2f}</h3>', unsafe_allow_html=True)
    else:
        st.error("Model tidak ditemukan, silakan cek file model di direktori.")
