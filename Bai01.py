import streamlit as st
import pandas as pd

file = st.file_uploader("Up file")

# đọc dữ liệu
if not (file is None):
    data = pd.read_csv(file)

# hiển thị dữ liệu
if not (file is None):
    st.write(data)

# thống kê mô tả dữ liệu:
if not (file is None):
    st.write(data.describe())

# vẽ biểu đồ histogram
if not (file is None):
    chon_cot = st.selectbox("Chọn cột", data.columns)
    st.bar_chart(data[chon_cot])

# tính hệ số tương quan
if not (file is None):
    he_so_tuong_quan = data.corr()
    st.write(he_so_tuong_quan)

# chọn biến phụ thuộc và vẽ biểu đồ phân tán
if not (file is None):
    bien_phu_thuoc = st.selectbox("Chọn biến phụ thuộc", data.columns)
    chon_bien_doc_lap = st.multiselect("Chọn biến độc lâp", data.columns)
    for variable in chon_bien_doc_lap:
        st.scatter_chart(data[[bien_phu_thuoc, variable]])
