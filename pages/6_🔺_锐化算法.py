import streamlit as st
import time
import numpy as np
from Image_process import imageP
from Image_process.imageP import CLimages

def page():
    st.set_page_config(page_title="导入原始图片", page_icon="🏞️")

    st.markdown("# 通过本页面可以导入原始图片")
    st.sidebar.header("导入原始图片")
    st.write(
        """
        在下面的工具栏中，可以选择拍照或者是上传图片👇
        """
    )


if __name__ == "__main__":
    page()