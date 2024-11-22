import streamlit as st
import cv2
import numpy as np
from PIL import Image
from Image_process import imageP

def intro():
    st.set_page_config(
        page_title="DIP",
        page_icon="👋",
    )

    st.write("# 欢迎来到数字图像处理APP")
    st.sidebar.success("选择一个模块")

    st.markdown(
        """
        **这是基于Web的数字图像处理课程设计👋**

        目前已经实现：
        1. 图像灰度变换（线性变换和非线性伽马）
        2. 图像直方图增强（直方图均衡化）
        3. 图像平滑算法（中值滤波、均值滤波、高斯滤波）
        4. 图像锐化算法（拉普拉斯算子）
        5. 图像频率域同态滤波
        6. 图像的三角形、S 形、凹形（内凹、外凹）变换
        7. 畸变图像的校正
        8. 图像的形态学变化

        **👈请在边栏选择一个项目**

        """
    )

def main():
    intro()

if __name__ == "__main__":
    main()