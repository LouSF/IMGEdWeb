import streamlit as st
import time
import numpy as np
from Image_process import imageP
from Image_process.imageP import CLimages

def page():
    st.set_page_config(page_title="平滑算法", page_icon="〰️")

    st.markdown("# 通过本页面可以应用图像平滑")
    st.sidebar.header("参数调整")
    st.write(
        """
        通过下面的工具栏，可以应用图像平滑
        """
    )

    if imageP.CVimage.have_img:

        # 选择平滑算法
        smoothing_option = st.sidebar.selectbox(
            "选择平滑算法",
            ("无", "中值滤波", "均值滤波", "高斯滤波")
        )

        # 设置核大小
        if smoothing_option != "无":
            kernel_size = st.sidebar.slider("核大小", 1, 31, 3, step=2)

        # 应用选择的平滑算法
        if smoothing_option == "中值滤波":
            transformed_img = imageP.CVimage.apply_median_blur(kernel_size)
        elif smoothing_option == "均值滤波":
            transformed_img = imageP.CVimage.apply_mean_blur(kernel_size)
        elif smoothing_option == "高斯滤波":
            transformed_img = imageP.CVimage.apply_gaussian_blur(kernel_size)
        else:
            transformed_img = imageP.CVimage.image

        col_org, col_after = st.columns(2)
        with col_org:
            st.image(imageP.CVimage.image, caption="原始图片")

        with col_after:
            if transformed_img is not None:
                st.image(transformed_img, caption="修改后的图片")

                if st.button("更新图片"):
                    imageP.CVimage.update_img(transformed_img)
                    st.success("图片已更新")

    else:
        st.write(
            """
            **目前未找到图片！**

            工具已隐藏
            """
        )

if __name__ == "__main__":
    page()
