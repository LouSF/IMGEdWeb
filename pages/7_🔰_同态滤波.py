import streamlit as st
import time
import numpy as np
from Image_process import imageP
from Image_process.imageP import CLimages, CVimage


def page():
    st.set_page_config(page_title="同态滤波", page_icon="🔰")

    st.markdown("# 通过本页面可以应用同态滤波")
    st.sidebar.header("参数调整")
    st.write(
        """
        通过下面的工具栏，可以修改同态滤波的参数
        """
    )

    if imageP.CVimage.have_img:

        cutoff_freq = st.sidebar.slider("截止频率", 1, 100, 30)
        gamma_l = st.sidebar.slider("低频增益", 0.1, 1.0, 0.5)
        gamma_h = st.sidebar.slider("高频增益", 1.0, 3.0, 2.0)
        c = st.sidebar.slider("c", 2, 0, 5)

        transformed_img = CLimages.homomorphic_filter_torch(CVimage.image, cutoff_freq,gamma_l, gamma_h, c)

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
