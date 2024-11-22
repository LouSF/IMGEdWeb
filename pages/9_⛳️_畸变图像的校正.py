import streamlit as st
import time
import numpy as np
from Image_process import imageP
from Image_process.imageP import CLimages

def page():
    st.set_page_config(page_title="畸变图像校正", page_icon="⛳️")

    st.markdown("# 通过本页面可以进行畸变图像校正")
    st.sidebar.header("参数调整")

    if imageP.CVimage.have_img:
        k1 = st.sidebar.slider("k1", -1.0, 1.0, 0.0)
        k2 = st.sidebar.slider("k2", -1.0, 1.0, 0.0)
        p1 = st.sidebar.slider("p1", -1.0, 1.0, 0.0)
        p2 = st.sidebar.slider("p2", -1.0, 1.0, 0.0)
        transformed_img = imageP.CVimage.correct_distortion(k1, k2, p1, p2)

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
