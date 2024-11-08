import streamlit as st
import time
import numpy as np
from Image_process import imageP
from Image_process.imageP import CLimages

def page():
    st.set_page_config(page_title="灰度变化", page_icon="🔳")

    st.markdown("# 通过本页面可以应用灰度变换")
    st.sidebar.header("参数调整")
    st.write(
        """
        通过下面的工具栏，可以应用灰度变换
        """
    )

    if imageP.CVimage.have_img:

        alpha = st.sidebar.slider("线性变换 - alpha", 0.1, 3.0, 1.0)
        beta = st.sidebar.slider("线性变换 - beta", 0, 100, 0)
        gamma = st.sidebar.slider("伽马变换", 0.1, 3.0, 1.0)

        transformed_img = CLimages.gamma_transform(imageP.CVimage.image, gamma)
        transformed_img = CLimages.linear_transform(imageP.CVimage.image, alpha, beta)

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
