import streamlit as st
import time
import numpy as np
from Image_process import imageP
from Image_process.imageP import CLimages

def page():
    st.set_page_config(page_title="直方图增强", page_icon="📊")

    st.markdown("# 通过本页面可以应用直方图增强")
    st.sidebar.header("此处无需参数调整")
    st.write(
        """
        通过下面的工具栏可以应用直方图增强
        """
    )

    if imageP.CVimage.have_img:
        transformed_img = imageP.CVimage.histogram_equalization()
        col_org, col_after = st.columns(2)
        with col_org:
            st.image(imageP.CVimage.image, caption="原始图片")

        with col_after:
            if transformed_img is not None:
                st.image(transformed_img, caption="直方图增强后的图片")

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
