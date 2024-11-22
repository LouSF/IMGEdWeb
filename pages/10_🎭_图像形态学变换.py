import streamlit as st
import time
import numpy as np
from Image_process import imageP
from Image_process.imageP import CLimages

def page():
    st.set_page_config(page_title="图像形态学变换", page_icon="🎭")

    st.markdown("# 通过本页面可以进行畸变图像校正")
    st.sidebar.header("参数调整")

    if imageP.CVimage.have_img:

        st.sidebar.subheader("选择操作")
        operation = st.sidebar.selectbox(
            "选择一个操作",
            ("无", "去除噪声", "提取边界", "校正光照")
        )

        processed_img = None

        if operation == "去除噪声":
            shape = st.sidebar.slider("核大小", 1, 10, 1, step=1)
            iterations = st.sidebar.slider("迭代次数", 1, 10, 1, step=1)
            processed_img = imageP.CVimage.remove_noise(shape, iterations)
        elif operation == "提取边界":
            shape = st.sidebar.slider("核大小", 1, 10, 1, step=1)
            processed_img = imageP.CVimage.extract_edges(shape)
        elif operation == "校正光照":

            operation = st.sidebar.selectbox(
                "选择一个运算",
                ("顶帽运算", "底帽运算")
            )

            shape = st.sidebar.slider("核大小", 1, 100, 1, step=1)

            if operation == "顶帽运算":
                processed_img = imageP.CVimage.correct_uneven_illumination('TOPHAT', shape)
            elif operation == "底帽运算":
                processed_img = imageP.CVimage.correct_uneven_illumination('BLACKHAT', shape)

        else:
            processed_img = imageP.CVimage.image

        col_org, col_after = st.columns(2)
        with col_org:
            st.image(imageP.CVimage.image, caption="原始图片")

        with col_after:
            if processed_img is not None:
                st.image(processed_img, caption="修改后的图片")

                if st.button("更新图片"):
                    imageP.CVimage.update_img(processed_img)
                    st.success("图片已更新")

    else:
        st.write(
            """
            **目前未找到图片！**

            请上传图片以使用工具。
            """
        )


if __name__ == "__main__":
    page()