import streamlit as st
import time
import numpy as np
from Image_process import imageP
from Image_process.imageP import CLimages

def page():
    st.set_page_config(page_title="变换方法", page_icon="🥅")

    st.markdown("# 图像处理工具")
    st.sidebar.header("参数调整")

    if imageP.CVimage.have_img:
        st.sidebar.subheader("选择操作")
        operation = st.sidebar.selectbox(
            "选择一个操作",
            ("无", "Hough 变换", "Otsu 变换", "车道线检测（Hough 变换）")
        )

        processed_img = None

        if operation == "Hough 变换":
            threshold = st.sidebar.slider("阈值", 1, 100, 50)
            min_line_length = st.sidebar.slider("最小线段长度", 1, 200, 100)
            max_line_gap = st.sidebar.slider("最大线段间隙", 1, 200, 100)
            processed_img = imageP.CVimage.hough_transform(threshold, min_line_length, max_line_gap)

        elif operation == "Otsu 变换":
            processed_img = imageP.CVimage.otsu_transform()

        elif operation == "车道线检测（Hough 变换）":
            canny_threshold1 = st.sidebar.slider("Canny 阈值 1", 50, 150, 50)
            canny_threshold2 = st.sidebar.slider("Canny 阈值 2", 50, 150, 150)
            hough_threshold = st.sidebar.slider("Hough 阈值", 1, 100, 50)
            min_line_length = st.sidebar.slider("最小线段长度", 1, 200, 50)
            max_line_gap = st.sidebar.slider("最大线段间隙", 1, 200, 50)

            processed_img = imageP.CVimage.lane_detection(canny_threshold1, canny_threshold2, hough_threshold,
                                                           min_line_length, max_line_gap)

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
