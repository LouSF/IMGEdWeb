import streamlit as st
import time
import numpy as np
from Image_process import imageP
from Image_process.imageP import CLimages

def page():
    st.set_page_config(page_title="图片的变化", page_icon="🪁")

    st.markdown("# 通过本页面可以进行图像的三角形、S 形、凹形（内凹、外凹）变换")
    st.sidebar.header("参数调整")

    if imageP.CVimage.have_img:
        st.sidebar.subheader("选择操作")
        operation = st.sidebar.selectbox(
            "选择一个操作",
            ("无", "三角形变形", "S形变形", "内外凹变形")
        )

        transformed_img = None


        if operation == "三角形变形":
            transformed_img = imageP.CVimage.warp_to_triangle()

        elif operation == "S形变形":
            range_val = st.sidebar.slider("变形大小", 0, 1500, 1000, step=100)
            transformed_img = imageP.CVimage.warp_s_shape(range_val = range_val)

        elif operation == "内外凹变形":
            range_val = st.sidebar.slider("变形大小", 0., 2., 1., step=0.01)
            transformed_img = imageP.CVimage.warp_concave(range_val)


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
