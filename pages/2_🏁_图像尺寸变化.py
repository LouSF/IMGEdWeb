import streamlit as st
import numpy as np
from Image_process import imageP
from Image_process.imageP import CLimages

def page():
    st.set_page_config(page_title="图片尺寸修改", page_icon="🏞️")

    st.markdown("# 通过本页面可以修改图片的尺寸")
    st.sidebar.header("参数调整")
    st.write(
        """
        通过下面的工具栏，可以修改尺寸与旋转
        """
    )

    if imageP.CVimage.have_img:

        # 使用滑块调整尺寸和旋转角度
        width = st.sidebar.slider("宽度", 10, imageP.CVimage.image.shape[1], imageP.CVimage.image.shape[1])
        height = st.sidebar.slider("高度", 10, imageP.CVimage.image.shape[0], imageP.CVimage.image.shape[0])
        angle = st.sidebar.slider("旋转角度🔄", 0, 360, 0)

        # 调整图像
        rotate_and_resize_img = imageP.CVimage.rotate_and_resize(angle, width, height)

        col_org, col_after = st.columns(2)
        with col_org:
            st.image(imageP.CVimage.image, caption="原始图片")

        with col_after:
            if rotate_and_resize_img is not None:
                st.image(rotate_and_resize_img, caption="修改后的图片")

                if st.button("更新图片"):
                    imageP.CVimage.update_img(rotate_and_resize_img)
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
