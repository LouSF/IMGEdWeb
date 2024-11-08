import streamlit as st
import numpy as np
from PIL import Image
from Image_process import imageP
from Image_process.imageP import CLimages


def page():
    st.set_page_config(page_title="导入原始图片", page_icon="🏞️")

    st.markdown("# 通过本页面可以导入原始图片")
    st.sidebar.header("导入原始图片")
    st.write(
        """
        在下面的工具栏中，可以选择拍照或者是上传图片👇
        """
    )
    st.markdown("## 数据载入")

    input_img = None

    # todo fix take photo
    # st.markdown("### 拍照")
    # st.write(
    #     """
    #     通过本模块，可以使用相机拍照并上传
    #     """
    # )
    # enable = st.checkbox("启动相机")
    # if enable:
    #     camera_img = st.camera_input("让我们拍一张照片📸")
    #     if camera_img is not None:
    #         input_img = np.array(camera_img)

    st.markdown("### 上传照片")
    st.write(
        """
        通过本模块，直接上传本地图片
        """
    )
    image_file = st.file_uploader("选择图片文件", type=["jpg", "jpeg", "png"])
    if image_file is not None:
        input_img = np.array(Image.open(image_file))

    if input_img is not None:
        imageP.CVimage = CLimages(input_img)

    if imageP.CVimage.have_img:
        with st.container():
            st.markdown("#### 图片预览")
            st.image(imageP.CVimage.image)
            st.markdown("#### 文件参数")
            file_info = {
                "属性": ["文件尺寸", "存储的数据类型", "占用大小"],
                "值": [imageP.CVimage.img_shape, imageP.CVimage.img_dtype, imageP.CVimage.img_size]
            }
            st.table(file_info)

            st.markdown("#### 导出照片")
            st.write(
                """
                通过本模块，直接导出图片
                """
            )

            btn = st.download_button(
                label="导出图片",
                data=imageP.CVimage.trans2jpg(),
                file_name="output.jpg",
                mime="image/png",
            )


if __name__ == "__main__":
    page()