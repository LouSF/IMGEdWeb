import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import torch
from PIL.ImageMath import imagemath_max


class CLimages:
    def __init__(self, input_image = None):
        self.image = input_image
        self.img_shape = 0
        self.img_dtype = 0
        self.img_size = 0
        self.have_img = False

        if self.image is not None:
            self.__sizeupdate__()
            self.have_img = True

    def update_img(self, input_image):
        self.image = input_image
        self.__sizeupdate__()

    def trans2jpg(self):
        rgb_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        _, buffer = cv2.imencode('.png', rgb_image)
        return BytesIO(buffer)


    def __sizeupdate__(self):
        self.img_shape = self.image.shape
        self.img_dtype = self.image.dtype
        self.img_size = self.image.size

    def resize(self, width, height):
        if self.have_img:
            return cv2.resize(self.image, (width, height))
        return None

    def rotate(self, angle):
        if self.have_img:
            (h, w) = self.image.shape[:2]
            center = (w // 2, h // 2)

            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            rotated_img = cv2.warpAffine(self.image, M, (new_w, new_h))

            return cv2.resize(rotated_img, (w, h))
        return None

    def rotate_and_resize(self, angle, width, height):
        if self.have_img:
            # 获取图像尺寸
            (h, w) = self.image.shape[:2]
            center = (w // 2, h // 2)

            # 计算旋转矩阵
            M = cv2.getRotationMatrix2D(center, angle, 1.0)

            # 计算旋转后的新尺寸
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            new_w = int((h * sin) + (w * cos))
            new_h = int((h * cos) + (w * sin))

            # 调整旋转矩阵中的平移部分
            M[0, 2] += (new_w / 2) - center[0]
            M[1, 2] += (new_h / 2) - center[1]

            # 旋转图像
            rotated_img = cv2.warpAffine(self.image, M, (new_w, new_h))

            # 调整图像大小
            resized_img = cv2.resize(rotated_img, (width, height))

            return resized_img
        return None

    @staticmethod
    def linear_transform(image, alpha, beta):
        """线性变换"""
        return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    @staticmethod
    def gamma_transform(image, gamma):
        """伽马变换"""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def histogram_equalization(self):
        if self.have_img:
            if len(self.image.shape) == 2:
                return cv2.equalizeHist(self.image)
            else:
                ycrcb = cv2.cvtColor(self.image, cv2.COLOR_BGR2YCrCb)
                ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
                return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        return None

    def apply_median_blur(self, kernel_size):
        """中值滤波"""
        return cv2.medianBlur(self.image, kernel_size)

    def apply_mean_blur(self, kernel_size):
        """均值滤波"""
        return cv2.blur(self.image, (kernel_size, kernel_size))

    def apply_gaussian_blur(self, kernel_size):
        """高斯滤波"""
        return cv2.GaussianBlur(self.image, (kernel_size, kernel_size), 0)

    # 弃用
    # def median_filter(self, ksize=3):
    #     if self.have_img:
    #         return cv2.medianBlur(self.image, ksize)
    #     return None
    #
    # def mean_filter(self, ksize=3):
    #     if self.have_img:
    #         return cv2.blur(self.image, (ksize, ksize))
    #     return None
    #
    # def gaussian_filter(self, ksize=3, sigma=0):
    #     if self.have_img:
    #         return cv2.GaussianBlur(self.image, (ksize, ksize), sigma)
    #     return None

    def laplacian_sharpen(self):
        if self.have_img:
            laplacian = cv2.Laplacian(self.image, cv2.CV_64F)
            sharp = cv2.convertScaleAbs(self.image - laplacian)
            return sharp
        return None



    @staticmethod
    def homomorphic_filter_torch(img, cutoff_freq, gamma_l, gamma_h, c):
        image_array = []
        for image in cv2.split(img):
            image = torch.tensor(image, dtype=torch.float32)
            log_image = torch.log1p(image)
            log_image = log_image / torch.log(torch.tensor(256.0))

            dft = torch.fft.fftshift(torch.fft.fft2(log_image))

            rows, cols = image.shape
            row_mid, col_mid = rows // 2, cols // 2
            y, x = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing='ij')
            dist = torch.sqrt((x - col_mid) ** 2 + (y - row_mid) ** 2)
            mask = (gamma_h - gamma_l) * (1 - torch.exp(-c * (dist ** 2 / cutoff_freq ** 2))) + gamma_l

            dft_filtered = dft * mask

            filtered_image = torch.fft.ifft2(torch.fft.ifftshift(dft_filtered))
            filtered_image = torch.exp(torch.abs(filtered_image)) - 1
            filtered_image = (filtered_image - filtered_image.min()) / (filtered_image.max() - filtered_image.min())
            filtered_image = filtered_image * 255
            filtered_image = torch.clamp(filtered_image, 0, 255).byte()

            image_array.append(filtered_image.numpy())

        image_merge = cv2.merge(image_array)

        return image_merge


CVimage:CLimages = CLimages()


## []
# CVimage:CLimages = CLimages(np.array(Image.open('./test/test_img.png')))
