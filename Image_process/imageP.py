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

    # to fix it
    # def warp_triangle(self, src_points, dst_points):
    #     if self.have_img:
    #         # 获得仿射变换矩阵
    #         M = cv2.getAffineTransform(np.float32(src_points), np.float32(dst_points))
    #         # 应用仿射变换
    #         warped_img = cv2.warpAffine(self.image, M, (self.image.shape[1], self.image.shape[0]))
    #         return warped_img
    #     return None

    def warp_to_triangle(self):
        if self.have_img:
            rows, cols = self.image.shape[:2]

            src_pts = np.float32([
                [0, 0],
                [cols - 1, 0],
                [cols - 1, rows - 1],
                [0, rows - 1]
            ])

            dst_pts = np.float32([
                [0, 0],
                [cols - 1, 0],
                [0, rows - 1],
                [0, 0]
            ])

            matrix = cv2.getAffineTransform(src_pts[:3], dst_pts[:3])
            transformed_img = cv2.warpAffine(self.image, matrix, (cols, rows))
            return transformed_img
        return None

    def warp_s_shape(self, range_val):
        if self.have_img:
            height, width = self.image.shape[:2]
            distorted_img = np.zeros_like(self.image)

            for i in range(height):
                temp = float(
                    (width - range_val) / 2 + (width - range_val) * np.sin((2 * np.pi * i) / height + np.pi) / 2)
                for j in range(int(temp + 0.5), int(range_val + temp)):
                    m = int(((j - temp) * width / range_val))
                    if m >= width:
                        m = width - 1
                    if m < 0:
                        m = 0
                    distorted_img[i, j] = self.image[i, m]
            return distorted_img
        return None

    import numpy as np

    def warp_concave(self, intensity=1):
        if self.have_img:
            height, width, channels = self.image.shape
            image = np.zeros((height, width, channels), dtype=np.uint8)

            for i in range(height):
                temp = float((width - width // 2 * intensity) * np.sin(((2 * np.pi * i) / height) /2))

                if intensity < 1:
                    temp = width  - temp

                for j in range(int(temp + 0.5), int(width - temp)):
                    distance = int(width - temp) - int(temp + 0.5)
                    ratio = distance / width
                    stepsize = 1.0 / ratio
                    for c in range(channels):
                        index = int((j - temp) * stepsize)
                        image[i, j, c] = self.image[i, index, c]
            return image
        return None

    def correct_distortion(self, k1, k2, p1, p2):
        if self.have_img:
            h, w = self.image.shape[:2]
            dist_coeffs = np.array([k1, k2, p1, p2, 0])
            camera_matrix = np.array([[w, 0, w / 2],
                                      [0, h, h / 2],
                                      [0, 0, 1]])

            new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
            corrected_img = cv2.undistort(self.image, camera_matrix, dist_coeffs, None, new_camera_matrix)
            return corrected_img
        return None

    def remove_noise(self, shape, iterations):
        if self.have_img:
            kernel = np.ones((shape, shape), np.uint8)
            denoised = self.image
            denoised = cv2.dilate(denoised, kernel, iterations=iterations)
            denoised = cv2.erode(denoised, kernel, iterations=iterations)
            _, denoised = cv2.threshold(denoised, 127, 255, cv2.THRESH_BINARY)
            return denoised
        return None

    def extract_edges(self, shape):
        if self.have_img:
            kernel = np.ones((shape, shape), np.uint8)
            tmp_img = cv2.erode(self.image, kernel)
            # edges = tmp_img - self.image
            edges = cv2.bitwise_xor(tmp_img, self.image)
            edges = cv2.bitwise_not(edges)
            return edges
        return None

    def correct_uneven_illumination(self, type, shape):
        if self.have_img:
            if type == 'TOPHAT':
                type = cv2.MORPH_TOPHAT
            elif type == 'BLACKHAT':
                type = cv2.MORPH_BLACKHAT

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (shape, shape))
            tophat = cv2.morphologyEx(self.image, type, kernel)
            tophat = cv2.bitwise_not(tophat)
            return tophat
        return None

    def hough_transform(self, threshold, min_line_length, max_line_gap):
        if self.have_img:
            edges = cv2.Canny(self.image, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=min_line_length,
                                    maxLineGap=max_line_gap)
            line_image = np.copy(self.image)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            return line_image
        return None

    def otsu_transform(self):
        if self.have_img:
            gray = self.image if len(self.image.shape) == 2 else cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            _, otsu_thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return otsu_thresh
        return None

    def lane_detection(self, canny_threshold1, canny_threshold2, hough_threshold, min_line_length, max_line_gap):
        if self.have_img:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

            _, gray_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

            edges = cv2.Canny(gray_img, canny_threshold1, canny_threshold2)

            mask = np.zeros_like(edges)
            height, width = edges.shape
            polygon = np.array([[
                (0, height),
                (width, height),
                (width - width // 2.5, height // 2.7),
                (width // 2.5, height // 2.7),
            ]], dtype=np.int32)
            cv2.fillPoly(mask, polygon, 255)

            masked_edges = cv2.bitwise_and(edges, mask)

            lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, hough_threshold,
                                    minLineLength=min_line_length,
                                    maxLineGap=max_line_gap)


            line_image = np.zeros_like(self.image)

            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            combined_image = cv2.addWeighted(self.image, 0.8, line_image, 1, 0)
            return combined_image
        return None




CVimage:CLimages = CLimages()


## [测试部分]
# CVimage:CLimages = CLimages(np.array(Image.open('./test/test_img.png')))
# CVimage:CLimages = CLimages(np.array(Image.open('./test/work3_1.png')))
# CVimage:CLimages = CLimages(np.array(Image.open('./test/work5.jpg')))
# CVimage:CLimages = CLimages(np.array(Image.open('./test/Original-image-of-Lena-in-512x512-pixels.png')))
# CVimage:CLimages = CLimages(np.array(Image.open('./test/polymercell.tif')))

