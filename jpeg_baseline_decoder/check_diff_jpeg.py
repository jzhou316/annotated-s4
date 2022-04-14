"""Check the differences of the compressed JPEG codes/bytes from different sources:
from directly reading the .jpg file, or compress the RGB image with cv2 or torchvision.

This is because the current jpeg decoder can work with 'jpegs/kaori.jpg' when it's read from file, but
there is an error when we recompress it to jpegs.
"""
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
import simplejpeg


def jpg_bytes_from_file(path: str):
    with open(path, 'rb') as src:
        buffer = src.read()
    return buffer


def check_jpg_header(jpg_bytes: bytes):
    height, width, colorspace, subsampling = simplejpeg.decode_jpeg_header(jpg_bytes)
    return height, width, colorspace, subsampling


def load_jpg2img(path: str, flag: bool):
    img = cv2.imread(str(path), flag)    # does not work with Path object -> need to convert to str
    # img is np.ndarray, of size (nrows, ncols, 3) if flag else (nrows, ncols)
    return img


def compress_jpg(img: np.ndarray, quality: int):
    # encode image to compressed jpg code
    retval, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    # Default quality value is 95
    # https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html#ga292d81be8d76901bff7988d18d2b42ac
    # retval is bool, buf is np.ndarray
    assert retval, 'failed to encode image'
    jpg_bytes = buf.tobytes()    # type `bytes`

    return jpg_bytes


def compress_jpg_torch(img: np.ndarray, quality: int):
    if img.ndim == 2:
        # grayscale
        img = torch.tensor(img).unsqueeze(0)
    elif img.ndim == 3:
        # RGB, 3 channels
        img = torch.tensor(img).permute(2, 0, 1)

    buf = torchvision.io.encode_jpeg(img, quality=quality)
    # Default quality value is 75
    jpg_bytes = buf.numpy().tobytes()

    return jpg_bytes


def compress_jpg_simplejpeg(img: np.ndarray, quality: int, colorsubsampling: str = '444'):
    if img.ndim == 2:
        # grayscale
        img = np.ascontiguousarray(np.expand_dims(img, axis=2))
        colorspace = 'GRAY'
    elif img.ndim == 3:
        # RGB, 3 channels
        colorspace = 'RGB'

    jpg_bytes = simplejpeg.encode_jpeg(img, quality=quality, colorspace=colorspace, colorsubsampling=colorsubsampling)
    # Default quality value is 85

    return jpg_bytes


if __name__ == '__main__':
    # for this image: bytes_cv2 != bytes_tv
    jpeg_path = Path(__file__).resolve().parent.parent / 'jpegs' / 'kaori.jpg'
    flag = 1

    # # for this image: bytes_cv2 == bytes_tv, quality is close to 83 -> simplejpeg can recover the exact jpeg codes
    # jpeg_path = Path(__file__).resolve().parent.parent / 'jpegs' / '8-bit-256-x-256-Grayscale-Lena-Image_W640.jpg'
    # flag = 0

    bytes_file = jpg_bytes_from_file(jpeg_path)
    print(f'length of bytes_file: {len(bytes_file)}')

    # check the header information
    height, width, colorspace, subsampling = check_jpg_header(bytes_file)
    print('===== original jpeg file =====')
    print(f'height: {height}')
    print(f'width: {width}')
    print(f'colorspace: {colorspace}')
    print(f'color subsampling: {subsampling}')

    # breakpoint()

    img = load_jpg2img(jpeg_path, flag)

    quality = 83

    bytes_cv2 = compress_jpg(img, quality)
    bytes_tv = compress_jpg_torch(img, quality)
    bytes_sim = compress_jpg_simplejpeg(img, quality, colorsubsampling='444')
    # colorsubsampling: subsampling factor for color channels; one of ‘444’, ‘422’, ‘420’, ‘440’, ‘411’, ‘Gray’.

    # NOTE they are not always the same
    print()
    if bytes_cv2 == bytes_tv:
        print('bytes_cv2 == bytes_tv')
    else:
        print('bytes_cv2 != bytes_tv')

    if bytes_cv2 == bytes_sim:
        print('bytes_cv2 == bytes_sim')
    else:
        print('bytes_cv2 != bytes_sim')


    print(f'length of bytes_cv2: {len(bytes_cv2)}')
    print(f'length of bytes_tv: {len(bytes_tv)}')
    print(f'length of bytes_sim: {len(bytes_sim)}')

    # check the header information
    print('===== cv2 encoded jpeg =====')
    height, width, colorspace, subsampling = check_jpg_header(bytes_cv2)
    print(f'height: {height}')
    print(f'width: {width}')
    print(f'colorspace: {colorspace}')
    print(f'color subsampling: {subsampling}')

    # check the header information
    print('===== torchvision encoded jpeg =====')
    height, width, colorspace, subsampling = check_jpg_header(bytes_tv)
    print(f'height: {height}')
    print(f'width: {width}')
    print(f'colorspace: {colorspace}')
    print(f'color subsampling: {subsampling}')

    # check the header information
    print('===== simplejpeg encoded jpeg =====')
    height, width, colorspace, subsampling = check_jpg_header(bytes_sim)
    print(f'height: {height}')
    print(f'width: {width}')
    print(f'colorspace: {colorspace}')
    print(f'color subsampling: {subsampling}')

    # breakpoint()
