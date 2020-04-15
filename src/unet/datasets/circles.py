from typing import Tuple, List

import numpy as np
import tensorflow as tf

channels = 1
classes = 2


def load_data(count:int, splits:Tuple[float]=(0.7, 0.2, 0.1), **kwargs) -> List[tf.data.Dataset]:
    return [tf.data.Dataset.from_tensor_slices(_build_samples(int(split * count), **kwargs))
            for split in splits]


def _build_samples(sample_count:int, nx:int, ny:int, **kwargs) -> Tuple[np.array, np.array]:
    images = np.empty((sample_count, nx, ny, 1))
    labels = np.empty((sample_count, nx, ny, 2))
    for i in range(sample_count):
        image, mask = _create_image_and_mask(nx, ny, **kwargs)
        images[i] = image
        labels[i, ..., 0] = ~mask
        labels[i, ..., 1] = mask
    return images, labels


def _create_image_and_mask(nx, ny, cnt=10, r_min=3, r_max=10, border=32, sigma=20):
    image = np.ones((nx, ny, 1))
    mask = np.zeros((nx, ny), dtype=np.bool)

    for _ in range(cnt):
        a = np.random.randint(border, nx - border)
        b = np.random.randint(border, ny - border)
        r = np.random.randint(r_min, r_max)
        h = np.random.randint(1, 255)

        y, x = np.ogrid[-a:nx - a, -b:ny - b]
        m = x * x + y * y <= r * r
        mask = np.logical_or(mask, m)

        image[m] = h

    image += np.random.normal(scale=sigma, size=image.shape)
    image -= np.amin(image)
    image /= np.amax(image)

    return image, mask
