import numpy as np
import pytest

from unet import utils


@pytest.mark.parametrize("channels", [
    1,2,3,4
])
def test_to_rgb(channels):
    tensor = np.random.normal(size=(5, 12, 12, channels))

    tensor[1, 5, 5, 0] = np.nan

    rgb_img = utils.to_rgb(tensor)

    assert rgb_img.shape[:2] == tensor.shape[:2]
    assert rgb_img.shape[3] == 3

    assert rgb_img.min() == 0
    assert rgb_img.max() == 1
