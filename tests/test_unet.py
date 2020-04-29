from unittest.mock import Mock, patch

from tensorflow.keras import layers
from tensorflow.keras import losses

from unet import unet


def test_build_model():
    nx = 572
    ny = 572
    channels = 3
    num_classes = 2
    kernel_size = 3
    pool_size = 2
    filters_root = 64
    layer_depth = 5
    model = unet.build_model(nx=nx,
                             ny=ny,
                             channels=channels,
                             num_classes=num_classes,
                             layer_depth=layer_depth,
                             filters_root=filters_root,
                             kernel_size=kernel_size,
                             pool_size=pool_size)

    input_shape = model.get_layer("inputs").output.shape
    assert tuple(input_shape) == (None, nx, ny, channels)
    output_shape = model.get_layer("outputs").output.shape
    assert tuple(output_shape) == (None, 388, 388, num_classes)

    filters_per_layer = [filters_root, 128, 256, 512, 1024, 512, 256, 128, filters_root]
    conv2D_layers = _collect_conv2d_layers(model)

    assert len(conv2D_layers) == 2 * len(filters_per_layer) + 1

    for conv2D_layer in conv2D_layers[:-1]:
        assert conv2D_layer.kernel_size == (kernel_size, kernel_size)

    for i, filters in enumerate(filters_per_layer):
        assert conv2D_layers[i*2].filters == filters
        assert conv2D_layers[i*2+1].filters == filters

    maxpool_layers = [layer for layer in model.layers if type(layer) == layers.MaxPool2D]

    assert len(maxpool_layers) == layer_depth - 1

    for maxpool_layer in maxpool_layers[:-1]:
        assert maxpool_layer.pool_size == (pool_size, pool_size)


def _collect_conv2d_layers(model):
    conv2d_layers = []
    for layer in model.layers:
        if type(layer) == layers.Conv2D:
            conv2d_layers.append(layer)
        elif type(layer) == unet.ConvBlock:
            conv2d_layers.append(layer.conv2d_1)
            conv2d_layers.append(layer.conv2d_2)

    return conv2d_layers


@patch.object(unet, "Adam")
def test_finalize_model(AdamMock:Mock):
    adam_instance = Mock()
    AdamMock.return_value = adam_instance
    metric_mock = Mock(name="metric")
    model = Mock(name="model")

    loss = losses.binary_crossentropy
    learning_rate = 1.0

    unet.finalize_model(model,
                        loss=loss,
                        optimizer=None,
                        metrics=[metric_mock],
                        dice_coefficient=True,
                        auc=True,
                        mean_iou=True,
                        learning_rate=learning_rate)

    __, kwargs = AdamMock.call_args
    assert kwargs["learning_rate"] == learning_rate

    args, kwargs = model.compile.call_args
    assert kwargs["loss"] == loss
    assert kwargs["optimizer"] == adam_instance

    metrics = kwargs["metrics"]
    assert len(metrics) == 4
    assert metrics[0] == metric_mock
