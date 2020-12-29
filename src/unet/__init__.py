# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

from unet import metrics
from unet.schedulers import SchedulerType
from unet.trainer import Trainer
from unet.unet import build_model, finalize_model

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

__all__ = [build_model,
           finalize_model,
           SchedulerType,
           Trainer]


custom_objects = {'mean_iou': metrics.mean_iou,
                  'dice_coefficient': metrics.dice_coefficient}