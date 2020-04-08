# unet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# unet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with unet.  If not, see <http://www.gnu.org/licenses/>.
import logging

import numpy as np

import unet
from unet.datasets import circles

LEARNING_RATE = 1e-3

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
np.random.seed(98765)


def train():
    unet_model = unet.build_model(channels=circles.channels,
                                  num_classes=circles.classes,
                                  layer_depth=3,
                                  filters_root=16)

    unet.finalize_model(unet_model,
                        learning_rate=LEARNING_RATE)

    trainer = unet.Trainer(name="circles",
                           learning_rate_scheduler=unet.SchedulerType.WARMUP_LINEAR_DECAY,
                           warmup_proportion=0.1,
                           learning_rate=LEARNING_RATE)

    train_dataset, validation_dataset, test_dataset = circles.load_data(100, nx=272, ny=272, r_max=20)

    trainer.fit(unet_model,
                train_dataset,
                validation_dataset,
                test_dataset,
                epochs=25,
                batch_size=5)

    return unet_model


if __name__ == '__main__':
    train()
