========
Usage
========

To use Tensorflow Unet in a project::

    import unet
    from unet.datasets import circles

    #loading the datasets
    train_dataset, validation_dataset = circles.load_data(100, nx=200, ny=200,
                                                          splits=(0.8, 0.2))

    #building the model
    unet_model = unet.build_model(channels=circles.channels,
                              num_classes=circles.classes,
                              layer_depth=3,
                              filters_root=16)

    unet.finalize_model(unet_model)

    #training and validating the model
    trainer = unet.Trainer(checkpoint_callback=False)
    trainer.fit(unet_model,
                train_dataset,
                validation_dataset,
                epochs=5,
                batch_size=1)


Keep track of the learning progress using *Tensorboard*. **unet** automatically outputs relevant summaries.

.. image:: https://raw.githubusercontent.com/jakeret/unet/master/docs/stats.png
   :alt: Segmentation of a toy problem.
   :align: center

