=============================
Tensorflow Unet
=============================

.. image:: https://readthedocs.org/projects/u-net/badge/?version=latest
        :target: https://u-net.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: http://img.shields.io/badge/arXiv-1609.09077-orange.svg?style=flat
        :target: http://arxiv.org/abs/1609.09077

.. image:: https://img.shields.io/badge/ascl-1611.002-blue.svg?colorB=262255
        :target: http://ascl.net/1611.002

.. image:: https://mybinder.org/badge.svg
        :target: https://mybinder.org/v2/gh/jakeret/unet/master?filepath=notebooks%2Fcicles.ipynb

.. image:: https://img.shields.io/badge/colab-unet-orange.svg?style=flat
        :target: https://colab.research.google.com/drive/1BArjvM_DiPlEfMjVRjlkz4JF2-7movLK



This is a generic **U-Net** implementation as proposed by `Ronneberger et al. <https://arxiv.org/pdf/1505.04597.pdf>`_ developed with **Tensorflow 2** and is a reimplementation of the original `tf_unet <https://github.com/jakeret/tf_unet>`_.

The original code was developed and used for `Radio Frequency Interference mitigation using deep convolutional neural networks <http://arxiv.org/abs/1609.09077>`_ .

The network can be trained to perform image segmentation on arbitrary imaging data. Checkout the `Usage <http://u-net.readthedocs.io/en/latest/usage.html>`_ section or the included Jupyter notebooks for a `toy problem <https://github.com/jakeret/unet/blob/master/notebooks/circles.ipynb>`_ .

The code is not tied to a specific segmentation such that it can be used in a toy problem to detect circles in a noisy image.

.. image:: https://raw.githubusercontent.com/jakeret/unet/master/docs/toy_problem.png
   :alt: Segmentation of a toy problem.
   :align: center

To more complex application such as the detection of radio frequency interference (RFI) in radio astronomy.

.. image:: https://raw.githubusercontent.com/jakeret/unet/master/docs/rfi.png
   :alt: Segmentation of RFI in radio data.
   :align: center

Or to detect galaxies and star in wide field imaging data.

.. image:: https://raw.githubusercontent.com/jakeret/unet/master/docs/galaxies.png
   :alt: Segmentation of a galaxies.
   :align: center


As you use **unet** for your exciting discoveries, please cite the paper that describes the package::


	@article{akeret2017radio,
	  title={Radio frequency interference mitigation using deep convolutional neural networks},
	  author={Akeret, Joel and Chang, Chihway and Lucchi, Aurelien and Refregier, Alexandre},
	  journal={Astronomy and Computing},
	  volume={18},
	  pages={35--39},
	  year={2017},
	  publisher={Elsevier}
	}
