=============================
Tensorflow Unet
=============================

.. image:: https://readthedocs.org/projects/u-net/badge/?version=latest
        :target: https://u-net.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://travis-ci.com/jakeret/unet.svg?branch=master
    :target: https://travis-ci.com/jakeret/unet

.. image:: http://img.shields.io/badge/arXiv-1609.09077-orange.svg?style=flat
        :target: http://arxiv.org/abs/1609.09077

.. image:: https://camo.githubusercontent.com/c8e5db7a5d15b0e7c13480a0ed81db1ae2128b80/68747470733a2f2f62696e6465722e70616e67656f2e696f2f62616467655f6c6f676f2e737667
        :target: https://mybinder.org/v2/gh/jakeret/unet/master?filepath=notebooks%2Fcicles.ipynb

.. image:: https://camo.githubusercontent.com/52feade06f2fecbf006889a904d221e6a730c194/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667
        :target: https://colab.research.google.com/drive/1laPoOaGcqEBB3jTvb-pGnmDU21zwtgJB

This is a generic **U-Net** implementation as proposed by `Ronneberger et al. <https://arxiv.org/pdf/1505.04597.pdf>`_ developed with **Tensorflow 2**. This project is a reimplementation of the original `tf_unet <https://github.com/jakeret/tf_unet>`_.

Originally, the code was developed and used for `Radio Frequency Interference mitigation using deep convolutional neural networks <http://arxiv.org/abs/1609.09077>`_ .

The network can be trained to perform image segmentation on arbitrary imaging data. Checkout the `Usage <http://u-net.readthedocs.io/en/latest/usage.html>`_ section, the included `Jupyter notebooks <https://github.com/jakeret/unet/blob/master/notebooks/circles.ipynb>`_  or `on Google Colab <https://colab.research.google.com/drive/1BArjvM_DiPlEfMjVRjlkz4JF2-7movLK>`_ for a toy problem or the Oxford Pet Segmentation example available on `Google Colab <https://colab.research.google.com/drive/1laPoOaGcqEBB3jTvb-pGnmDU21zwtgJB>`_.

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


The architectural elements of a U-Net consist of a contracting and expanding path:

.. image:: https://raw.githubusercontent.com/jakeret/unet/master/docs/unet.png
   :alt: Unet architecture.
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
