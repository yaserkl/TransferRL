
********************
TransferRL
********************

.. image:: https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat
    :target: https://github.com/yaserkl/TransferRL/pulls
.. image:: https://img.shields.io/badge/Made%20with-Python-1f425f.svg
      :target: https://www.python.org/
.. image:: https://img.shields.io/pypi/l/ansicolortags.svg
      :target: https://github.com/yaserkl/TransferRL/blob/master/LICENSE.txt
.. image:: https://img.shields.io/github/contributors/Naereen/StrapDown.js.svg
      :target: https://github.com/yaserkl/TransferRL/graphs/contributors
.. image:: https://img.shields.io/github/issues/Naereen/StrapDown.js.svg
      :target: https://github.com/yaserkl/TransferRL/issues
.. image:: https://img.shields.io/badge/arXiv-1805.09461-red.svg?style=flat
   :target: https://arxiv.org/abs/

This repository contains the code developed in TensorFlow_ for the following paper submitted to SIAM SDM 2019:


| `Deep Transfer Reinforcement Learning for Text Summarization`_,
| by: `Yaser Keneshloo`_, `Naren Ramakrishnan`_, and `Chandan K. Reddy`_


.. _Deep Transfer Reinforcement Learning for Text Summarization: https://arxiv.org/abs/
.. _TensorFlow: https://www.tensorflow.org/
.. _Yaser Keneshloo: https://github.com/yaserkl
.. _Naren Ramakrishnan: http://people.cs.vt.edu/naren/
.. _Chandan K. Reddy: http://people.cs.vt.edu/~reddy/


If you used this code, please kindly consider citing the following paper:

.. code:: shell

    @article{keneshloo2018transferrl,
     title={Deep Transfer Reinforcement Learning for Text Summarization},
     author={Keneshloo, Yaser and Ramakrishnan, Naren and Reddy, Chandan K.},
     journal={arXiv preprint arXiv:},
     year={2018}
    }



#################
Table of Contents
#################
.. contents::
  :local:
  :depth: 3


..  Chapter 1 Title
..  ===============

..  Section 1.1 Title
..  -----------------

..  Subsection 1.1.1 Title
..  ~~~~~~~~~~~~~~~~~~~~~~

.. image:: docs/_img/transferrl.png
    :target: docs/_img/transferrl.png

============
Motivation
============

Deep neural networks are data hungry models and thus they face difficulties when used for training on small size data. Transfer learning is a method that could potentially help in such situations. Although transfer learning achieved great success in image processing, its effect in the text domain is yet to be well established especially due to several intricacies that arise in the context of document analysis and understanding. In this paper, we study the problem of transfer learning for text summarization and discuss why the existing state-of-the-art models for this problem fail to generalize well on other (unseen) datasets. We propose a reinforcement learning framework based on self-critic policy gradient method which solves this problem and achieves good generalization and state-of-the-art results on a variety of datasets. Through an extensive set of experiments, we also show the ability of our proposed framework in fine-tuning the text summarization model only with a few training samples. To the best of our knowledge, this is first work that studies transfer learning in text summarization and provides a generic solution that works well on unseen data.

---------------------------------------------------------------------------

====================
Requirements
====================
-------------
Python
-------------
  - Use Python 2.7

Python requirements can be installed as follows:

.. code:: bash

    pip install -r python_requirements.txt

-------------
TensorFlow
-------------

  - Use Tensorflow 1.10 or newer

-------------
GPU
-------------

  - CUDA 8 or 9
  - CUDNN 6 or 7

---------------------------------------------------------------------------

============
DATASET
============
----------------------
CNN/Daily Mail dataset
----------------------
https://github.com/abisee/cnn-dailymail

----------------------
Newsroom dataset
----------------------
https://summari.es/

We have provided helper codes to download the cnn-dailymail dataset and
pre-process this dataset and newsroom dataset.
Please refer to `this link <src/helper>`_ to access them.

We saw a large improvement on the ROUGE measure by using our processed version of these datasets
in the summarization results, therefore, we strongly suggest using these pre-processed files for
all the training.

---------------------------------------------------------------------------

====================
Running Experiments
====================

To train our best performing model, please check the following `file <src/helper/commands.txt>`_:

