Time Evolution Algorithms
*************************

In this tutorial we will introduce the time evolution algorithms available in Renormalizer.
For more details please refer to the literatures at the bottom of the package.

Renormalizer by default uses the :ref:`pc_label`
time evolution method. The method requires relatively less knowledge and experience with TD-DMRG.
However, it's accuracy and speed is not satisfactory compared to the state-of-the-art TDVP methods.

In the following we summarize the features of each time evolution algorithms in a table,
and more details are explained in the subsequent sections.


.. list-table::
    :header-rows: 1

    * - Scheme
      - Accuracy
      - Speed
      - Adaptive Bond Dimension
      - Adaptive Step Size
    * - TDVP-PS2
      - High
      - Low
      - Yes
      - Limited

.. _pc_label:

Propagation and Compression (P&C)
=================================


Time Dependent Variational Principle (TDVP)
===========================================

Projector Splitting
-------------------

Variable Mean Field
-------------------

Constant Mean Field
-------------------

Matrix Unfolding
----------------


Bibtex entry::

    @article{li2020numerical,
      title={Numerical assessment for accuracy and GPU acceleration of TD-DMRG time evolution schemes},
      author={Li, Weitang and Ren, Jiajun and Shuai, Zhigang},
      journal={The Journal of Chemical Physics},
      volume={152},
      number={2},
      pages={024127},
      year={2020},
      publisher={AIP Publishing LLC}
    }

..
    Add WIRES paper and maybe the chinese cjcu paper.