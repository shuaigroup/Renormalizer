Model
*************************************

Use the :class:`~renormalizer.model.Model` classes to define the system to be calculated.
We provide the Holstein model :class:`renormalizer.model.HolsteinModel`
and the spin-boson model :class:`~renormalizer.model.SpinBosonModel` out of box, yet any arbitrary model
with sum-of-product Hamiltonian
can be constructed by using the most general :class:`~renormalizer.model.Model` class.
These classes require :class:`~renormalizer.model.basis.BasisSet` and :class:`~renormalizer.model.Op`
as input.

The Model classes
=================

.. autoclass:: renormalizer.model.Model
    :members:
    :inherited-members:


.. autoclass:: renormalizer.model.HolsteinModel
    :members:
    :inherited-members:


.. autoclass:: renormalizer.model.SpinBosonModel
    :members:
    :inherited-members:


.. autoclass:: renormalizer.model.TI1DModel
    :members:
    :inherited-members:


Basis Functions
===============

.. autoclass:: renormalizer.model.basis.BasisSet
    :members:
    :inherited-members:


.. autoclass:: renormalizer.model.basis.BasisSHO
    :members:
    :inherited-members:

.. autoclass:: renormalizer.model.basis.BasisHopsBoson
    :members:
    :inherited-members:

.. autoclass:: renormalizer.model.basis.BasisSineDVR
    :members:
    :inherited-members:

.. autoclass:: renormalizer.model.basis.BasisSimpleElectron
    :members:
    :inherited-members:

.. autoclass:: renormalizer.model.basis.BasisMultiElectron
    :members:
    :inherited-members:

.. autoclass:: renormalizer.model.basis.BasisMultiElectronVac
    :members:
    :inherited-members:

.. autoclass:: renormalizer.model.basis.BasisHalfSpin
    :members:
    :inherited-members:


The Operator Class
==================

.. autoclass:: renormalizer.model.Op
    :members:
    :inherited-members:
