Frequently Asked Questions
**************************

Why so slow?
============
TBA

How to build docs and view them locally?
========================================

In the ``doc`` directory where ``Makefile`` is located, run

.. code-block::

    make html

This will build the document.
Then, in the ``html`` directory, run

.. code-block::

    python -m http.server

This will setup a HTTP server on the 8000 port.
You can visit the document via ``http://localhost:8000``
if the document is built in local host.
Or you may need to change ``localhost`` to the actual IP of the host where the doc is built.

