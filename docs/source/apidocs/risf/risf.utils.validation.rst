:py:mod:`risf.utils.validation`
===============================

.. py:module:: risf.utils.validation

.. autodoc2-docstring:: risf.utils.validation
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`prepare_X <risf.utils.validation.prepare_X>`
     - .. autodoc2-docstring:: risf.utils.validation.prepare_X
          :summary:
   * - :py:obj:`check_max_samples <risf.utils.validation.check_max_samples>`
     - .. autodoc2-docstring:: risf.utils.validation.check_max_samples
          :summary:
   * - :py:obj:`check_random_state <risf.utils.validation.check_random_state>`
     - .. autodoc2-docstring:: risf.utils.validation.check_random_state
          :summary:
   * - :py:obj:`check_distance <risf.utils.validation.check_distance>`
     - .. autodoc2-docstring:: risf.utils.validation.check_distance
          :summary:

API
~~~

.. py:function:: prepare_X(X: typing.Union[risf.risf_data.RisfData, pandas.DataFrame, list, numpy.ndarray]) -> typing.Tuple[numpy.ndarray, typing.List[typing.Tuple[int, int]]]
   :canonical: risf.utils.validation.prepare_X

   .. autodoc2-docstring:: risf.utils.validation.prepare_X

.. py:function:: check_max_samples(max_samples: typing.Union[str, float, int], X: numpy.ndarray) -> int
   :canonical: risf.utils.validation.check_max_samples

   .. autodoc2-docstring:: risf.utils.validation.check_max_samples

.. py:function:: check_random_state(random_state)
   :canonical: risf.utils.validation.check_random_state

   .. autodoc2-docstring:: risf.utils.validation.check_random_state

.. py:function:: check_distance(distance, n_features)
   :canonical: risf.utils.validation.check_distance

   .. autodoc2-docstring:: risf.utils.validation.check_distance
