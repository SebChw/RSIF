:py:mod:`risf.risf_data`
========================

.. py:module:: risf.risf_data

.. autodoc2-docstring:: risf.risf_data
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`RisfData <risf.risf_data.RisfData>`
     - .. autodoc2-docstring:: risf.risf_data.RisfData
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`list_to_numpy <risf.risf_data.list_to_numpy>`
     - .. autodoc2-docstring:: risf.risf_data.list_to_numpy
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SUPPORTED_TYPES <risf.risf_data.SUPPORTED_TYPES>`
     - .. autodoc2-docstring:: risf.risf_data.SUPPORTED_TYPES
          :summary:

API
~~~

.. py:function:: list_to_numpy(transformed)
   :canonical: risf.risf_data.list_to_numpy

   .. autodoc2-docstring:: risf.risf_data.list_to_numpy

.. py:data:: SUPPORTED_TYPES
   :canonical: risf.risf_data.SUPPORTED_TYPES
   :value: ((), (), ())

   .. autodoc2-docstring:: risf.risf_data.SUPPORTED_TYPES

.. py:class:: RisfData(num_of_selected_objects: int = None, random_state=23)
   :canonical: risf.risf_data.RisfData

   Bases: :py:obj:`list`

   .. autodoc2-docstring:: risf.risf_data.RisfData

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.risf_data.RisfData.__init__

   .. py:method:: validate_column(X: typing.Union[numpy.ndarray, list, pandas.Series]) -> numpy.ndarray
      :canonical: risf.risf_data.RisfData.validate_column
      :classmethod:

      .. autodoc2-docstring:: risf.risf_data.RisfData.validate_column

   .. py:method:: distance_check(X: numpy.ndarray, dist: typing.Union[risf.distance.DistanceMixin, risf.distance.SelectiveDistance])
      :canonical: risf.risf_data.RisfData.distance_check
      :staticmethod:

      .. autodoc2-docstring:: risf.risf_data.RisfData.distance_check

   .. py:method:: update_metadata(name: typing.Optional[str] = None)
      :canonical: risf.risf_data.RisfData.update_metadata

      .. autodoc2-docstring:: risf.risf_data.RisfData.update_metadata

   .. py:method:: shape_check(X: numpy.ndarray)
      :canonical: risf.risf_data.RisfData.shape_check

      .. autodoc2-docstring:: risf.risf_data.RisfData.shape_check

   .. py:method:: add_distances(X: numpy.ndarray, distances: typing.List[typing.Union[risf.distance.TrainDistanceMixin, risf.distance.SelectiveDistance, str]])
      :canonical: risf.risf_data.RisfData.add_distances

      .. autodoc2-docstring:: risf.risf_data.RisfData.add_distances

   .. py:method:: add_data(X: numpy.ndarray, dist: typing.List[typing.Union[risf.distance.SelectiveDistance, risf.distance.DistanceMixin, str]], name: typing.Optional[str] = None)
      :canonical: risf.risf_data.RisfData.add_data

      .. autodoc2-docstring:: risf.risf_data.RisfData.add_data

   .. py:method:: precompute_distances(n_jobs=1, train_data: typing.Optional[risf.risf_data.RisfData] = None, selected_objects: typing.Optional[numpy.ndarray] = None)
      :canonical: risf.risf_data.RisfData.precompute_distances

      .. autodoc2-docstring:: risf.risf_data.RisfData.precompute_distances

   .. py:method:: impute_missing_values(distance, strategy='max')
      :canonical: risf.risf_data.RisfData.impute_missing_values

      .. autodoc2-docstring:: risf.risf_data.RisfData.impute_missing_values

   .. py:method:: transform(list_of_X: typing.List[numpy.ndarray], forest, n_jobs=1, precomputed_distances: typing.Optional[typing.List[risf.distance.TestDistanceMixin]] = None) -> risf.risf_data.RisfData
      :canonical: risf.risf_data.RisfData.transform

      .. autodoc2-docstring:: risf.risf_data.RisfData.transform
