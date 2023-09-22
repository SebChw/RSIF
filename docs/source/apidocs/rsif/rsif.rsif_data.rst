:py:mod:`rsif.rsif_data`
========================

.. py:module:: rsif.rsif_data

.. autodoc2-docstring:: rsif.rsif_data
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`RsifData <rsif.rsif_data.RsifData>`
     - .. autodoc2-docstring:: rsif.rsif_data.RsifData
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`list_to_numpy <rsif.rsif_data.list_to_numpy>`
     - .. autodoc2-docstring:: rsif.rsif_data.list_to_numpy
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SUPPORTED_TYPES <rsif.rsif_data.SUPPORTED_TYPES>`
     - .. autodoc2-docstring:: rsif.rsif_data.SUPPORTED_TYPES
          :summary:

API
~~~

.. py:function:: list_to_numpy(transformed)
   :canonical: rsif.rsif_data.list_to_numpy

   .. autodoc2-docstring:: rsif.rsif_data.list_to_numpy

.. py:data:: SUPPORTED_TYPES
   :canonical: rsif.rsif_data.SUPPORTED_TYPES
   :value: ((), (), ())

   .. autodoc2-docstring:: rsif.rsif_data.SUPPORTED_TYPES

.. py:class:: RsifData(num_of_selected_objects: int = None, random_state=23)
   :canonical: rsif.rsif_data.RsifData

   Bases: :py:obj:`list`

   .. autodoc2-docstring:: rsif.rsif_data.RsifData

   .. rubric:: Initialization

   .. autodoc2-docstring:: rsif.rsif_data.RsifData.__init__

   .. py:method:: validate_column(X: typing.Union[numpy.ndarray, list, pandas.Series]) -> numpy.ndarray
      :canonical: rsif.rsif_data.RsifData.validate_column
      :classmethod:

      .. autodoc2-docstring:: rsif.rsif_data.RsifData.validate_column

   .. py:method:: distance_check(X: numpy.ndarray, dist: typing.Union[rsif.distance.DistanceMixin, rsif.distance.SelectiveDistance])
      :canonical: rsif.rsif_data.RsifData.distance_check
      :staticmethod:

      .. autodoc2-docstring:: rsif.rsif_data.RsifData.distance_check

   .. py:method:: update_metadata(name: typing.Optional[str] = None)
      :canonical: rsif.rsif_data.RsifData.update_metadata

      .. autodoc2-docstring:: rsif.rsif_data.RsifData.update_metadata

   .. py:method:: shape_check(X: numpy.ndarray)
      :canonical: rsif.rsif_data.RsifData.shape_check

      .. autodoc2-docstring:: rsif.rsif_data.RsifData.shape_check

   .. py:method:: add_distances(X: numpy.ndarray, distances: typing.List[typing.Union[rsif.distance.TrainDistanceMixin, rsif.distance.SelectiveDistance, str]]) -> bool
      :canonical: rsif.rsif_data.RsifData.add_distances

      .. autodoc2-docstring:: rsif.rsif_data.RsifData.add_distances

   .. py:method:: add_data(X: numpy.ndarray, dist: typing.List[typing.Union[rsif.distance.SelectiveDistance, rsif.distance.DistanceMixin, str]], name: typing.Optional[str] = None)
      :canonical: rsif.rsif_data.RsifData.add_data

      .. autodoc2-docstring:: rsif.rsif_data.RsifData.add_data

   .. py:method:: precompute_distances(n_jobs=1, train_data: typing.Optional[rsif.rsif_data.RsifData] = None, selected_objects: typing.Optional[numpy.ndarray] = None)
      :canonical: rsif.rsif_data.RsifData.precompute_distances

      .. autodoc2-docstring:: rsif.rsif_data.RsifData.precompute_distances

   .. py:method:: impute_missing_values(distance, strategy='max')
      :canonical: rsif.rsif_data.RsifData.impute_missing_values

      .. autodoc2-docstring:: rsif.rsif_data.RsifData.impute_missing_values

   .. py:method:: transform(list_of_X: typing.List[numpy.ndarray], forest, n_jobs=1, precomputed_distances: typing.Optional[typing.List[rsif.distance.TestDistanceMixin]] = None) -> rsif.rsif_data.RsifData
      :canonical: rsif.rsif_data.RsifData.transform

      .. autodoc2-docstring:: rsif.rsif_data.RsifData.transform
