:py:mod:`risf.distance`
=======================

.. py:module:: risf.distance

.. autodoc2-docstring:: risf.distance
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SelectiveDistance <risf.distance.SelectiveDistance>`
     - .. autodoc2-docstring:: risf.distance.SelectiveDistance
          :summary:
   * - :py:obj:`DistanceMixin <risf.distance.DistanceMixin>`
     - .. autodoc2-docstring:: risf.distance.DistanceMixin
          :summary:
   * - :py:obj:`TrainDistanceMixin <risf.distance.TrainDistanceMixin>`
     -
   * - :py:obj:`TestDistanceMixin <risf.distance.TestDistanceMixin>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_parallel_on_array <risf.distance._parallel_on_array>`
     - .. autodoc2-docstring:: risf.distance._parallel_on_array
          :summary:
   * - :py:obj:`split_distance_mixin <risf.distance.split_distance_mixin>`
     - .. autodoc2-docstring:: risf.distance.split_distance_mixin
          :summary:

API
~~~

.. py:class:: SelectiveDistance(projection_func: typing.Callable, min_n: int, max_n: int)
   :canonical: risf.distance.SelectiveDistance

   .. autodoc2-docstring:: risf.distance.SelectiveDistance

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance.SelectiveDistance.__init__

   .. py:method:: project(X: numpy.ndarray, Op: numpy.ndarray, Oq: numpy.ndarray, tree, random_instance: numpy.random.RandomState) -> numpy.ndarray
      :canonical: risf.distance.SelectiveDistance.project

      .. autodoc2-docstring:: risf.distance.SelectiveDistance.project

.. py:class:: DistanceMixin(distance: typing.Callable, selected_objects: typing.Optional[numpy.ndarray] = None)
   :canonical: risf.distance.DistanceMixin

   Bases: :py:obj:`abc.ABC`

   .. autodoc2-docstring:: risf.distance.DistanceMixin

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance.DistanceMixin.__init__

   .. py:method:: project(id_x: numpy.ndarray, id_p: numpy.ndarray, id_q: numpy.ndarray, tree=None, random_instance=None) -> numpy.ndarray
      :canonical: risf.distance.DistanceMixin.project

      .. autodoc2-docstring:: risf.distance.DistanceMixin.project

   .. py:method:: _generate_indices_splits(pairs_of_indices: numpy.ndarray, n_jobs: int) -> typing.List[typing.Tuple[int, int]]
      :canonical: risf.distance.DistanceMixin._generate_indices_splits

      .. autodoc2-docstring:: risf.distance.DistanceMixin._generate_indices_splits

   .. py:method:: precompute_distances(X: numpy.ndarray, X_test: typing.Optional[numpy.ndarray] = None, n_jobs: int = 1, prefer: typing.Optional[str] = None)
      :canonical: risf.distance.DistanceMixin.precompute_distances

      .. autodoc2-docstring:: risf.distance.DistanceMixin.precompute_distances

   .. py:method:: _generate_indices(num_train_objects: int, num_test_objects: int) -> numpy.ndarray
      :canonical: risf.distance.DistanceMixin._generate_indices
      :abstractmethod:

      .. autodoc2-docstring:: risf.distance.DistanceMixin._generate_indices

   .. py:method:: _assign_to_distance_matrix(rows_id: numpy.ndarray, cols_id: numpy.ndarray, concatenated_distances: numpy.ndarray)
      :canonical: risf.distance.DistanceMixin._assign_to_distance_matrix
      :abstractmethod:

      .. autodoc2-docstring:: risf.distance.DistanceMixin._assign_to_distance_matrix

.. py:function:: _parallel_on_array(indices: numpy.ndarray, X1: numpy.ndarray, X2: numpy.ndarray, function: typing.Callable) -> typing.List[float]
   :canonical: risf.distance._parallel_on_array

   .. autodoc2-docstring:: risf.distance._parallel_on_array

.. py:class:: TrainDistanceMixin(distance: typing.Callable, selected_objects: typing.Optional[numpy.ndarray] = None)
   :canonical: risf.distance.TrainDistanceMixin

   Bases: :py:obj:`risf.distance.DistanceMixin`

   .. py:method:: _generate_indices(num_train_objects: int, num_test_objects=None) -> numpy.ndarray
      :canonical: risf.distance.TrainDistanceMixin._generate_indices

   .. py:method:: _assign_to_distance_matrix(row_ids: numpy.ndarray, col_ids: numpy.ndarray, concatenated_distances: numpy.ndarray)
      :canonical: risf.distance.TrainDistanceMixin._assign_to_distance_matrix

.. py:class:: TestDistanceMixin(distance: typing.Callable, selected_objects)
   :canonical: risf.distance.TestDistanceMixin

   Bases: :py:obj:`risf.distance.DistanceMixin`

   .. py:attribute:: __test__
      :canonical: risf.distance.TestDistanceMixin.__test__
      :value: False

      .. autodoc2-docstring:: risf.distance.TestDistanceMixin.__test__

   .. py:method:: _generate_indices(num_train_objects: int, num_test_objects: int) -> numpy.ndarray
      :canonical: risf.distance.TestDistanceMixin._generate_indices

   .. py:method:: _assign_to_distance_matrix(row_ids: numpy.ndarray, col_ids: numpy.ndarray, concatenated_distances: numpy.ndarray)
      :canonical: risf.distance.TestDistanceMixin._assign_to_distance_matrix

      .. autodoc2-docstring:: risf.distance.TestDistanceMixin._assign_to_distance_matrix

.. py:function:: split_distance_mixin(distance_mixin: risf.distance.TrainDistanceMixin, train_indices: numpy.ndarray) -> typing.Tuple[risf.distance.TrainDistanceMixin, risf.distance.TestDistanceMixin]
   :canonical: risf.distance.split_distance_mixin

   .. autodoc2-docstring:: risf.distance.split_distance_mixin
