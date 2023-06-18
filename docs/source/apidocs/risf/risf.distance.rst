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
     -
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

.. py:class:: SelectiveDistance(projection_func: callable, min_n: int, max_n: int)
   :canonical: risf.distance.SelectiveDistance

   .. autodoc2-docstring:: risf.distance.SelectiveDistance

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance.SelectiveDistance.__init__

   .. py:method:: project(X, Op, Oq, tree, random_instance)
      :canonical: risf.distance.SelectiveDistance.project

      .. autodoc2-docstring:: risf.distance.SelectiveDistance.project

.. py:class:: DistanceMixin(distance: callable, selected_objects=None)
   :canonical: risf.distance.DistanceMixin

   Bases: :py:obj:`abc.ABC`

   .. py:method:: project(id_x, id_p, id_q, tree=None, random_instance=None)
      :canonical: risf.distance.DistanceMixin.project

      .. autodoc2-docstring:: risf.distance.DistanceMixin.project

   .. py:method:: _generate_indices_splits(pairs_of_indices, n_jobs)
      :canonical: risf.distance.DistanceMixin._generate_indices_splits

      .. autodoc2-docstring:: risf.distance.DistanceMixin._generate_indices_splits

   .. py:method:: precompute_distances(X, X_test=None, n_jobs=1, prefer=None)
      :canonical: risf.distance.DistanceMixin.precompute_distances

      .. autodoc2-docstring:: risf.distance.DistanceMixin.precompute_distances

   .. py:method:: _generate_indices(num_train_objects, num_test_objects)
      :canonical: risf.distance.DistanceMixin._generate_indices
      :abstractmethod:

      .. autodoc2-docstring:: risf.distance.DistanceMixin._generate_indices

   .. py:method:: _assign_to_distance_matrix(rows_id, cols_id, concatenated_distances)
      :canonical: risf.distance.DistanceMixin._assign_to_distance_matrix
      :abstractmethod:

      .. autodoc2-docstring:: risf.distance.DistanceMixin._assign_to_distance_matrix

.. py:function:: _parallel_on_array(indices, X1, X2, function)
   :canonical: risf.distance._parallel_on_array

   .. autodoc2-docstring:: risf.distance._parallel_on_array

.. py:class:: TrainDistanceMixin(distance: callable, selected_objects=None)
   :canonical: risf.distance.TrainDistanceMixin

   Bases: :py:obj:`risf.distance.DistanceMixin`

   .. py:method:: _generate_indices(num_train_objects, num_test_objects=None)
      :canonical: risf.distance.TrainDistanceMixin._generate_indices

      .. autodoc2-docstring:: risf.distance.TrainDistanceMixin._generate_indices

   .. py:method:: _assign_to_distance_matrix(row_ids, col_ids, concatenated_distances)
      :canonical: risf.distance.TrainDistanceMixin._assign_to_distance_matrix

      .. autodoc2-docstring:: risf.distance.TrainDistanceMixin._assign_to_distance_matrix

.. py:class:: TestDistanceMixin(distance: callable, selected_objects)
   :canonical: risf.distance.TestDistanceMixin

   Bases: :py:obj:`risf.distance.DistanceMixin`

   .. py:attribute:: __test__
      :canonical: risf.distance.TestDistanceMixin.__test__
      :value: False

      .. autodoc2-docstring:: risf.distance.TestDistanceMixin.__test__

   .. py:method:: _generate_indices(num_train_objects, num_test_objects)
      :canonical: risf.distance.TestDistanceMixin._generate_indices

      .. autodoc2-docstring:: risf.distance.TestDistanceMixin._generate_indices

   .. py:method:: _assign_to_distance_matrix(row_ids, col_ids, concatenated_distances)
      :canonical: risf.distance.TestDistanceMixin._assign_to_distance_matrix

      .. autodoc2-docstring:: risf.distance.TestDistanceMixin._assign_to_distance_matrix

.. py:function:: split_distance_mixin(distance_mixin: risf.distance.TrainDistanceMixin, train_indices: numpy.ndarray) -> typing.Tuple[risf.distance.TrainDistanceMixin, risf.distance.TestDistanceMixin]
   :canonical: risf.distance.split_distance_mixin

   .. autodoc2-docstring:: risf.distance.split_distance_mixin
