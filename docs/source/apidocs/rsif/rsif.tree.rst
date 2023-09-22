:py:mod:`rsif.tree`
===================

.. py:module:: rsif.tree

.. autodoc2-docstring:: rsif.tree
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`RandomSimilarityIsolationTree <rsif.tree.RandomSimilarityIsolationTree>`
     - .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree
          :summary:

API
~~~

.. py:class:: RandomSimilarityIsolationTree(distances: typing.List[typing.List[typing.Union[rsif.distance.SelectiveDistance, rsif.distance.TrainDistanceMixin]]], features_span: typing.List[typing.Tuple[int, int]], random_state: typing.Optional[typing.Union[int, numpy.random.RandomState]] = None, max_depth: int = 8, depth: int = 0, pair_strategy='local')
   :canonical: rsif.tree.RandomSimilarityIsolationTree

   .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree

   .. rubric:: Initialization

   .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree.__init__

   .. py:method:: fit(X: numpy.ndarray, y=None) -> rsif.tree.RandomSimilarityIsolationTree
      :canonical: rsif.tree.RandomSimilarityIsolationTree.fit

      .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree.fit

   .. py:method:: prepare_to_fit(X)
      :canonical: rsif.tree.RandomSimilarityIsolationTree.prepare_to_fit

      .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree.prepare_to_fit

   .. py:method:: set_test_distances(distances: typing.List[typing.List[rsif.distance.TestDistanceMixin]])
      :canonical: rsif.tree.RandomSimilarityIsolationTree.set_test_distances

      .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree.set_test_distances

   .. py:method:: select_split_point() -> numpy.ndarray
      :canonical: rsif.tree.RandomSimilarityIsolationTree.select_split_point

      .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree.select_split_point

   .. py:method:: _partition() -> typing.Tuple[numpy.ndarray, numpy.ndarray]
      :canonical: rsif.tree.RandomSimilarityIsolationTree._partition

      .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree._partition

   .. py:method:: _set_leaf()
      :canonical: rsif.tree.RandomSimilarityIsolationTree._set_leaf

      .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree._set_leaf

   .. py:method:: _create_node(samples: numpy.ndarray) -> rsif.tree.RandomSimilarityIsolationTree
      :canonical: rsif.tree.RandomSimilarityIsolationTree._create_node

      .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree._create_node

   .. py:method:: choose_reference_points(selected_objects_indices, selected_distance=None) -> typing.Tuple[numpy.ndarray, numpy.ndarray]
      :canonical: rsif.tree.RandomSimilarityIsolationTree.choose_reference_points

      .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree.choose_reference_points

   .. py:method:: path_lengths_(X: numpy.ndarray) -> numpy.ndarray
      :canonical: rsif.tree.RandomSimilarityIsolationTree.path_lengths_

      .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree.path_lengths_

   .. py:method:: depth_estimate() -> int
      :canonical: rsif.tree.RandomSimilarityIsolationTree.depth_estimate

      .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree.depth_estimate

   .. py:method:: _get_selected_objects(selected_distance: typing.Union[rsif.distance.TrainDistanceMixin, rsif.distance.SelectiveDistance]) -> typing.Union[int, numpy.ndarray, None]
      :canonical: rsif.tree.RandomSimilarityIsolationTree._get_selected_objects

      .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree._get_selected_objects

   .. py:method:: get_leaf_x(x: numpy.ndarray) -> rsif.tree.RandomSimilarityIsolationTree
      :canonical: rsif.tree.RandomSimilarityIsolationTree.get_leaf_x

      .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree.get_leaf_x

   .. py:method:: get_used_points() -> set
      :canonical: rsif.tree.RandomSimilarityIsolationTree.get_used_points

      .. autodoc2-docstring:: rsif.tree.RandomSimilarityIsolationTree.get_used_points
