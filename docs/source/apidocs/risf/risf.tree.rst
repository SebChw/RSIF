:py:mod:`risf.tree`
===================

.. py:module:: risf.tree

.. autodoc2-docstring:: risf.tree
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`RandomIsolationSimilarityTree <risf.tree.RandomIsolationSimilarityTree>`
     - .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree
          :summary:

API
~~~

.. py:class:: RandomIsolationSimilarityTree(distances: typing.List[typing.List[typing.Union[risf.distance.SelectiveDistance, risf.distance.TrainDistanceMixin]]], features_span: typing.List[typing.Tuple[int, int]], random_state: typing.Union[int, numpy.random.RandomState], max_depth: int = 8, depth: int = 0)
   :canonical: risf.tree.RandomIsolationSimilarityTree

   .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.__init__

   .. py:method:: fit(X: numpy.ndarray, y=None) -> risf.tree.RandomIsolationSimilarityTree
      :canonical: risf.tree.RandomIsolationSimilarityTree.fit

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.fit

   .. py:method:: prepare_to_fit(X)
      :canonical: risf.tree.RandomIsolationSimilarityTree.prepare_to_fit

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.prepare_to_fit

   .. py:method:: set_test_distances(distances: typing.List[typing.List[risf.distance.TestDistanceMixin]])
      :canonical: risf.tree.RandomIsolationSimilarityTree.set_test_distances

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.set_test_distances

   .. py:method:: select_split_point() -> numpy.ndarray
      :canonical: risf.tree.RandomIsolationSimilarityTree.select_split_point

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.select_split_point

   .. py:method:: _partition() -> typing.Tuple[numpy.ndarray, numpy.ndarray]
      :canonical: risf.tree.RandomIsolationSimilarityTree._partition

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree._partition

   .. py:method:: _set_leaf()
      :canonical: risf.tree.RandomIsolationSimilarityTree._set_leaf

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree._set_leaf

   .. py:method:: _create_node(samples: numpy.ndarray) -> risf.tree.RandomIsolationSimilarityTree
      :canonical: risf.tree.RandomIsolationSimilarityTree._create_node

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree._create_node

   .. py:method:: choose_reference_points(selected_objects) -> typing.Tuple[numpy.ndarray, numpy.ndarray]
      :canonical: risf.tree.RandomIsolationSimilarityTree.choose_reference_points

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.choose_reference_points

   .. py:method:: path_lengths_(X: numpy.ndarray) -> numpy.ndarray
      :canonical: risf.tree.RandomIsolationSimilarityTree.path_lengths_

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.path_lengths_

   .. py:method:: depth_estimate() -> int
      :canonical: risf.tree.RandomIsolationSimilarityTree.depth_estimate

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.depth_estimate

   .. py:method:: _get_selected_objects(selected_distance: typing.Union[risf.distance.TrainDistanceMixin, risf.distance.SelectiveDistance]) -> typing.Union[int, numpy.ndarray, None]
      :canonical: risf.tree.RandomIsolationSimilarityTree._get_selected_objects

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree._get_selected_objects

   .. py:method:: get_leaf_x(x: numpy.ndarray) -> risf.tree.RandomIsolationSimilarityTree
      :canonical: risf.tree.RandomIsolationSimilarityTree.get_leaf_x

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.get_leaf_x

   .. py:method:: get_used_points() -> set
      :canonical: risf.tree.RandomIsolationSimilarityTree.get_used_points

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.get_used_points
