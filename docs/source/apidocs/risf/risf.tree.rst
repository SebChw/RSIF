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

.. py:class:: RandomIsolationSimilarityTree(distances, features_span, random_state=None, max_depth=8, depth=0)
   :canonical: risf.tree.RandomIsolationSimilarityTree

   .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.__init__

   .. py:method:: fit(X, y=None)
      :canonical: risf.tree.RandomIsolationSimilarityTree.fit

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.fit

   .. py:method:: prepare_to_fit(X)
      :canonical: risf.tree.RandomIsolationSimilarityTree.prepare_to_fit

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.prepare_to_fit

   .. py:method:: set_test_distances(distances)
      :canonical: risf.tree.RandomIsolationSimilarityTree.set_test_distances

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.set_test_distances

   .. py:method:: select_split_point()
      :canonical: risf.tree.RandomIsolationSimilarityTree.select_split_point

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.select_split_point

   .. py:method:: _partition()
      :canonical: risf.tree.RandomIsolationSimilarityTree._partition

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree._partition

   .. py:method:: _set_leaf()
      :canonical: risf.tree.RandomIsolationSimilarityTree._set_leaf

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree._set_leaf

   .. py:method:: _create_node(samples)
      :canonical: risf.tree.RandomIsolationSimilarityTree._create_node

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree._create_node

   .. py:method:: choose_reference_points(selected_objects)
      :canonical: risf.tree.RandomIsolationSimilarityTree.choose_reference_points

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.choose_reference_points

   .. py:method:: path_lengths_(X)
      :canonical: risf.tree.RandomIsolationSimilarityTree.path_lengths_

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.path_lengths_

   .. py:method:: depth_estimate()
      :canonical: risf.tree.RandomIsolationSimilarityTree.depth_estimate

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.depth_estimate

   .. py:method:: _get_selected_objects(selected_distance)
      :canonical: risf.tree.RandomIsolationSimilarityTree._get_selected_objects

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree._get_selected_objects

   .. py:method:: get_leaf_x(x)
      :canonical: risf.tree.RandomIsolationSimilarityTree.get_leaf_x

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.get_leaf_x

   .. py:method:: get_used_points()
      :canonical: risf.tree.RandomIsolationSimilarityTree.get_used_points

      .. autodoc2-docstring:: risf.tree.RandomIsolationSimilarityTree.get_used_points
