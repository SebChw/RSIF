:py:mod:`risf.forest`
=====================

.. py:module:: risf.forest

.. autodoc2-docstring:: risf.forest
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`RandomIsolationSimilarityForest <risf.forest.RandomIsolationSimilarityForest>`
     - .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_build_tree <risf.forest._build_tree>`
     - .. autodoc2-docstring:: risf.forest._build_tree
          :summary:

API
~~~

.. py:class:: RandomIsolationSimilarityForest(distances=[SelectiveDistance(euclidean_projection, 1, 1)], n_estimators=100, max_samples='auto', contamination='auto', max_depth=8, bootstrap=False, n_jobs=None, random_state=23, verbose=0)
   :canonical: risf.forest.RandomIsolationSimilarityForest

   Bases: :py:obj:`sklearn.base.BaseEstimator`, :py:obj:`sklearn.base.OutlierMixin`

   .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.__init__

   .. py:method:: fit(X: numpy.array, y=None)
      :canonical: risf.forest.RandomIsolationSimilarityForest.fit

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.fit

   .. py:method:: prepare_to_fit(X)
      :canonical: risf.forest.RandomIsolationSimilarityForest.prepare_to_fit

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.prepare_to_fit

   .. py:method:: create_trees()
      :canonical: risf.forest.RandomIsolationSimilarityForest.create_trees

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.create_trees

   .. py:method:: set_offset(y=None)
      :canonical: risf.forest.RandomIsolationSimilarityForest.set_offset

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.set_offset

   .. py:method:: calculate_mean_path_lengths(X: numpy.array)
      :canonical: risf.forest.RandomIsolationSimilarityForest.calculate_mean_path_lengths

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.calculate_mean_path_lengths

   .. py:method:: score_samples(X: numpy.array)
      :canonical: risf.forest.RandomIsolationSimilarityForest.score_samples

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.score_samples

   .. py:method:: decision_function(X: numpy.array)
      :canonical: risf.forest.RandomIsolationSimilarityForest.decision_function

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.decision_function

   .. py:method:: predict_train(return_raw_scores=False)
      :canonical: risf.forest.RandomIsolationSimilarityForest.predict_train

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.predict_train

   .. py:method:: predict(X: numpy.array, return_raw_scores=False)
      :canonical: risf.forest.RandomIsolationSimilarityForest.predict

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.predict

   .. py:method:: get_used_points()
      :canonical: risf.forest.RandomIsolationSimilarityForest.get_used_points

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.get_used_points

.. py:function:: _build_tree(tree: risf.tree.RandomIsolationSimilarityTree, X: numpy.array, tree_idx: int, n_trees: int, subsample_size: int = 256, verbose: int = 0, bootstrap: bool = False)
   :canonical: risf.forest._build_tree

   .. autodoc2-docstring:: risf.forest._build_tree
