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

.. py:class:: RandomIsolationSimilarityForest(distances: typing.List[typing.List[typing.Union[risf.distance.SelectiveDistance, risf.distance.TrainDistanceMixin]]] = [[SelectiveDistance(euclidean_projection, 1, 1)]], n_estimators: int = 100, max_samples: typing.Union[int, float, str] = 'auto', contamination: typing.Union[str, float] = 'auto', max_depth: int = 8, bootstrap: bool = False, n_jobs: typing.Optional[int] = None, random_state: int = 23, verbose: bool = False)
   :canonical: risf.forest.RandomIsolationSimilarityForest

   Bases: :py:obj:`sklearn.base.BaseEstimator`, :py:obj:`sklearn.base.OutlierMixin`

   .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.__init__

   .. py:method:: fit(X: typing.Union[numpy.ndarray, risf.risf_data.RisfData], y: typing.Optional[numpy.ndarray] = None) -> risf.forest.RandomIsolationSimilarityForest
      :canonical: risf.forest.RandomIsolationSimilarityForest.fit

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.fit

   .. py:method:: prepare_to_fit(X: typing.Union[numpy.ndarray, risf.risf_data.RisfData])
      :canonical: risf.forest.RandomIsolationSimilarityForest.prepare_to_fit

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.prepare_to_fit

   .. py:method:: create_trees() -> typing.List[risf.tree.RandomIsolationSimilarityTree]
      :canonical: risf.forest.RandomIsolationSimilarityForest.create_trees

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.create_trees

   .. py:method:: set_offset(y: typing.Optional[numpy.ndarray] = None)
      :canonical: risf.forest.RandomIsolationSimilarityForest.set_offset

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.set_offset

   .. py:method:: calculate_mean_path_lengths(X: numpy.ndarray) -> numpy.ndarray
      :canonical: risf.forest.RandomIsolationSimilarityForest.calculate_mean_path_lengths

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.calculate_mean_path_lengths

   .. py:method:: score_samples(X: numpy.ndarray) -> numpy.ndarray
      :canonical: risf.forest.RandomIsolationSimilarityForest.score_samples

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.score_samples

   .. py:method:: decision_function(X: numpy.ndarray) -> numpy.ndarray
      :canonical: risf.forest.RandomIsolationSimilarityForest.decision_function

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.decision_function

   .. py:method:: predict_train(return_raw_scores: bool = False) -> numpy.ndarray
      :canonical: risf.forest.RandomIsolationSimilarityForest.predict_train

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.predict_train

   .. py:method:: predict(X: numpy.ndarray, return_raw_scores: bool = False) -> numpy.ndarray
      :canonical: risf.forest.RandomIsolationSimilarityForest.predict

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.predict

   .. py:method:: get_used_points() -> set
      :canonical: risf.forest.RandomIsolationSimilarityForest.get_used_points

      .. autodoc2-docstring:: risf.forest.RandomIsolationSimilarityForest.get_used_points

.. py:function:: _build_tree(tree: risf.tree.RandomIsolationSimilarityTree, X: numpy.ndarray, tree_idx: int, n_trees: int, subsample_size: int = 256, verbose: int = 0, bootstrap: bool = False)
   :canonical: risf.forest._build_tree

   .. autodoc2-docstring:: risf.forest._build_tree
