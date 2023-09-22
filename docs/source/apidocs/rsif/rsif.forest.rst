:py:mod:`rsif.forest`
=====================

.. py:module:: rsif.forest

.. autodoc2-docstring:: rsif.forest
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`RandomSimilarityIsolationForest <rsif.forest.RandomSimilarityIsolationForest>`
     - .. autodoc2-docstring:: rsif.forest.RandomSimilarityIsolationForest
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_build_tree <rsif.forest._build_tree>`
     - .. autodoc2-docstring:: rsif.forest._build_tree
          :summary:

API
~~~

.. py:class:: RandomSimilarityIsolationForest(distances: typing.List[typing.List[typing.Union[rsif.distance.SelectiveDistance, rsif.distance.TrainDistanceMixin]]] = [[SelectiveDistance(euclidean_projection, 1, 1)]], n_estimators: int = 100, max_samples: typing.Union[int, float, str] = 'auto', contamination: typing.Union[str, float] = 'auto', bootstrap: bool = False, n_jobs: typing.Optional[int] = None, random_state: int = 23, verbose: bool = False, pairs_strategy='local', **kwargs)
   :canonical: rsif.forest.RandomSimilarityIsolationForest

   Bases: :py:obj:`sklearn.base.BaseEstimator`, :py:obj:`sklearn.base.OutlierMixin`

   .. autodoc2-docstring:: rsif.forest.RandomSimilarityIsolationForest

   .. rubric:: Initialization

   .. autodoc2-docstring:: rsif.forest.RandomSimilarityIsolationForest.__init__

   .. py:method:: fit(X: typing.Union[numpy.ndarray, rsif.rsif_data.RsifData], y: typing.Optional[numpy.ndarray] = None) -> rsif.forest.RandomSimilarityIsolationForest
      :canonical: rsif.forest.RandomSimilarityIsolationForest.fit

      .. autodoc2-docstring:: rsif.forest.RandomSimilarityIsolationForest.fit

   .. py:method:: prepare_to_fit(X: typing.Union[numpy.ndarray, rsif.rsif_data.RsifData])
      :canonical: rsif.forest.RandomSimilarityIsolationForest.prepare_to_fit

      .. autodoc2-docstring:: rsif.forest.RandomSimilarityIsolationForest.prepare_to_fit

   .. py:method:: create_trees() -> typing.List[rsif.tree.RandomSimilarityIsolationTree]
      :canonical: rsif.forest.RandomSimilarityIsolationForest.create_trees

      .. autodoc2-docstring:: rsif.forest.RandomSimilarityIsolationForest.create_trees

   .. py:method:: set_offset(y: typing.Optional[numpy.ndarray] = None)
      :canonical: rsif.forest.RandomSimilarityIsolationForest.set_offset

      .. autodoc2-docstring:: rsif.forest.RandomSimilarityIsolationForest.set_offset

   .. py:method:: calculate_mean_path_lengths(X: numpy.ndarray) -> numpy.ndarray
      :canonical: rsif.forest.RandomSimilarityIsolationForest.calculate_mean_path_lengths

      .. autodoc2-docstring:: rsif.forest.RandomSimilarityIsolationForest.calculate_mean_path_lengths

   .. py:method:: score_samples(X: numpy.ndarray) -> numpy.ndarray
      :canonical: rsif.forest.RandomSimilarityIsolationForest.score_samples

      .. autodoc2-docstring:: rsif.forest.RandomSimilarityIsolationForest.score_samples

   .. py:method:: decision_function(X: numpy.ndarray) -> numpy.ndarray
      :canonical: rsif.forest.RandomSimilarityIsolationForest.decision_function

      .. autodoc2-docstring:: rsif.forest.RandomSimilarityIsolationForest.decision_function

   .. py:method:: predict_train(return_raw_scores: bool = False) -> numpy.ndarray
      :canonical: rsif.forest.RandomSimilarityIsolationForest.predict_train

      .. autodoc2-docstring:: rsif.forest.RandomSimilarityIsolationForest.predict_train

   .. py:method:: predict(X: typing.Union[numpy.ndarray, rsif.rsif_data.RsifData], return_raw_scores: bool = False) -> numpy.ndarray
      :canonical: rsif.forest.RandomSimilarityIsolationForest.predict

      .. autodoc2-docstring:: rsif.forest.RandomSimilarityIsolationForest.predict

   .. py:method:: get_used_points() -> set
      :canonical: rsif.forest.RandomSimilarityIsolationForest.get_used_points

      .. autodoc2-docstring:: rsif.forest.RandomSimilarityIsolationForest.get_used_points

.. py:function:: _build_tree(tree: rsif.tree.RandomSimilarityIsolationTree, X: numpy.ndarray, tree_idx: int, n_trees: int, subsample_size: int = 256, verbose: int = 0, bootstrap: bool = False)
   :canonical: rsif.forest._build_tree

   .. autodoc2-docstring:: rsif.forest._build_tree
