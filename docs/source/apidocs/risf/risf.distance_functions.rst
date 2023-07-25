:py:mod:`risf.distance_functions`
=================================

.. py:module:: risf.distance_functions

.. autodoc2-docstring:: risf.distance_functions
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`EditDistanceSequencesOfSets <risf.distance_functions.EditDistanceSequencesOfSets>`
     - .. autodoc2-docstring:: risf.distance_functions.EditDistanceSequencesOfSets
          :summary:
   * - :py:obj:`DiceDist <risf.distance_functions.DiceDist>`
     - .. autodoc2-docstring:: risf.distance_functions.DiceDist
          :summary:
   * - :py:obj:`EuclideanDist <risf.distance_functions.EuclideanDist>`
     - .. autodoc2-docstring:: risf.distance_functions.EuclideanDist
          :summary:
   * - :py:obj:`ManhattanDist <risf.distance_functions.ManhattanDist>`
     - .. autodoc2-docstring:: risf.distance_functions.ManhattanDist
          :summary:
   * - :py:obj:`ChebyshevDist <risf.distance_functions.ChebyshevDist>`
     - .. autodoc2-docstring:: risf.distance_functions.ChebyshevDist
          :summary:
   * - :py:obj:`DTWDist <risf.distance_functions.DTWDist>`
     - .. autodoc2-docstring:: risf.distance_functions.DTWDist
          :summary:
   * - :py:obj:`GraphDist <risf.distance_functions.GraphDist>`
     - .. autodoc2-docstring:: risf.distance_functions.GraphDist
          :summary:
   * - :py:obj:`CrossCorrelationDist <risf.distance_functions.CrossCorrelationDist>`
     - .. autodoc2-docstring:: risf.distance_functions.CrossCorrelationDist
          :summary:
   * - :py:obj:`WassersteinDist <risf.distance_functions.WassersteinDist>`
     - .. autodoc2-docstring:: risf.distance_functions.WassersteinDist
          :summary:
   * - :py:obj:`JensenShannonDivDist <risf.distance_functions.JensenShannonDivDist>`
     - .. autodoc2-docstring:: risf.distance_functions.JensenShannonDivDist
          :summary:
   * - :py:obj:`TSEuclidean <risf.distance_functions.TSEuclidean>`
     - .. autodoc2-docstring:: risf.distance_functions.TSEuclidean
          :summary:
   * - :py:obj:`HistEuclidean <risf.distance_functions.HistEuclidean>`
     - .. autodoc2-docstring:: risf.distance_functions.HistEuclidean
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`euclidean_projection <risf.distance_functions.euclidean_projection>`
     - .. autodoc2-docstring:: risf.distance_functions.euclidean_projection
          :summary:
   * - :py:obj:`manhattan_projection <risf.distance_functions.manhattan_projection>`
     - .. autodoc2-docstring:: risf.distance_functions.manhattan_projection
          :summary:
   * - :py:obj:`chebyshev_projection <risf.distance_functions.chebyshev_projection>`
     - .. autodoc2-docstring:: risf.distance_functions.chebyshev_projection
          :summary:
   * - :py:obj:`cosine_sim <risf.distance_functions.cosine_sim>`
     - .. autodoc2-docstring:: risf.distance_functions.cosine_sim
          :summary:
   * - :py:obj:`cosine_projection <risf.distance_functions.cosine_projection>`
     - .. autodoc2-docstring:: risf.distance_functions.cosine_projection
          :summary:
   * - :py:obj:`jaccard_sim <risf.distance_functions.jaccard_sim>`
     - .. autodoc2-docstring:: risf.distance_functions.jaccard_sim
          :summary:
   * - :py:obj:`jaccard_projection <risf.distance_functions.jaccard_projection>`
     - .. autodoc2-docstring:: risf.distance_functions.jaccard_projection
          :summary:

API
~~~

.. py:function:: euclidean_projection(X, p, q)
   :canonical: risf.distance_functions.euclidean_projection

   .. autodoc2-docstring:: risf.distance_functions.euclidean_projection

.. py:function:: manhattan_projection(X, p, q)
   :canonical: risf.distance_functions.manhattan_projection

   .. autodoc2-docstring:: risf.distance_functions.manhattan_projection

.. py:function:: chebyshev_projection(X, p, q)
   :canonical: risf.distance_functions.chebyshev_projection

   .. autodoc2-docstring:: risf.distance_functions.chebyshev_projection

.. py:function:: cosine_sim(X: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray
   :canonical: risf.distance_functions.cosine_sim

   .. autodoc2-docstring:: risf.distance_functions.cosine_sim

.. py:function:: cosine_projection(X, p, q)
   :canonical: risf.distance_functions.cosine_projection

   .. autodoc2-docstring:: risf.distance_functions.cosine_projection

.. py:function:: jaccard_sim(X, p)
   :canonical: risf.distance_functions.jaccard_sim

   .. autodoc2-docstring:: risf.distance_functions.jaccard_sim

.. py:function:: jaccard_projection(X, p, q)
   :canonical: risf.distance_functions.jaccard_projection

   .. autodoc2-docstring:: risf.distance_functions.jaccard_projection

.. py:class:: EditDistanceSequencesOfSets
   :canonical: risf.distance_functions.EditDistanceSequencesOfSets

   .. autodoc2-docstring:: risf.distance_functions.EditDistanceSequencesOfSets

   .. py:method:: jaccard(set1, set2)
      :canonical: risf.distance_functions.EditDistanceSequencesOfSets.jaccard

      .. autodoc2-docstring:: risf.distance_functions.EditDistanceSequencesOfSets.jaccard

   .. py:method:: __call__(s1, s2)
      :canonical: risf.distance_functions.EditDistanceSequencesOfSets.__call__

      .. autodoc2-docstring:: risf.distance_functions.EditDistanceSequencesOfSets.__call__

.. py:class:: DiceDist
   :canonical: risf.distance_functions.DiceDist

   .. autodoc2-docstring:: risf.distance_functions.DiceDist

   .. py:method:: __call__(x, y)
      :canonical: risf.distance_functions.DiceDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.DiceDist.__call__

.. py:class:: EuclideanDist
   :canonical: risf.distance_functions.EuclideanDist

   .. autodoc2-docstring:: risf.distance_functions.EuclideanDist

   .. py:method:: __call__(x, y)
      :canonical: risf.distance_functions.EuclideanDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.EuclideanDist.__call__

.. py:class:: ManhattanDist
   :canonical: risf.distance_functions.ManhattanDist

   .. autodoc2-docstring:: risf.distance_functions.ManhattanDist

   .. py:method:: __call__(x, y)
      :canonical: risf.distance_functions.ManhattanDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.ManhattanDist.__call__

.. py:class:: ChebyshevDist
   :canonical: risf.distance_functions.ChebyshevDist

   .. autodoc2-docstring:: risf.distance_functions.ChebyshevDist

   .. py:method:: __call__(x, y)
      :canonical: risf.distance_functions.ChebyshevDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.ChebyshevDist.__call__

.. py:class:: DTWDist
   :canonical: risf.distance_functions.DTWDist

   .. autodoc2-docstring:: risf.distance_functions.DTWDist

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.DTWDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.DTWDist.__call__

   .. py:method:: dist(x1, x2)
      :canonical: risf.distance_functions.DTWDist.dist

      .. autodoc2-docstring:: risf.distance_functions.DTWDist.dist

.. py:class:: GraphDist(dist_class, params: dict = {})
   :canonical: risf.distance_functions.GraphDist

   .. autodoc2-docstring:: risf.distance_functions.GraphDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance_functions.GraphDist.__init__

   .. py:method:: __call__(G1, G2)
      :canonical: risf.distance_functions.GraphDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.GraphDist.__call__

.. py:class:: CrossCorrelationDist()
   :canonical: risf.distance_functions.CrossCorrelationDist

   .. autodoc2-docstring:: risf.distance_functions.CrossCorrelationDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance_functions.CrossCorrelationDist.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.CrossCorrelationDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.CrossCorrelationDist.__call__

   .. py:method:: dist(Arr1, Arr2)
      :canonical: risf.distance_functions.CrossCorrelationDist.dist

      .. autodoc2-docstring:: risf.distance_functions.CrossCorrelationDist.dist

.. py:class:: WassersteinDist
   :canonical: risf.distance_functions.WassersteinDist

   .. autodoc2-docstring:: risf.distance_functions.WassersteinDist

   .. py:method:: __call__(hist1, hist2)
      :canonical: risf.distance_functions.WassersteinDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.WassersteinDist.__call__

.. py:class:: JensenShannonDivDist()
   :canonical: risf.distance_functions.JensenShannonDivDist

   .. autodoc2-docstring:: risf.distance_functions.JensenShannonDivDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance_functions.JensenShannonDivDist.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.JensenShannonDivDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.JensenShannonDivDist.__call__

   .. py:method:: adjust(Arr1, Arr2)
      :canonical: risf.distance_functions.JensenShannonDivDist.adjust

      .. autodoc2-docstring:: risf.distance_functions.JensenShannonDivDist.adjust

   .. py:method:: dist(Arr1, Arr2)
      :canonical: risf.distance_functions.JensenShannonDivDist.dist

      .. autodoc2-docstring:: risf.distance_functions.JensenShannonDivDist.dist

.. py:class:: TSEuclidean()
   :canonical: risf.distance_functions.TSEuclidean

   .. autodoc2-docstring:: risf.distance_functions.TSEuclidean

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance_functions.TSEuclidean.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.TSEuclidean.__call__

      .. autodoc2-docstring:: risf.distance_functions.TSEuclidean.__call__

   .. py:method:: adjust(Arr1, Arr2)
      :canonical: risf.distance_functions.TSEuclidean.adjust

      .. autodoc2-docstring:: risf.distance_functions.TSEuclidean.adjust

   .. py:method:: dist(Arr1, Arr2)
      :canonical: risf.distance_functions.TSEuclidean.dist

      .. autodoc2-docstring:: risf.distance_functions.TSEuclidean.dist

.. py:class:: HistEuclidean()
   :canonical: risf.distance_functions.HistEuclidean

   .. autodoc2-docstring:: risf.distance_functions.HistEuclidean

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance_functions.HistEuclidean.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.HistEuclidean.__call__

      .. autodoc2-docstring:: risf.distance_functions.HistEuclidean.__call__

   .. py:method:: adjust(values1, bins1, values2, bins2)
      :canonical: risf.distance_functions.HistEuclidean.adjust

      .. autodoc2-docstring:: risf.distance_functions.HistEuclidean.adjust

   .. py:method:: dist(hist1, hist2)
      :canonical: risf.distance_functions.HistEuclidean.dist

      .. autodoc2-docstring:: risf.distance_functions.HistEuclidean.dist
