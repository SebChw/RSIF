:py:mod:`rsif.distance_functions`
=================================

.. py:module:: rsif.distance_functions

.. autodoc2-docstring:: rsif.distance_functions
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`EditDistanceSequencesOfSets <rsif.distance_functions.EditDistanceSequencesOfSets>`
     - .. autodoc2-docstring:: rsif.distance_functions.EditDistanceSequencesOfSets
          :summary:
   * - :py:obj:`DiceDist <rsif.distance_functions.DiceDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.DiceDist
          :summary:
   * - :py:obj:`EuclideanDist <rsif.distance_functions.EuclideanDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.EuclideanDist
          :summary:
   * - :py:obj:`ProperEuclideanDist <rsif.distance_functions.ProperEuclideanDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.ProperEuclideanDist
          :summary:
   * - :py:obj:`ManhattanDist <rsif.distance_functions.ManhattanDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.ManhattanDist
          :summary:
   * - :py:obj:`BagOfWordDist <rsif.distance_functions.BagOfWordDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.BagOfWordDist
          :summary:
   * - :py:obj:`NumericalReprDist <rsif.distance_functions.NumericalReprDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.NumericalReprDist
          :summary:
   * - :py:obj:`CosineDist <rsif.distance_functions.CosineDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.CosineDist
          :summary:
   * - :py:obj:`ChebyshevDist <rsif.distance_functions.ChebyshevDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.ChebyshevDist
          :summary:
   * - :py:obj:`DTWDist <rsif.distance_functions.DTWDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.DTWDist
          :summary:
   * - :py:obj:`LinDist <rsif.distance_functions.LinDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.LinDist
          :summary:
   * - :py:obj:`Goodall3Dist <rsif.distance_functions.Goodall3Dist>`
     - .. autodoc2-docstring:: rsif.distance_functions.Goodall3Dist
          :summary:
   * - :py:obj:`OFDist <rsif.distance_functions.OFDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.OFDist
          :summary:
   * - :py:obj:`GraphDist <rsif.distance_functions.GraphDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.GraphDist
          :summary:
   * - :py:obj:`CrossCorrelationDist <rsif.distance_functions.CrossCorrelationDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.CrossCorrelationDist
          :summary:
   * - :py:obj:`WassersteinDist <rsif.distance_functions.WassersteinDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.WassersteinDist
          :summary:
   * - :py:obj:`JensenShannonDivDist <rsif.distance_functions.JensenShannonDivDist>`
     - .. autodoc2-docstring:: rsif.distance_functions.JensenShannonDivDist
          :summary:
   * - :py:obj:`TSEuclidean <rsif.distance_functions.TSEuclidean>`
     - .. autodoc2-docstring:: rsif.distance_functions.TSEuclidean
          :summary:
   * - :py:obj:`HistEuclidean <rsif.distance_functions.HistEuclidean>`
     - .. autodoc2-docstring:: rsif.distance_functions.HistEuclidean
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`euclidean_projection <rsif.distance_functions.euclidean_projection>`
     - .. autodoc2-docstring:: rsif.distance_functions.euclidean_projection
          :summary:
   * - :py:obj:`manhattan_projection <rsif.distance_functions.manhattan_projection>`
     - .. autodoc2-docstring:: rsif.distance_functions.manhattan_projection
          :summary:
   * - :py:obj:`chebyshev_projection <rsif.distance_functions.chebyshev_projection>`
     - .. autodoc2-docstring:: rsif.distance_functions.chebyshev_projection
          :summary:
   * - :py:obj:`dummy_projection <rsif.distance_functions.dummy_projection>`
     - .. autodoc2-docstring:: rsif.distance_functions.dummy_projection
          :summary:
   * - :py:obj:`BagOfWord_projection <rsif.distance_functions.BagOfWord_projection>`
     - .. autodoc2-docstring:: rsif.distance_functions.BagOfWord_projection
          :summary:
   * - :py:obj:`cosine_sim <rsif.distance_functions.cosine_sim>`
     - .. autodoc2-docstring:: rsif.distance_functions.cosine_sim
          :summary:
   * - :py:obj:`cosine_projection <rsif.distance_functions.cosine_projection>`
     - .. autodoc2-docstring:: rsif.distance_functions.cosine_projection
          :summary:
   * - :py:obj:`jaccard_sim <rsif.distance_functions.jaccard_sim>`
     - .. autodoc2-docstring:: rsif.distance_functions.jaccard_sim
          :summary:
   * - :py:obj:`jaccard_projection <rsif.distance_functions.jaccard_projection>`
     - .. autodoc2-docstring:: rsif.distance_functions.jaccard_projection
          :summary:
   * - :py:obj:`get_frequency_table <rsif.distance_functions.get_frequency_table>`
     - .. autodoc2-docstring:: rsif.distance_functions.get_frequency_table
          :summary:

API
~~~

.. py:function:: euclidean_projection(X, p, q)
   :canonical: rsif.distance_functions.euclidean_projection

   .. autodoc2-docstring:: rsif.distance_functions.euclidean_projection

.. py:function:: manhattan_projection(X, p, q)
   :canonical: rsif.distance_functions.manhattan_projection

   .. autodoc2-docstring:: rsif.distance_functions.manhattan_projection

.. py:function:: chebyshev_projection(X, p, q)
   :canonical: rsif.distance_functions.chebyshev_projection

   .. autodoc2-docstring:: rsif.distance_functions.chebyshev_projection

.. py:function:: dummy_projection(X, p, q)
   :canonical: rsif.distance_functions.dummy_projection

   .. autodoc2-docstring:: rsif.distance_functions.dummy_projection

.. py:function:: BagOfWord_projection(X, p, q)
   :canonical: rsif.distance_functions.BagOfWord_projection

   .. autodoc2-docstring:: rsif.distance_functions.BagOfWord_projection

.. py:function:: cosine_sim(X: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray
   :canonical: rsif.distance_functions.cosine_sim

   .. autodoc2-docstring:: rsif.distance_functions.cosine_sim

.. py:function:: cosine_projection(X, p, q)
   :canonical: rsif.distance_functions.cosine_projection

   .. autodoc2-docstring:: rsif.distance_functions.cosine_projection

.. py:function:: jaccard_sim(X, p)
   :canonical: rsif.distance_functions.jaccard_sim

   .. autodoc2-docstring:: rsif.distance_functions.jaccard_sim

.. py:function:: jaccard_projection(X, p, q)
   :canonical: rsif.distance_functions.jaccard_projection

   .. autodoc2-docstring:: rsif.distance_functions.jaccard_projection

.. py:class:: EditDistanceSequencesOfSets
   :canonical: rsif.distance_functions.EditDistanceSequencesOfSets

   .. autodoc2-docstring:: rsif.distance_functions.EditDistanceSequencesOfSets

   .. py:method:: jaccard(set1, set2)
      :canonical: rsif.distance_functions.EditDistanceSequencesOfSets.jaccard

      .. autodoc2-docstring:: rsif.distance_functions.EditDistanceSequencesOfSets.jaccard

   .. py:method:: __call__(s1, s2)
      :canonical: rsif.distance_functions.EditDistanceSequencesOfSets.__call__

      .. autodoc2-docstring:: rsif.distance_functions.EditDistanceSequencesOfSets.__call__

.. py:class:: DiceDist
   :canonical: rsif.distance_functions.DiceDist

   .. autodoc2-docstring:: rsif.distance_functions.DiceDist

   .. py:method:: __call__(x, y)
      :canonical: rsif.distance_functions.DiceDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.DiceDist.__call__

.. py:class:: EuclideanDist
   :canonical: rsif.distance_functions.EuclideanDist

   .. autodoc2-docstring:: rsif.distance_functions.EuclideanDist

   .. py:method:: __call__(x, y)
      :canonical: rsif.distance_functions.EuclideanDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.EuclideanDist.__call__

.. py:class:: ProperEuclideanDist
   :canonical: rsif.distance_functions.ProperEuclideanDist

   .. autodoc2-docstring:: rsif.distance_functions.ProperEuclideanDist

   .. py:method:: __call__(x, y)
      :canonical: rsif.distance_functions.ProperEuclideanDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.ProperEuclideanDist.__call__

.. py:class:: ManhattanDist
   :canonical: rsif.distance_functions.ManhattanDist

   .. autodoc2-docstring:: rsif.distance_functions.ManhattanDist

   .. py:method:: __call__(x, y)
      :canonical: rsif.distance_functions.ManhattanDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.ManhattanDist.__call__

.. py:class:: BagOfWordDist
   :canonical: rsif.distance_functions.BagOfWordDist

   .. autodoc2-docstring:: rsif.distance_functions.BagOfWordDist

   .. py:method:: __call__(x, y)
      :canonical: rsif.distance_functions.BagOfWordDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.BagOfWordDist.__call__

.. py:class:: NumericalReprDist
   :canonical: rsif.distance_functions.NumericalReprDist

   .. autodoc2-docstring:: rsif.distance_functions.NumericalReprDist

   .. py:method:: __call__(x, y)
      :canonical: rsif.distance_functions.NumericalReprDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.NumericalReprDist.__call__

.. py:class:: CosineDist
   :canonical: rsif.distance_functions.CosineDist

   .. autodoc2-docstring:: rsif.distance_functions.CosineDist

   .. py:method:: __call__(x, y)
      :canonical: rsif.distance_functions.CosineDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.CosineDist.__call__

.. py:class:: ChebyshevDist
   :canonical: rsif.distance_functions.ChebyshevDist

   .. autodoc2-docstring:: rsif.distance_functions.ChebyshevDist

   .. py:method:: __call__(x, y)
      :canonical: rsif.distance_functions.ChebyshevDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.ChebyshevDist.__call__

.. py:class:: DTWDist
   :canonical: rsif.distance_functions.DTWDist

   .. autodoc2-docstring:: rsif.distance_functions.DTWDist

   .. py:method:: __call__(*args, **kwargs)
      :canonical: rsif.distance_functions.DTWDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.DTWDist.__call__

   .. py:method:: dist(x1, x2)
      :canonical: rsif.distance_functions.DTWDist.dist

      .. autodoc2-docstring:: rsif.distance_functions.DTWDist.dist

.. py:function:: get_frequency_table(data, normalize=True)
   :canonical: rsif.distance_functions.get_frequency_table

   .. autodoc2-docstring:: rsif.distance_functions.get_frequency_table

.. py:class:: LinDist(data)
   :canonical: rsif.distance_functions.LinDist

   .. autodoc2-docstring:: rsif.distance_functions.LinDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: rsif.distance_functions.LinDist.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: rsif.distance_functions.LinDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.LinDist.__call__

   .. py:method:: dist(x1, x2)
      :canonical: rsif.distance_functions.LinDist.dist

      .. autodoc2-docstring:: rsif.distance_functions.LinDist.dist

.. py:class:: Goodall3Dist(data)
   :canonical: rsif.distance_functions.Goodall3Dist

   .. autodoc2-docstring:: rsif.distance_functions.Goodall3Dist

   .. rubric:: Initialization

   .. autodoc2-docstring:: rsif.distance_functions.Goodall3Dist.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: rsif.distance_functions.Goodall3Dist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.Goodall3Dist.__call__

   .. py:method:: dist(x1, x2)
      :canonical: rsif.distance_functions.Goodall3Dist.dist

      .. autodoc2-docstring:: rsif.distance_functions.Goodall3Dist.dist

.. py:class:: OFDist(data)
   :canonical: rsif.distance_functions.OFDist

   .. autodoc2-docstring:: rsif.distance_functions.OFDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: rsif.distance_functions.OFDist.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: rsif.distance_functions.OFDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.OFDist.__call__

   .. py:method:: dist(x1, x2)
      :canonical: rsif.distance_functions.OFDist.dist

      .. autodoc2-docstring:: rsif.distance_functions.OFDist.dist

.. py:class:: GraphDist(dist_class, params: dict = {})
   :canonical: rsif.distance_functions.GraphDist

   .. autodoc2-docstring:: rsif.distance_functions.GraphDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: rsif.distance_functions.GraphDist.__init__

   .. py:method:: __call__(G1, G2)
      :canonical: rsif.distance_functions.GraphDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.GraphDist.__call__

.. py:class:: CrossCorrelationDist()
   :canonical: rsif.distance_functions.CrossCorrelationDist

   .. autodoc2-docstring:: rsif.distance_functions.CrossCorrelationDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: rsif.distance_functions.CrossCorrelationDist.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: rsif.distance_functions.CrossCorrelationDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.CrossCorrelationDist.__call__

   .. py:method:: dist(Arr1, Arr2)
      :canonical: rsif.distance_functions.CrossCorrelationDist.dist

      .. autodoc2-docstring:: rsif.distance_functions.CrossCorrelationDist.dist

.. py:class:: WassersteinDist
   :canonical: rsif.distance_functions.WassersteinDist

   .. autodoc2-docstring:: rsif.distance_functions.WassersteinDist

   .. py:method:: __call__(hist1, hist2)
      :canonical: rsif.distance_functions.WassersteinDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.WassersteinDist.__call__

.. py:class:: JensenShannonDivDist()
   :canonical: rsif.distance_functions.JensenShannonDivDist

   .. autodoc2-docstring:: rsif.distance_functions.JensenShannonDivDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: rsif.distance_functions.JensenShannonDivDist.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: rsif.distance_functions.JensenShannonDivDist.__call__

      .. autodoc2-docstring:: rsif.distance_functions.JensenShannonDivDist.__call__

   .. py:method:: adjust(Arr1, Arr2)
      :canonical: rsif.distance_functions.JensenShannonDivDist.adjust

      .. autodoc2-docstring:: rsif.distance_functions.JensenShannonDivDist.adjust

   .. py:method:: dist(Arr1, Arr2)
      :canonical: rsif.distance_functions.JensenShannonDivDist.dist

      .. autodoc2-docstring:: rsif.distance_functions.JensenShannonDivDist.dist

.. py:class:: TSEuclidean()
   :canonical: rsif.distance_functions.TSEuclidean

   .. autodoc2-docstring:: rsif.distance_functions.TSEuclidean

   .. rubric:: Initialization

   .. autodoc2-docstring:: rsif.distance_functions.TSEuclidean.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: rsif.distance_functions.TSEuclidean.__call__

      .. autodoc2-docstring:: rsif.distance_functions.TSEuclidean.__call__

   .. py:method:: adjust(Arr1, Arr2)
      :canonical: rsif.distance_functions.TSEuclidean.adjust

      .. autodoc2-docstring:: rsif.distance_functions.TSEuclidean.adjust

   .. py:method:: dist(Arr1, Arr2)
      :canonical: rsif.distance_functions.TSEuclidean.dist

      .. autodoc2-docstring:: rsif.distance_functions.TSEuclidean.dist

.. py:class:: HistEuclidean()
   :canonical: rsif.distance_functions.HistEuclidean

   .. autodoc2-docstring:: rsif.distance_functions.HistEuclidean

   .. rubric:: Initialization

   .. autodoc2-docstring:: rsif.distance_functions.HistEuclidean.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: rsif.distance_functions.HistEuclidean.__call__

      .. autodoc2-docstring:: rsif.distance_functions.HistEuclidean.__call__

   .. py:method:: adjust(values1, bins1, values2, bins2)
      :canonical: rsif.distance_functions.HistEuclidean.adjust

      .. autodoc2-docstring:: rsif.distance_functions.HistEuclidean.adjust

   .. py:method:: dist(hist1, hist2)
      :canonical: rsif.distance_functions.HistEuclidean.dist

      .. autodoc2-docstring:: rsif.distance_functions.HistEuclidean.dist
