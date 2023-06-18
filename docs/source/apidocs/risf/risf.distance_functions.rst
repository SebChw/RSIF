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

   * - :py:obj:`JaccardDist <risf.distance_functions.JaccardDist>`
     - .. autodoc2-docstring:: risf.distance_functions.JaccardDist
          :summary:
   * - :py:obj:`DiceDist <risf.distance_functions.DiceDist>`
     - .. autodoc2-docstring:: risf.distance_functions.DiceDist
          :summary:
   * - :py:obj:`DTWDist <risf.distance_functions.DTWDist>`
     - .. autodoc2-docstring:: risf.distance_functions.DTWDist
          :summary:
   * - :py:obj:`JaccardGraphDist <risf.distance_functions.JaccardGraphDist>`
     - .. autodoc2-docstring:: risf.distance_functions.JaccardGraphDist
          :summary:
   * - :py:obj:`IpsenMikailovDist <risf.distance_functions.IpsenMikailovDist>`
     - .. autodoc2-docstring:: risf.distance_functions.IpsenMikailovDist
          :summary:
   * - :py:obj:`NetSmileDist <risf.distance_functions.NetSmileDist>`
     - .. autodoc2-docstring:: risf.distance_functions.NetSmileDist
          :summary:
   * - :py:obj:`PortraitDivergenceDist <risf.distance_functions.PortraitDivergenceDist>`
     - .. autodoc2-docstring:: risf.distance_functions.PortraitDivergenceDist
          :summary:
   * - :py:obj:`DegreeDivergenceDist <risf.distance_functions.DegreeDivergenceDist>`
     - .. autodoc2-docstring:: risf.distance_functions.DegreeDivergenceDist
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
   * - :py:obj:`cosine_projection <risf.distance_functions.cosine_projection>`
     - .. autodoc2-docstring:: risf.distance_functions.cosine_projection
          :summary:
   * - :py:obj:`jaccard_projection <risf.distance_functions.jaccard_projection>`
     - .. autodoc2-docstring:: risf.distance_functions.jaccard_projection
          :summary:
   * - :py:obj:`dice_projection <risf.distance_functions.dice_projection>`
     - .. autodoc2-docstring:: risf.distance_functions.dice_projection
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

.. py:function:: cosine_projection(X, p, q)
   :canonical: risf.distance_functions.cosine_projection

   .. autodoc2-docstring:: risf.distance_functions.cosine_projection

.. py:function:: jaccard_projection(X, p, q)
   :canonical: risf.distance_functions.jaccard_projection

   .. autodoc2-docstring:: risf.distance_functions.jaccard_projection

.. py:function:: dice_projection(X, p, q)
   :canonical: risf.distance_functions.dice_projection

   .. autodoc2-docstring:: risf.distance_functions.dice_projection

.. py:class:: JaccardDist
   :canonical: risf.distance_functions.JaccardDist

   .. autodoc2-docstring:: risf.distance_functions.JaccardDist

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.JaccardDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.JaccardDist.__call__

   .. py:method:: dist(x1, x2)
      :canonical: risf.distance_functions.JaccardDist.dist

      .. autodoc2-docstring:: risf.distance_functions.JaccardDist.dist

.. py:class:: DiceDist
   :canonical: risf.distance_functions.DiceDist

   .. autodoc2-docstring:: risf.distance_functions.DiceDist

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.DiceDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.DiceDist.__call__

   .. py:method:: dist(x1, x2)
      :canonical: risf.distance_functions.DiceDist.dist

      .. autodoc2-docstring:: risf.distance_functions.DiceDist.dist

.. py:class:: DTWDist
   :canonical: risf.distance_functions.DTWDist

   .. autodoc2-docstring:: risf.distance_functions.DTWDist

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.DTWDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.DTWDist.__call__

   .. py:method:: dist(x1, x2)
      :canonical: risf.distance_functions.DTWDist.dist

      .. autodoc2-docstring:: risf.distance_functions.DTWDist.dist

.. py:class:: JaccardGraphDist()
   :canonical: risf.distance_functions.JaccardGraphDist

   .. autodoc2-docstring:: risf.distance_functions.JaccardGraphDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance_functions.JaccardGraphDist.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.JaccardGraphDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.JaccardGraphDist.__call__

   .. py:method:: dist(G1, G2)
      :canonical: risf.distance_functions.JaccardGraphDist.dist

      .. autodoc2-docstring:: risf.distance_functions.JaccardGraphDist.dist

.. py:class:: IpsenMikailovDist()
   :canonical: risf.distance_functions.IpsenMikailovDist

   .. autodoc2-docstring:: risf.distance_functions.IpsenMikailovDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance_functions.IpsenMikailovDist.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.IpsenMikailovDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.IpsenMikailovDist.__call__

   .. py:method:: dist(G1, G2, hwhm=0.08)
      :canonical: risf.distance_functions.IpsenMikailovDist.dist

      .. autodoc2-docstring:: risf.distance_functions.IpsenMikailovDist.dist

.. py:class:: NetSmileDist()
   :canonical: risf.distance_functions.NetSmileDist

   .. autodoc2-docstring:: risf.distance_functions.NetSmileDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance_functions.NetSmileDist.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.NetSmileDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.NetSmileDist.__call__

   .. py:method:: dist(G1, G2)
      :canonical: risf.distance_functions.NetSmileDist.dist

      .. autodoc2-docstring:: risf.distance_functions.NetSmileDist.dist

.. py:class:: PortraitDivergenceDist()
   :canonical: risf.distance_functions.PortraitDivergenceDist

   .. autodoc2-docstring:: risf.distance_functions.PortraitDivergenceDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance_functions.PortraitDivergenceDist.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.PortraitDivergenceDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.PortraitDivergenceDist.__call__

   .. py:method:: dist(G1, G2)
      :canonical: risf.distance_functions.PortraitDivergenceDist.dist

      .. autodoc2-docstring:: risf.distance_functions.PortraitDivergenceDist.dist

.. py:class:: DegreeDivergenceDist()
   :canonical: risf.distance_functions.DegreeDivergenceDist

   .. autodoc2-docstring:: risf.distance_functions.DegreeDivergenceDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance_functions.DegreeDivergenceDist.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.DegreeDivergenceDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.DegreeDivergenceDist.__call__

   .. py:method:: dist(G1, G2)
      :canonical: risf.distance_functions.DegreeDivergenceDist.dist

      .. autodoc2-docstring:: risf.distance_functions.DegreeDivergenceDist.dist

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

.. py:class:: WassersteinDist()
   :canonical: risf.distance_functions.WassersteinDist

   .. autodoc2-docstring:: risf.distance_functions.WassersteinDist

   .. rubric:: Initialization

   .. autodoc2-docstring:: risf.distance_functions.WassersteinDist.__init__

   .. py:method:: __call__(*args, **kwargs)
      :canonical: risf.distance_functions.WassersteinDist.__call__

      .. autodoc2-docstring:: risf.distance_functions.WassersteinDist.__call__

   .. py:method:: dist(hist1, hist2)
      :canonical: risf.distance_functions.WassersteinDist.dist

      .. autodoc2-docstring:: risf.distance_functions.WassersteinDist.dist

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
