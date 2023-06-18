:py:mod:`risf.risf_data`
========================

.. py:module:: risf.risf_data

.. autodoc2-docstring:: risf.risf_data
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`RisfData <risf.risf_data.RisfData>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`list_to_numpy <risf.risf_data.list_to_numpy>`
     - .. autodoc2-docstring:: risf.risf_data.list_to_numpy
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SUPPORTED_TYPES <risf.risf_data.SUPPORTED_TYPES>`
     - .. autodoc2-docstring:: risf.risf_data.SUPPORTED_TYPES
          :summary:

API
~~~

.. py:function:: list_to_numpy(transformed)
   :canonical: risf.risf_data.list_to_numpy

   .. autodoc2-docstring:: risf.risf_data.list_to_numpy

.. py:data:: SUPPORTED_TYPES
   :canonical: risf.risf_data.SUPPORTED_TYPES
   :value: ((), (), ())

   .. autodoc2-docstring:: risf.risf_data.SUPPORTED_TYPES

.. py:class:: RisfData(num_of_selected_objects: int = None, random_state=23)
   :canonical: risf.risf_data.RisfData

   Bases: :py:obj:`list`

   .. py:method:: validate_column(X)
      :canonical: risf.risf_data.RisfData.validate_column
      :classmethod:

      .. autodoc2-docstring:: risf.risf_data.RisfData.validate_column

   .. py:method:: distance_check(X, dist)
      :canonical: risf.risf_data.RisfData.distance_check
      :staticmethod:

      .. autodoc2-docstring:: risf.risf_data.RisfData.distance_check

   .. py:method:: update_metadata(name)
      :canonical: risf.risf_data.RisfData.update_metadata

      .. autodoc2-docstring:: risf.risf_data.RisfData.update_metadata

   .. py:method:: shape_check(X)
      :canonical: risf.risf_data.RisfData.shape_check

      .. autodoc2-docstring:: risf.risf_data.RisfData.shape_check

   .. py:method:: add_distances(X, distances)
      :canonical: risf.risf_data.RisfData.add_distances

      .. autodoc2-docstring:: risf.risf_data.RisfData.add_distances

   .. py:method:: add_data(X, dist: list, name=None)
      :canonical: risf.risf_data.RisfData.add_data

      .. autodoc2-docstring:: risf.risf_data.RisfData.add_data

   .. py:method:: precompute_distances(n_jobs=1, train_data: list = None, selected_objects=None)
      :canonical: risf.risf_data.RisfData.precompute_distances

      .. autodoc2-docstring:: risf.risf_data.RisfData.precompute_distances

   .. py:method:: impute_missing_values(distance, strategy='max')
      :canonical: risf.risf_data.RisfData.impute_missing_values

      .. autodoc2-docstring:: risf.risf_data.RisfData.impute_missing_values

   .. py:method:: transform(list_of_X: list, forest, n_jobs=1, precomputed_distances=None)
      :canonical: risf.risf_data.RisfData.transform

      .. autodoc2-docstring:: risf.risf_data.RisfData.transform
