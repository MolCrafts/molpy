"""Bundled force-field files; locate them with :func:`molpy.data.get_forcefield_path`.

* ``tip3p.xml`` — TIP3P water, an OpenMM force-field XML
  (``mp.ff.forcefield.read_forcefield_xml``).
* ``clp.xml`` — the CL&P ionic-liquid typing force field
  (``mp.ff.typifier.OPLSAATypifier(get_forcefield_path("clp.xml"))``).

OPLS-AA and the CL&Pol Drude table (``mp.ff.params.clpol_polarizability``)
ship with molrs.
"""
