"""Bundled force-field files; locate them with :func:`molpy.resources.get_path`.

* ``tip3p.xml`` — TIP3P water, an OpenMM force-field XML
  (``mp.io.read_openmm_xml_forcefield(get_path("forcefield/tip3p.xml"))``).
* ``clp.xml`` — the CL&P ionic-liquid typing force field
  (``mp.ff.typifier.OplsAaTypifier(get_path("forcefield/clp.xml"))``).

OPLS-AA and the CL&Pol Drude table (``mp.ff.params.clpol_polarizability``)
ship with molrs.
"""
