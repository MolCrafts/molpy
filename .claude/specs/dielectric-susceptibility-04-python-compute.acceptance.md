---
slug: dielectric-susceptibility-04-python-compute
criteria:
  - id: ac-001
    summary: ACFAnalyzer extracts columns and delegates to molrs.signal.acf_fft
    type: code
    pass_when: |
      Trajectory with 10+ frames, each with atoms["x","y","z"] and Box, yields ACFResult
      with acf shape (max_lag+1,). Mock confirms molrs.signal.acf_fft called once.
    status: pending
  - id: ac-002
    summary: ACFAnalyzer calls Box.diff_dr per frame when unwrap=True
    type: code
    pass_when: |
      With unwrap=True, Box.diff_dr called (n_frames−1) times.
    status: pending
  - id: ac-003
    summary: ACFAnalyzer raises ValueError for missing column, missing box, insufficient frames
    type: code
    pass_when: |
      Missing "x" → ValueError mentioning "x". No box + unwrap=True → ValueError.
      < 2 frames → ValueError.
    status: pending
  - id: ac-004
    summary: ACFAnalyzer returns normalized ACF (ACF[0] == 1.0)
    type: scientific
    pass_when: |
      On a synthetic trajectory, ACFResult.acf[0] == 1.0 to 1e-10.
    status: pending
  - id: ac-005
    summary: SpectralAnalyzer calls molrs.signal.apply_window and frequency_grid
    type: code
    pass_when: |
      Mock confirms apply_window called with ACF array + window_type;
      frequency_grid called with n_lags + dt.
    status: pending
  - id: ac-006
    summary: SpectralAnalyzer returns SpectralResult with correct shapes
    type: code
    pass_when: |
      frequency and spectrum arrays have same length = n_lags//2 + 1.
    status: pending
  - id: ac-007
    summary: DielectricSusceptibility calls molrs.dielectric.compute_dipole_moment
    type: code
    pass_when: |
      Mock confirms molrs.dielectric.compute_dipole_moment called with charge array and position array.
    status: pending
  - id: ac-008
    summary: DielectricSusceptibility calls molrs.dielectric.compute_current_density
    type: code
    pass_when: |
      Mock confirms molrs.dielectric.compute_current_density called with dipole_moments, dt, volume.
    status: pending
  - id: ac-009
    summary: DielectricSusceptibility calls einstein_helfand_spectrum when route enabled
    type: code
    pass_when: |
      With routes=["einstein-helfand"], molrs.dielectric.einstein_helfand_spectrum called once.
    status: pending
  - id: ac-010
    summary: DielectricSusceptibility calls green_kubo_spectrum when route enabled
    type: code
    pass_when: |
      With routes=["green-kubo"], molrs.dielectric.green_kubo_spectrum called once.
    status: pending
  - id: ac-011
    summary: DielectricSusceptibility calls static_dielectric_constant
    type: code
    pass_when: |
      molrs.dielectric.static_dielectric_constant called with M_series, volume, temperature, epsilon_inf.
    status: pending
  - id: ac-012
    summary: DielectricSusceptibility returns complete result with all route keys
    type: code
    pass_when: |
      Result dict has keys matching all enabled routes × components (e.g. "EH-full", "GK-full").
    status: pending
  - id: ac-013
    summary: Zero Python physics computation in dielectric.py
    type: code
    pass_when: |
      Source review: no np.fft calls, no manual physics formulas in dielectric.py.
      All computation via molrs.signal.* and molrs.dielectric.* calls.
    status: pending
  - id: ac-014
    summary: Input Trajectory not mutated by any Compute class
    type: runtime
    pass_when: |
      After calling ACFAnalyzer, SpectralAnalyzer, or DielectricSusceptibility,
      trajectory frames and box are byte-identical to pre-call state.
    status: pending
  - id: ac-015
    summary: All 3 classes exported from molpy.compute
    type: code
    pass_when: |
      from molpy.compute import ACFAnalyzer, SpectralAnalyzer, DielectricSusceptibility
    status: pending
  - id: ac-016
    summary: Full test suite passes
    type: code
    pass_when: |
      pytest tests/test_compute/test_dielectric.py -v -m "not external" exits 0.
    status: pending
