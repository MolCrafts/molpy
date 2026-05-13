---
slug: dielectric-susceptibility-03-python-types
criteria:
  - id: ac-001
    summary: ACFResult is a frozen dataclass extending TimeSeriesResult
    type: code
    pass_when: |
      dataclasses.is_dataclass(ACFResult); ACFResult.__dataclass_params__.frozen is True;
      issubclass(ACFResult, TimeSeriesResult)
    status: pending
  - id: ac-002
    summary: ACFResult has acf and n_lags fields with correct types
    type: code
    pass_when: |
      ACFResult has acf: NDArray[np.float64] and n_lags: int fields.
    status: pending
  - id: ac-003
    summary: SpectralResult is a frozen dataclass extending TimeSeriesResult
    type: code
    pass_when: |
      dataclasses.is_dataclass(SpectralResult); frozen=True; issubclass(SpectralResult, TimeSeriesResult)
    status: pending
  - id: ac-004
    summary: SpectralResult has frequency and spectrum fields
    type: code
    pass_when: |
      SpectralResult has frequency: NDArray[np.float64] and spectrum: NDArray[np.float64].
    status: pending
  - id: ac-005
    summary: DielectricResult is a frozen dataclass extending TimeSeriesResult
    type: code
    pass_when: |
      frozen=True; issubclass(DielectricResult, TimeSeriesResult)
    status: pending
  - id: ac-006
    summary: DielectricResult has all required fields
    type: code
    pass_when: |
      frequency, epsilon_real, epsilon_imag, epsilon_static, epsilon_inf, route, component are required;
      conductivity is Optional[NDArray] defaulting to None.
    status: pending
  - id: ac-007
    summary: DielectricSusceptibilityResult is a frozen dataclass extending Result
    type: code
    pass_when: |
      frozen=True; issubclass(DielectricSusceptibilityResult, Result)
    status: pending
  - id: ac-008
    summary: DielectricSusceptibilityResult has results dict and metadata fields
    type: code
    pass_when: |
      results: dict[str, DielectricResult]; metadata: dict[str, Any]
    status: pending
  - id: ac-009
    summary: All 4 classes construct without error
    type: runtime
    pass_when: |
      Each class can be instantiated with required fields only; no construction error.
    status: pending
  - id: ac-010
    summary: Frozen immutability enforced
    type: runtime
    pass_when: |
      Attribute assignment on any of the 4 classes raises FrozenInstanceError.
    status: pending
  - id: ac-011
    summary: to_dict serializes all fields including nested
    type: runtime
    pass_when: |
      DielectricSusceptibilityResult.to_dict() recursively serializes nested DielectricResult values.
    status: pending
  - id: ac-012
    summary: All 4 classes importable from molpy.compute
    type: code
    pass_when: |
      from molpy.compute import ACFResult, SpectralResult, DielectricResult, DielectricSusceptibilityResult
    status: pending
