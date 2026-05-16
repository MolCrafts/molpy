---
slug: dielectric-susceptibility-01-molrs-signal
criteria:
  - id: ac-001
    summary: acf_fft returns un-normalized ACF with correct shape
    type: code
    pass_when: |
      molrs.signal.acf_fft(np.array([1.0, 2.0, 3.0]), max_lag=2) returns ndarray of shape (3,), float64.
      ACF of constant signal [c]*N at lag 0 equals N*c² (un-normalized).
    status: verified
    last_checked: 2026-05-13
  - id: ac-002
    summary: acf_fft rejects max_lag >= data length
    type: code
    pass_when: |
      acf_fft(np.array([1.0, 2.0]), max_lag=3) raises ValueError with message containing "max_lag".
    status: verified
    last_checked: 2026-05-13
  - id: ac-003
    summary: acf_fft does not mutate input array
    type: runtime
    pass_when: |
      Input array before/after acf_fft is elementwise equal (np.allclose).
    status: verified
    last_checked: 2026-05-13
  - id: ac-004
    summary: apply_window returns new array with correct shape
    type: code
    pass_when: |
      apply_window(np.ones((5, 3)), "hann", axis=0) returns (5, 3) where each column equals Hann coefficients.
      apply_window(np.ones((5, 3)), "blackman", axis=1) returns (5, 3) where each row equals Blackman coefficients.
    status: verified
    last_checked: 2026-05-13
  - id: ac-005
    summary: apply_window rejects unknown window_type
    type: code
    pass_when: |
      apply_window(np.ones(5), "hamming", axis=0) raises ValueError.
    status: verified
    last_checked: 2026-05-13
  - id: ac-006
    summary: apply_window does not mutate input array
    type: runtime
    pass_when: |
      Input array before/after apply_window is elementwise equal.
    status: verified
    last_checked: 2026-05-13
  - id: ac-007
    summary: frequency_grid returns correct angular frequencies
    type: scientific
    pass_when: |
      frequency_grid(8, 0.5) returns length 5, first=0.0, last=π/0.5=2π, spacing=π/2.
    status: pending
  - id: ac-008
    summary: frequency_grid minimal case (n_fft=2)
    type: code
    pass_when: |
      frequency_grid(2, 1.0) returns [0.0, π].
    status: verified
    last_checked: 2026-05-13
  - id: ac-009
    summary: signal module imports without error
    type: runtime
    pass_when: |
      import molrs; from molrs import signal — no ImportError.
    status: verified
    last_checked: 2026-05-13
  - id: ac-010
    summary: Rust crate compiles without warning
    type: code
    pass_when: |
      cargo build -p molrs-signal emits zero warnings, zero errors.
    status: verified
    last_checked: 2026-05-13
  - id: ac-011
    summary: signal feature flags work in facade
    type: code
    pass_when: |
      Default features: molrs::signal not available. Feature "signal": molrs::signal available, all 3 functions callable from Rust.
    status: verified
    last_checked: 2026-05-16
