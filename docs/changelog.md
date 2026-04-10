```{eval-rst}
.. currentmodule:: pytket.extensions.quantinuum
```

# Changelog

## 0.58.1 (April 2026)

- Updates to documentation

## 0.58.0 (February 2026)

- Update pytket version requirement to 2.13.0.
- Update pytket-pecos version requirement to 0.3.2.
- Update quantum-pecos to 0.8.0.dev3.

## 0.57.0 (January 2026)

- Update pytket version requirement to 2.11.0.

## 0.56.2 (December 2025)

- fix to quantum-pecos to 0.7.0.dev4

## 0.56.1 (December 2025)

- Remove all methods providing remote access to Quantinuum devices. (Please use
  `qnexus` instead.)

## 0.55.3 (April 2026)

- Changes to this version are not available in 0.56.1 and higher
- Update pytket version requirement to 2.16.0.
- Update pytket-qir version requirement to 1.0.0.
- Update pytket-pecos version requirement to 0.3.2.
- Update quantum-pecos version requirement to 0.8.0.dev8.
- Update pytket-phir version requirement to 0.10.1.
- Set scratch_reg_resize_pass max_size to 63 in the default passes

## 0.55.2 (December 2025)

- fix to quantum-pecos to 0.7.0.dev4

## 0.55.1 (November 2025)

- Add static method `pass_from_info` to `QuantinuumBackend` so that we can compile directly from a `BackendInfo`.
- Remove references to H1 except in offline API; update offline machine data.
- Update pytket-pecos version requirement to 0.2.2.

## 0.54.0 (October 2025)

- Fix request_options for batch submissions
- Update pytket version requirement to 2.9.3.
- Update pytket-qir version requirement to 0.25.

## 0.53.0 (August 2025)

- Update pytket version requirement to 2.9.1.
- Update pytket-pecos version requirement to 0.2.1.
- Allow 64-bit classical registers.

## 0.52.0 (July 2025)

- Add optional argument `request_raw_results` to `QuantinuumAPI.retrieve_job_status`.

## 0.51.0 (July 2025)

- Update list of accepted op types for validity check.
- Update pytket minimum version requirement to 2.7.0.

## 0.50.0 (June 2025)

- Update pytket-qir minimum version requirement to 0.24.1.
- Emit deprecation warning from submit_program().
- Add optional `data` argument to `QuantinuumBackend` constructor to allow
  offline usage (as compilation engine or local emulator) when device data are
  known.
- Update pytket minimum version requirement to 2.6.0.

## 0.48.0 (May 2025)

- Update pytket minimum version requirement to 2.4.1.
- Update pytket-qir minimum version requirement to 0.23.
- Add `start-date` to fields returned as part of circuit status.

## 0.47.0 (May 2025)

- Add `max_cost` parameter to {py:meth}`~.QuantinuumBackend.process_circuits` and `submit_program()`.

## 0.46.0 (April 2025)

- Update pytket minimum version requirement to 2.2.0.
- Update pytket-qir minimum version requirement to 0.22.

## 0.45.0 (March 2025)

- Add {py:class}`~pytket.predicates.CliffordCircuitPredicate` for circuit compilation and processing
  when using the stabilizer simulator.
- Update pytket version requirement to 2.0.1.
- Update pytket-qir version requirement to 0.21.
- Update pytket-pecos version requirement to 0.2.0.

## 0.44.0 (February 2025)

- Remove `noise_specs` from local-emulator backends.
- Updated pytket version requirement to 1.39.
- Updated pytket-qir version requirement to 0.20.

## 0.43.0 (January 2025)

- Check for language support when submitting programs.
- Remove all `Phase` operations from circuit when compiling for backend.
- Updated pytket version requirement to 1.39.

## 0.42.0 (December 2024)

- Updated pytket version requirement to 1.37.
- Updated pytket-pecos version requirement to 0.1.32.

## 0.41.0 (November 2024)

- Add optimisation level 3 that uses {py:meth}`~pytket.passes.GreedyPauliSimp`.

## 0.40.0 (November 2024)

- Update pytket-qir version requirement to 0.17.
- Updated pytket version requirement to 1.34.
- Updated pytket-pecos version requirement to 0.1.31.
- Allow circuits containing `OpType.ClExpr` operations.

## 0.39.0 (November 2024)

- Use QIR by default for program submission.
- Update pytket-qir version requirement to 0.16.
- Add new `kwarg` `n_leakage_detection_qubits` to {py:meth}`~.QuantinuumBackend.process_circuits`
- Fix `allow_implicit_swaps` configuration not handled correctly in default passes.
- Add handling for negative results

## 0.38.1 (October 2024)

- Query device information only when needed to avoid unnecessary API calls.
- Update minimum versions of some dependencies (requests, websockets).

## 0.38.0 (October 2024)

- Updated pytket version requirement to 1.33.
- Fix handling of unused bits in local emulator.
- Update machine specs in offline API.
- Update pytket-qir version requirement to 0.13.
- Add language option for profile compatible QIR

## 0.37.0 (August 2024)

- Determine maximum classical register width from backend info.
- Permit numpy 2.
- Update pytket_pecos version requirement to 0.1.29.
- Updated pytket version requirement to 1.31.

## 0.36.0 (July 2024)

- Updated pytket version requirement to 1.30.
- Update pytket-qir version requirement to 0.12.

## 0.35.0 (June 2024)

- Update pytket version requirement to 1.29.
- Update pytket_pecos version requirement to 0.1.28.

## 0.34.1 (June 2024)

- Restrict classical registers to a maximum size of 32 (until pytket can
  support larger values).

## 0.34.0 (June 2024)

- Update pytket_pecos version requirement to 0.1.27.
- Update Leakage Detection to reuse circuit qubits.
- Update pytket version requirement to 1.28.
- Update pytket-qir version requirement to 0.11.
- Update offline machine specs to match real devices as at 5 June 2024.

## 0.33.0 (April 2024)

- Updated pytket version requirement to 1.27.
- Update pytket_pecos version requirement to 0.1.24.

## 0.32.0 (March 2024)

- Remove `no_opt` and `allow_2q_gate_rebase` options to
  {py:meth}`~.QuantinuumBackend.process_circuits` and `submit_program()`, and assume that the submitted
  circuit is exactly what is desired to be run.
- Update pytket_pecos version requirement to 0.1.22.

## 0.31.0 (March 2024)

- Updated pytket version requirement to 1.26.
- Update pytket_pecos version requirement to 0.1.19.
- Add methods to enable visibility of Quantinuum H-Series

operations calendar with and without matplotlib. \* Support TK2 as native gate. \* Update pytket version requirement to 1.26. \* Update pytket-qir version requirement to 0.9.

## 0.30.0 (February 2024)

- Make pytket-qir an automatic dependency.
- Update pytket version requirement to 1.25.
- Update pytket-qir version requirement to 0.5.
- Update pytket_pecos version requirement to 0.1.17.

## 0.29.0 (January 2024)

- Updated pytket_pecos version requirement to 0.1.13.
- Fix handling of results in local emulator with non-default classical
  registers.
- Add WASM support to local emulators.
- Add multithreading support to local emulators, via new `multithreading`
  keyword argument passed to {py:meth}`~.QuantinuumBackend.process_circuits`.

## 0.28.0 (January 2024)

- Updated pytket version requirement to 1.24.
- Python 3.12 support added, 3.9 dropped.
- pytket_pecos dependency updated to 0.1.9.

## 0.27.0 (January 2024)

- Updated pytket version requirement to 1.23.
- `QuantinuumBackend.cost()` now raises an error if the `syntax_checker`
  argument doesn't correspond to the device's reported syntax checker or if it
  specifies a device that isn't a syntax checker; and the method returns 0 if
  called on syntax-checker backends.
- Add partial support for local emulator backends, if installed with the
  `pecos` option.

## 0.26.0 (November 2023)

- Updated pytket version requirement to 1.22.
- Add `QuantinuumConfigCredentialStorage` for caching API tokens in local pytket
  configuration file.
- Add an additonal {py:meth}`~pytket.passes.RemoveRedundancies` pass to the default passes for levels 1 and 2 to remove Rz gates before measurement.

## 0.25.0 (October 2023)

- Updated pytket version requirement to 1.21.

## 0.24.0 (October 2023)

- Don't include {py:meth}`~pytket.passes.SimplifyInitial` in default passes; instead make it an option
  to {py:meth}`~.QuantinuumBackend.process_circuits`.
- Fix: set default two-qubit gate when compilation config is provided without
  specifying one.

## 0.23.0 (September 2023)

- Update pytket-qir version requirement to 0.3.
- Update pytket version requirement to 1.20.

## 0.22.0 (September 2023)

- Update pytket version requirement to 1.19.

## 0.21.0 (September 2023)

- Ensure results retrieval works even with an old-format {py:class}`~pytket.backends.resulthandle.ResultHandle`
  (generated by a pre-0.17.0 version of pytket-quantinuum).
- Add properties {py:attr}`~.QuantinuumBackend.default_two_qubit_gate` and
  {py:attr}`~.QuantinuumBackend.two_qubit_gate_set` providing the default and supported
  two-qubit gates for a device.
- Make `ZZPhase` the default two-qubit gate target on all devices.
- Add {py:class}`~.QuantinuumBackendCompilationConfig` dataclass, which can be passed as
  an optional argument when constructing a {py:class}`~.QuantinuumBackend`. Configuration
  can be inspected using {py:meth}`~.QuantinuumBackend.get_compilation_config` and
  modified using the methods
  {py:meth}`~.QuantinuumBackend.set_compilation_config_allow_implicit_swaps` and
  {py:meth}`~.QuantinuumBackend.set_compilation_config_target_2qb_gate`.
- Add optional argument `allow_2q_gate_rebase` argument to
  {py:meth}`~pytket.backends.backend.Backend.process_circuit`, {py:meth}`~.QuantinuumBackend.process_circuits` and `submit_program()` to
  allow the backend to rebase to rebase the circuit to a different two-qubit
  gate judged to have better fidelity before being run. The default is to not
  allow this.
- Fix handling of multiple classical registers when submitting QIR.
- Change {py:class}`~pytket.backends.resulthandle.ResultHandle` format. (Old {py:class}`~pytket.backends.resulthandle.ResultHandle` objects will continue to
  work after upgrading.)
- Fix: Ignore erased scratch bits when constructing {py:class}`~pytket.backends.resulthandle.ResultHandle`.

## 0.20.0 (August 2023)

- Update pytket version requirement to 1.18.
- Add `implicit_swaps` option to
  {py:attr}`~.QuantinuumBackend.rebase_pass`, which
  can use implicit wire swaps (represented in the circuit qubit permutation)
  to help implement some gates when chosen. Defaults to `False`.
- Add `implicit_swaps` option to
  {py:attr}`~.QuantinuumBackend.default_compilation_pass`, which
  is used in the rebase step. Defaults to `True`.

## 0.19.0 (August 2023)

- Update {py:class}`~pytket.architecture.FullyConnected` Architecture to label Node with "q", matching
  compilation by {py:meth}`~pytket.passes.FlattenRelabelRegistersPass`.

## 0.18.0 (July 2023)

- Update pytket version requirement to 1.17.
- Add `leakage_detection` option to {py:meth}`~.QuantinuumBackend.process_circuits`
  that automatically modifies Circuits with ancillas for detecting leakage
  errors. Also provides a new method `prune_shots_detected_as_leaky` for
  removing erroneous shots from {py:class}`~pytket.backends.backendresult.BackendResult`.

## 0.17.0 (June 2023)

- Add {py:class}`~.Language` enum to control language used for circuit submission, with
  values `Language.QASM` and `Language.QIR`.
- Renamed `QuantinuumBackend.submit_qasm()` to
  `submit_program()`, with a `language` argument.
- Add a `language` kwarg to {py:meth}`~.QuantinuumBackend.process_circuits`,
  defaulting to `Language.QASM`. (Support for `Language.QIR` is
  experimental and its use is not recommended; a warning will be emitted. You
  must install the `pytket-qir` package separately in order to use this
  feature.)
- Use "q" instead of "node" as the name of the single qubit register in compiled
  circuits.
- Updated pytket version requirement to 1.16.

## 0.16.0 (May 2023)

- Updated pytket version requirement to 1.15.
- cost function now takes the same kwargs as process_circuits
- add check for the number of classical registers to the backend
- add `get_partial_result()` method to {py:class}`~.QuantinuumBackend`.
- add `Rxxyyzz` gate support.

## 0.15.0 (April 2023)

- Darkmode added to the documentation
- Updated pytket version requirement to 1.13.2
- Default compilation passes updated to correctly track initial and final maps during compilation

## 0.14.0 (March 2023)

- Use default `Node` register for flattening in default compilation pass.
- Prefer `ZZPhase` to `ZZMax` gates if available.
- Updated pytket version requirement to 1.13.

## 0.13.0 (January 2023)

- Drop support for Python 3.8; add support for 3.11.
- The backend now works in threads other than the main.
- Updated pytket version requirement to 1.11.

## 0.12.0 (December 2022)

- Updated pytket version requirement to 1.10.
- Default compilation pass update to flatten registers

## 0.11.0 (November 2022)

- Updated pytket version requirement to 1.9.
- Add optional `no_opt` argument to {py:meth}`~.QuantinuumBackend.process_circuits` and
  `submit_qasm()`, requesting no optimization.
- Change default optimization level in
  {py:attr}`~.QuantinuumBackend.default_compilation_pass` to 2.
- {py:attr}`~.QuantinuumBackend.default_compilation_pass` now flattens qubit registers when compiling Circuits.

## 0.10.0 (November 2022)

- Break up `pytket` internal scratch registers if their widths exceed limit.
- Updated pytket version requirement to 1.8.

## 0.9.0 (October 2022)

- Add `session` parameter to {py:class}`~.QuantinuumAPI`. Creates a new session
  if `None` is provided.
- Add facility to specify default `options` paramater to
  {py:meth}`~.QuantinuumBackend.process_circuits` and `submit_qasm()` when constructing backend, and
  include this information in {py:attr}`~.QuantinuumBackend.backend_info`.
- Updated pytket version requirement to 1.7.

## 0.8.0 (September 2022)

- Add `options` parameter to {py:meth}`~.QuantinuumBackend.process_circuits` and `submit_qasm()`.
- Updated pytket version requirement to 1.6.

## 0.7.0 (August 2022)

- Add new {py:class}`~.QuantinuumAPIOffline` for allowing usage of the backend without API calls.
- New `api_handler` parameter for {py:class}`~.QuantinuumBackend`, allowing to choose
  online or offline options. Default value is the standard online api.
- Updated pytket version requirement to 1.5.

## 0.6.0 (July 2022)

- Changed batching interface: {py:meth}`~.QuantinuumBackend.process_circuits` no longer batches, use
  `start_batching` and `add_to_batch` methods to explicitly start and append to
  batches.
- New `submit_qasm` backend method to enable direct submission of a QASM program.

## 0.5.0 (July 2022)

- Updated pytket version requirement to 1.4.
- Add support for multi-factor authentication and microsoft federated login.

## 0.4.0 (June 2022)

- Add wasm support
- Add support for `OpType.CopyBits` and `OpType.ClassicalExpBox` in {py:class}`~.QuantinuumBackend`
- Updated pytket version requirement to 1.3.
- Add optional argument `group` to {py:class}`~.QuantinuumBackend`

## 0.3.1 (May 2022)

- Updated to pyjwt 2.4. This fixes a potential security vulnerability
  (CVE-2022-29217).

## 0.3.0 (May 2022)

- `QuantinuumBackend.cost_estimate` deprecated, new `QuantinuumBackend.cost()`
  method now uses the syntax checker devices to directly return the cost.
- Updated pytket version requirement to 1.2.

## 0.2.0 (April 2022)

- Updated pytket version requirement to 1.1.

## 0.1.2 (April 2022)

- Fix batch handling in {py:meth}`~.QuantinuumBackend.process_circuits`.

## 0.1.1 (March 2022)

- Update device names.

## 0.1.0 (March 2022)

- Module renamed from `pytket.extensions.honeywell` to
  `pytket.extensions.quantinumm`, with corresponding name changes throughout.
- Simplify authentication: use `QuantinuumBackend.login()` to log in once per session.
- Updated pytket version requirement to 1.0.

Old changelog for "pytket-honeywell":

### 0.21.0 (February 2022)

- Updated pytket version requirement to 0.19.
- Drop support for Python 3.7; add support for 3.10.

### 0.20.0 (January 2022)

- Added optional `group` field to circuit submission.

### 0.19.0 (January 2022)

- Updated pytket version requirement to 0.18.

### 0.18.0 (November 2021)

- Updated pytket version requirement to 0.17.

### 0.17.0 (October 2021)

- Updated pytket version requirement to 0.16.
- Renamed `HoneywellBackend.available_devices` to `_available_devices` so as
  not to conflict with abstract {py:class}`~pytket.backends.backend.Backend` method.

### 0.16.0 (September 2021)

- Updated pytket version requirement to 0.15.

### 0.15.0 (September 2021)

- Updated pytket version requirement to 0.14.

### 0.14.0 (August 2021)

- Support new Honeywell simulator options in `HoneywellBackend`:
  "simulator" for simulator type, and "noisy_simulation" to toggle simulations
  with and without error models.
- Device name no longer optional on `HoneywellBackend` construction.

### 0.13.0 (July 2021)

- Updated pytket version requirement to 0.13.

### 0.12.0 (June 2021)

- Updated pytket version requirement to 0.12.

### 0.11.0 (May 2021)

- Updated pytket version requirement to 0.11.

### 0.10.0 (May 2021)

- Contextual optimisation added to default compilation passes (except at optimisation level 0).
