# Copyright Quantinuum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import os
import time
from base64 import b64encode
from collections import Counter
from collections.abc import Callable  # pylint: disable=unused-import
from pathlib import Path
from typing import Any, cast

import hypothesis.strategies as st
import numpy as np
import pyqir
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.strategies._internal import SearchStrategy
from pytket.backends import CircuitNotValidError
from pytket.backends.status import StatusEnum
from pytket.circuit import (
    Bit,
    Circuit,
    Conditional,
    Node,
    OpType,
    Qubit,
    if_not_bit,
    reg_eq,
    reg_geq,
    reg_gt,
    reg_leq,
    reg_lt,
    reg_neq,
)
from pytket.circuit.clexpr import wired_clexpr_from_logic_exp
from pytket.passes import BasePass, SequencePass
from pytket.passes.resizeregpass import _gen_scratch_transformation
from pytket.predicates import CompilationUnit
from pytket.wasm import WasmFileHandler

from pytket.extensions.quantinuum import (
    Language,
    QuantinuumBackend,
    QuantinuumBackendCompilationConfig,
    prune_shots_detected_as_leaky,
)
from pytket.extensions.quantinuum.backends.api_wrappers import (
    QuantinuumAPI,
    QuantinuumAPIError,
)
from pytket.extensions.quantinuum.backends.quantinuum import _ALL_GATES, MAX_C_REG_WIDTH

skip_remote_tests: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS") is None
skip_remote_tests_prod: bool = os.getenv("PYTKET_RUN_REMOTE_TESTS_PROD") is None
skip_mpl_tests: bool = os.getenv("PYTKET_RUN_MPL_TESTS") is None

REASON = (
    "PYTKET_RUN_REMOTE_TESTS not set (requires configuration of Quantinuum username)"
)

REASON_PROD = "PYTKET_RUN_REMOTE_TESTS_PROD not set \
(requires configuration of Quantinuum username)"

REASON_MPL = "PYTKET_RUN_MPL_TESTS not set \
(requires configuration of Quantinuum username)"


def prog_from_qir_text(qir: str) -> str:
    bitcode = pyqir.Module.from_ir(pyqir.Context(), qir).bitcode
    return b64encode(bitcode).decode("utf-8")


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("authenticated_quum_backend_qa", [None], indirect=True)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR, Language.PQIR])
@pytest.mark.timeout(120)
def test_quantinuum(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    backend = authenticated_quum_backend_qa
    c = Circuit(4, 4, "test 1")
    c.H(0)
    c.CX(0, 1)
    c.Rz(0.3, 2)
    c.CSWAP(0, 1, 2)
    c.CRz(0.4, 2, 3)
    c.CY(1, 3)
    c.ZZPhase(0.1, 2, 0)
    c.Tdg(3)
    c.measure_all()
    c = backend.get_compiled_circuit(c)
    n_shots = 4
    handle = backend.process_circuits([c], n_shots)[0]
    correct_shots = np.zeros((4, 4))
    correct_counts = {(0, 0, 0, 0): 4}
    res = backend.get_result(handle, timeout=49)
    shots = res.get_shots()
    counts = res.get_counts()
    assert backend.circuit_status(handle).status is StatusEnum.COMPLETED
    assert np.all(shots == correct_shots)
    assert counts == correct_counts
    res = backend.run_circuit(c, n_shots=4, timeout=49, language=language)
    newshots = res.get_shots()
    assert np.all(newshots == correct_shots)
    newcounts = res.get_counts()
    assert newcounts == correct_counts
    if skip_remote_tests:
        assert backend.backend_info is None


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1SC"}], indirect=True
)
@pytest.mark.timeout(120)
def test_max_classical_register(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    backend = authenticated_quum_backend_qa

    info = backend.backend_info
    assert info is not None
    n_cl_reg = info.n_cl_reg
    assert isinstance(n_cl_reg, int)

    c = Circuit(4, 4, "test 1")
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = backend.get_compiled_circuit(c)
    assert backend._check_all_circuits([c])  # noqa: SLF001
    for i in range(n_cl_reg - 1):
        c.add_c_register(f"creg-{i}", 32)

    assert backend._check_all_circuits([c])  # noqa: SLF001

    c.add_c_register("creg-extra", 32)

    with pytest.raises(CircuitNotValidError):
        backend._check_all_circuits([c])  # noqa: SLF001


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1SC"}], indirect=True
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR, Language.PQIR])
@pytest.mark.timeout(120)
def test_bell(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    b = authenticated_quum_backend_qa
    c = Circuit(2, 2, "test 2")
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 10
    shots = b.run_circuit(c, n_shots=n_shots, language=language).get_shots()
    assert all(q[0] == q[1] for q in shots)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": "H2-1SC", "label": "test 3"}],
    indirect=True,
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR, Language.PQIR])
@pytest.mark.timeout(120)
def test_multireg(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    b = authenticated_quum_backend_qa
    c = Circuit()
    q1 = Qubit("q1", 0)
    q2 = Qubit("q2", 0)
    c1 = Bit("c1", 0)
    c2 = Bit("c2", 0)
    for q in (q1, q2):
        c.add_qubit(q)
    for cb in (c1, c2):
        c.add_bit(cb)
    c.H(q1)
    c.CX(q1, q2)
    c.Measure(q1, c1)
    c.Measure(q2, c2)
    c = b.get_compiled_circuit(c)

    n_shots = 10
    shots = b.run_circuit(c, n_shots=n_shots, language=language).get_shots()
    assert np.array_equal(shots, np.zeros((10, 2)))


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.timeout(120)
def test_default_pass(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    b = authenticated_quum_backend_qa
    for ol in range(3):
        comp_pass = b.default_compilation_pass(ol)
        c = Circuit(3, 3)
        q0 = Qubit("test0", 5)
        q1 = Qubit("test1", 6)
        c.add_qubit(q0)
        c.H(q0)
        c.H(0)
        c.CX(0, 1)
        c.CSWAP(1, 0, 2)
        c.ZZPhase(0.84, 2, 0)
        c.measure_all()
        c.add_qubit(q1)
        cu = CompilationUnit(c)
        comp_pass.apply(cu)
        # 5 qubits added to Circuit, one is removed when flattening registers
        assert cu.circuit.qubits == [
            Node("q", 0),
            Node("q", 1),
            Node("q", 2),
            Node("q", 3),
        ]
        assert cu.initial_map[Qubit(0)] == Node("q", 0)
        assert cu.initial_map[Qubit(1)] == Node("q", 1)
        assert cu.initial_map[Qubit(2)] == Node("q", 2)
        assert cu.initial_map[q0] == Node("q", 3)
        assert cu.initial_map[q1] == q1
        for pred in b.required_predicates:
            assert pred.verify(cu.circuit)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [
        {"device_name": name, "label": "test cancel"}
        for name in pytest.ALL_SIMULATOR_NAMES  # type: ignore
    ],
    indirect=True,
)
@pytest.mark.timeout(120)
def test_cancel(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    b = authenticated_quum_backend_qa
    c = Circuit(2, 2).H(0).CX(0, 1).measure_all()
    c = b.get_compiled_circuit(c)
    handle = b.process_circuit(c, 10)
    try:
        # will raise HTTP error if job is already completed
        b.cancel(handle)
        time.sleep(1.0)
        assert b.circuit_status(handle).status in [StatusEnum.CANCELLED]  # noqa: FURB171
    except QuantinuumAPIError as err:
        check_completed = "job has completed already" in str(err)
        assert check_completed
        if not check_completed:
            raise err


@st.composite
def circuits(
    draw: Callable[[SearchStrategy[Any]], Any],
    n_qubits: SearchStrategy[int] = st.integers(min_value=2, max_value=6),  # noqa: B008
    depth: SearchStrategy[int] = st.integers(min_value=1, max_value=100),  # noqa: B008
) -> Circuit:
    total_qubits = draw(n_qubits)
    circuit = Circuit(total_qubits, total_qubits)
    for _ in range(draw(depth)):
        gate = draw(st.sampled_from(list(_ALL_GATES)))
        control = draw(st.integers(min_value=0, max_value=total_qubits - 1))
        if gate == OpType.ZZMax:
            target = draw(
                st.integers(min_value=0, max_value=total_qubits - 1).filter(
                    lambda x: x != control  # noqa: B023
                )
            )
            circuit.add_gate(gate, [control, target])
        elif gate == OpType.Measure:
            circuit.add_gate(gate, [control, control])
            circuit.add_gate(OpType.Reset, [control])
        elif gate == OpType.Rz:
            param = draw(st.floats(min_value=0, max_value=2))
            circuit.add_gate(gate, [param], [control])
        elif gate == OpType.PhasedX:
            param1 = draw(st.floats(min_value=0, max_value=2))
            param2 = draw(st.floats(min_value=0, max_value=2))
            circuit.add_gate(gate, [param1, param2], [control])
    circuit.measure_all()

    return circuit


@pytest.mark.skipif(skip_remote_tests_prod, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_prod",
    [
        {"device_name": name}
        for name in [
            *pytest.ALL_QUANTUM_HARDWARE_NAMES,  # type: ignore
            *pytest.ALL_SYNTAX_CHECKER_NAMES,  # type: ignore
        ]
    ],
    indirect=True,
)
@given(
    c=circuits(),  # pylint: disable=no-value-for-parameter
    n_shots=st.integers(min_value=1, max_value=10000),
)
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@pytest.mark.timeout(120)
def test_cost_estimate(
    authenticated_quum_backend_prod: QuantinuumBackend,
    c: Circuit,
    n_shots: int,
) -> None:
    b = authenticated_quum_backend_prod
    c = b.get_compiled_circuit(c)
    estimate = None
    if b._device_name.endswith("SC"):  # noqa: SLF001
        estimate = b.cost(c, n_shots)
        assert estimate == 0.0
    else:
        # All other real hardware backends should have the
        # "syntax_checker" misc property set, so there should be no
        # need of providing it explicitly.
        estimate = b.cost(c, n_shots)
        if estimate is None:
            pytest.skip("API is flaky, sometimes returns None unexpectedly.")
        assert isinstance(estimate, float)
        assert estimate > 0.0


@pytest.mark.skipif(skip_remote_tests_prod, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_prod",
    [
        {"device_name": name}
        for name in [
            *pytest.ALL_QUANTUM_HARDWARE_NAMES,  # type: ignore
        ]
    ],
    indirect=True,
)
@pytest.mark.timeout(120)
def test_cost_estimate_wrong_syntax_checker(
    authenticated_quum_backend_prod: QuantinuumBackend,
) -> None:
    b = authenticated_quum_backend_prod
    c = Circuit(1).PhasedX(0.5, 0.5, 0).measure_all()
    with pytest.raises(ValueError):
        _ = b.cost(c, 10, syntax_checker="H6-2SC")


@pytest.mark.skipif(skip_remote_tests_prod, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_prod", [{"device_name": "H2-1E"}], indirect=True
)
@pytest.mark.timeout(120)
def test_cost_estimate_bad_syntax_checker(
    authenticated_quum_backend_prod: QuantinuumBackend,
) -> None:
    b = authenticated_quum_backend_prod
    c = Circuit(1).PhasedX(0.5, 0.5, 0).measure_all()
    with pytest.raises(ValueError):
        _ = b.cost(c, 10, syntax_checker="H2-1E")


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        Language.QIR,
        Language.PQIR,
    ],
)
@pytest.mark.timeout(120)
def test_classical(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    # circuit to cover capabilities covered in example notebook
    c = Circuit(1, name="test_classical")
    a = c.add_c_register("a", 8)
    b = c.add_c_register("b", 10)
    d = c.add_c_register("d", 10)

    c.add_c_setbits([True], [a[0]])
    c.add_c_setbits([False, True] + [False] * 6, a)  # type: ignore
    c.add_c_setbits([True, True] + [False] * 8, b)  # type: ignore

    c.add_c_setreg(23, a)
    c.add_c_copyreg(a, b)

    c.add_clexpr(*wired_clexpr_from_logic_exp(a + b, d.to_list()))
    c.add_clexpr(*wired_clexpr_from_logic_exp(a - b, d.to_list()))
    c.add_clexpr(*wired_clexpr_from_logic_exp(a << 1, a.to_list()))
    c.add_clexpr(*wired_clexpr_from_logic_exp(a >> 1, b.to_list()))

    c.X(0, condition=reg_eq(a ^ b, 1))
    c.X(0, condition=(a[0] ^ b[0]))
    c.X(0, condition=reg_eq(a & b, 1))
    c.X(0, condition=reg_eq(a | b, 1))

    c.X(0, condition=a[0])
    c.X(0, condition=reg_neq(a, 1))
    c.X(0, condition=if_not_bit(a[0]))
    c.X(0, condition=reg_gt(a, 1))
    c.X(0, condition=reg_lt(a, 1))
    c.X(0, condition=reg_geq(a, 1))
    c.X(0, condition=reg_leq(a, 1))
    c.Phase(0, condition=a[0])

    c.measure_all()

    backend = authenticated_quum_backend_qa

    c = backend.get_compiled_circuit(c)
    assert backend.run_circuit(c, n_shots=10, language=language).get_counts()


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.parametrize(
    "language",
    [
        Language.QIR,
        Language.PQIR,
        pytest.param(
            Language.QASM,
            marks=pytest.mark.xfail(reason="https://github.com/CQCL/tket/issues/1173"),
        ),
    ],
)
@pytest.mark.timeout(120)
def test_division(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    c = Circuit()
    a = c.add_c_register("a", 8)
    b = c.add_c_register("b", 10)
    d = c.add_c_register("d", 10)

    c.add_c_setbits([False, True] + [False] * 6, a)  # type: ignore
    c.add_c_setbits([True, True] + [False] * 8, b)  # type: ignore
    c.add_clexpr(*wired_clexpr_from_logic_exp(a * b // d, d.to_list()))

    backend = authenticated_quum_backend_qa

    c = backend.get_compiled_circuit(c)
    with pytest.raises(ValueError):
        backend.run_circuit(c, n_shots=10, language=language).get_counts()


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR, Language.PQIR])
@pytest.mark.timeout(120)
def test_postprocess(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    b = authenticated_quum_backend_qa
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.add_gate(OpType.PhasedX, [1, 1], [0])
    c.add_gate(OpType.PhasedX, [1, 1], [1])
    c.add_gate(OpType.ZZMax, [0, 1])
    c.measure_all()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, n_shots=10, postprocess=True, language=language)
    ppcirc = Circuit.from_dict(json.loads(cast("str", h[1])))
    ppcmds = ppcirc.get_commands()
    assert len(ppcmds) > 0
    assert all(ppcmd.op.type == OpType.ClassicalTransform for ppcmd in ppcmds)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.timeout(120)
def test_leakage_detection(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    b = authenticated_quum_backend_qa
    c = Circuit(2, 2).H(0).CZ(0, 1).Measure(0, 0).Measure(1, 1)

    with pytest.raises(ValueError):
        b.process_circuit(
            c, n_shots=10, leakage_detection=True, n_leakage_detection_qubits=1000
        )
    h = b.process_circuit(c, n_shots=10, leakage_detection=True)
    r = b.get_result(h)
    assert len(r.c_bits) == 4
    assert sum(r.get_counts().values()) == 10
    r_discarded = prune_shots_detected_as_leaky(r)
    assert len(r_discarded.c_bits) == 2
    assert sum(r_discarded.get_counts().values()) == 10


@given(
    n_shots=st.integers(min_value=1, max_value=10),
    n_bits=st.integers(min_value=0, max_value=10),
)
@pytest.mark.timeout(120)
def test_shots_bits_edgecases(n_shots: int, n_bits: int) -> None:
    quantinuum_backend = QuantinuumBackend("H2-1SC", machine_debug=True)
    c = Circuit(n_bits, n_bits)

    # TODO TKET-813 add more shot based backends and move to integration tests
    h = quantinuum_backend.process_circuit(c, n_shots)
    res = quantinuum_backend.get_result(h)

    correct_shots = np.zeros((n_shots, n_bits), dtype=int)
    correct_shape = (n_shots, n_bits)
    correct_counts = Counter({(0,) * n_bits: n_shots})
    # BackendResult
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts

    # Direct
    res = quantinuum_backend.run_circuit(c, n_shots=n_shots)
    assert np.array_equal(res.get_shots(), correct_shots)
    assert res.get_shots().shape == correct_shape
    assert res.get_counts() == correct_counts


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1E"}], indirect=True
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR, Language.PQIR])
@pytest.mark.timeout(200)
def test_simulator(
    authenticated_quum_handler: QuantinuumAPI,
    authenticated_quum_backend_qa: QuantinuumBackend,
    language: Language,
) -> None:
    circ = Circuit(2, name="sim_test").H(0).CX(0, 1).measure_all()
    n_shots = 1000
    state_backend = authenticated_quum_backend_qa
    stabilizer_backend = QuantinuumBackend(
        "H2-1E", simulator="stabilizer", api_handler=authenticated_quum_handler
    )

    circ = state_backend.get_compiled_circuit(circ)

    noisy_handle = state_backend.process_circuit(circ, n_shots, language=language)
    pure_handle = state_backend.process_circuit(
        circ, n_shots, noisy_simulation=False, language=language
    )
    stab_handle = stabilizer_backend.process_circuit(
        circ, n_shots, noisy_simulation=False, language=language
    )

    noisy_counts = state_backend.get_result(noisy_handle).get_counts()
    assert sum(noisy_counts.values()) == n_shots
    assert len(noisy_counts) > 2  # some noisy results likely

    pure_counts = state_backend.get_result(pure_handle).get_counts()
    assert sum(pure_counts.values()) == n_shots
    assert len(pure_counts) == 2

    stab_counts = stabilizer_backend.get_result(stab_handle).get_counts()
    assert sum(stab_counts.values()) == n_shots
    assert len(stab_counts) == 2

    # test non-clifford circuit fails predicate check
    # when run on stabilizer backend
    non_stab_circ = (
        Circuit(2, name="non_stab_circ").H(0).Rx(0.1, 0).CX(0, 1).measure_all()
    )
    non_stab_circ = stabilizer_backend.get_compiled_circuit(non_stab_circ)

    with pytest.raises(CircuitNotValidError):
        _ = stabilizer_backend.process_circuit(
            non_stab_circ, n_shots, noisy_simulation=False, language=language
        )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.timeout(120)
def test_retrieve_available_devices(
    authenticated_quum_backend_qa: QuantinuumBackend,
    authenticated_quum_handler: QuantinuumAPI,
) -> None:
    # authenticated_quum_backend_qa still needs a handler or it will
    # attempt to use the DEFAULT_API_HANDLER.
    backend_infos = authenticated_quum_backend_qa.available_devices(
        api_handler=authenticated_quum_handler
    )
    assert len(backend_infos) > 0
    assert all(
        {OpType.TK2, OpType.ZZMax, OpType.ZZPhase} & backend_info.gate_set
        for backend_info in backend_infos
    )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1E"}], indirect=True
)
@pytest.mark.timeout(120)
def test_batching(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    circ = Circuit(2, name="batching_test").H(0).CX(0, 1).measure_all()
    state_backend = authenticated_quum_backend_qa
    circ = state_backend.get_compiled_circuit(circ)
    # test batch can be resumed

    h1 = state_backend.start_batch(500, circ, 10)
    h2 = state_backend.add_to_batch(h1, circ, 10)
    h3 = state_backend.add_to_batch(h1, circ, 10, batch_end=True)

    assert state_backend.get_results([h1, h2, h3])


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR, Language.PQIR])
@pytest.mark.timeout(120)
def test_submission_with_group(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    b = authenticated_quum_backend_qa
    c = Circuit(2, 2, "test 2")
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 10
    shots = b.run_circuit(
        c,
        n_shots=n_shots,
        group=os.getenv("PYTKET_REMOTE_QUANTINUUM_GROUP", default="DEFAULT"),
        language=language,
    ).get_shots()
    assert all(q[0] == q[1] for q in shots)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1SC"}], indirect=True
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR, Language.PQIR])
@pytest.mark.timeout(120)
def test_zzphase(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    backend = authenticated_quum_backend_qa
    c = Circuit(2, 2, "test rzz")
    c.H(0)
    c.CX(0, 1)
    c.Rz(0.3, 0)
    c.CY(0, 1)
    c.ZZPhase(0.1, 1, 0)
    c.measure_all()
    c0 = backend.get_compiled_circuit(c, 0)

    assert c0.n_gates_of_type(backend.default_two_qubit_gate) > 0

    n_shots = 4
    handle = backend.process_circuits([c0], n_shots, language=language)[0]
    correct_counts = {(0, 0): 4}
    res = backend.get_result(handle, timeout=49)
    counts = res.get_counts()
    assert counts == correct_counts

    c = Circuit(2, 2, "test_rzz_1")
    c.H(0).H(1)
    c.ZZPhase(1, 1, 0)
    c.H(0).H(1)
    c1 = backend.get_compiled_circuit(c, 1)
    assert c1.n_gates_of_type(OpType.ZZPhase) == 0


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.timeout(120)
def test_zzphase_support_opti2(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    backend = authenticated_quum_backend_qa
    c = Circuit(3, 3, "test rzz synthesis")
    c.H(0)
    c.CX(0, 2)
    c.Rz(0.2, 2)
    c.CX(0, 2)
    c.measure_all()
    c0 = backend.get_compiled_circuit(c, 2)

    assert c0.n_gates_of_type(backend.default_two_qubit_gate) == 1


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.timeout(120)
def test_prefer_zzphase(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    # We should prefer small-angle ZZPhase to alternative ZZMax decompositions
    backend = authenticated_quum_backend_qa
    c = (
        Circuit(2)
        .H(0)
        .H(1)
        .ZZPhase(0.1, 0, 1)
        .Rx(0.2, 0)
        .Ry(0.3, 1)
        .ZZPhase(0.1, 0, 1)
        .H(0)
        .H(1)
        .measure_all()
    )
    c0 = backend.get_compiled_circuit(c)
    if backend.default_two_qubit_gate == OpType.ZZPhase:
        assert c0.n_gates_of_type(OpType.ZZPhase) == 2
    elif backend.default_two_qubit_gate == OpType.ZZMax:
        assert c0.n_gates_of_type(OpType.ZZMax) == 2
    else:
        assert backend.default_two_qubit_gate == OpType.TK2
        assert c0.n_gates_of_type(OpType.TK2) == 1


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("device_name", pytest.ALL_DEVICE_NAMES)  # type: ignore
@pytest.mark.timeout(120)
def test_device_state(
    device_name: str, authenticated_quum_handler: QuantinuumAPI
) -> None:
    assert isinstance(
        QuantinuumBackend.device_state(
            device_name, api_handler=authenticated_quum_handler
        ),
        str,
    )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1SC"}], indirect=True
)
@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        Language.QIR,
        Language.PQIR,
    ],
)
@pytest.mark.timeout(120)
def test_wasm_qa(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    wasfile = WasmFileHandler(str(Path(__file__).parent.parent / "wasm" / "add1.wasm"))
    c = Circuit(1)
    c.name = "test_wasm"
    a = c.add_c_register("a", 8)
    c.add_wasm_to_reg("add_one", wasfile, [a], [a])
    c.measure_all()

    b = authenticated_quum_backend_qa

    c = b.get_compiled_circuit(c)
    h = b.process_circuits(
        [c], n_shots=10, wasm_file_handler=wasfile, language=language
    )[0]

    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10


@pytest.mark.skipif(skip_remote_tests_prod, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_prod", [{"device_name": "H2-1SC"}], indirect=True
)
@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        Language.QIR,
        Language.PQIR,
    ],
)
@pytest.mark.timeout(120)
def test_wasm(
    authenticated_quum_backend_prod: QuantinuumBackend, language: Language
) -> None:
    wasfile = WasmFileHandler(str(Path(__file__).parent.parent / "wasm" / "add1.wasm"))
    c = Circuit(1, 1)
    c.name = "test_wasm"
    a = c.add_c_register("a", 8)
    c.add_wasm_to_reg("add_one", wasfile, [a], [a])
    c.measure_all()

    b = authenticated_quum_backend_prod

    c = b.get_compiled_circuit(c)
    h = b.process_circuits(
        [c], n_shots=10, wasm_file_handler=wasfile, language=language
    )[0]

    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1E"}], indirect=True
)
@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        Language.QIR,
        Language.PQIR,
    ],
)
@pytest.mark.timeout(120)
def test_wasm_costs(
    authenticated_quum_backend_qa: QuantinuumBackend,
    language: Language,
) -> None:
    wasfile = WasmFileHandler(str(Path(__file__).parent.parent / "wasm" / "add1.wasm"))
    c = Circuit(1)
    c.name = "test_wasm"
    a = c.add_c_register("a", 8)
    c.add_wasm_to_reg("add_one", wasfile, [a], [a])
    c.measure_all()

    b = authenticated_quum_backend_qa

    c = b.get_compiled_circuit(c)
    costs = b.cost(
        c,
        n_shots=10,
        syntax_checker="H2-1SC",
        wasm_file_handler=wasfile,
        language=language,
    )
    if costs is None:
        pytest.skip("API is flaky, sometimes returns None unexpectedly.")
    assert isinstance(costs, float)
    assert costs > 0.0


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.timeout(120)
def test_submit_qasm(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    qasm = """
    OPENQASM 2.0;
    include "hqslib1.inc";

    qreg q[2];
    creg c[2];
    U1q(0.5*pi,0.5*pi) q[0];
    measure q[0] -> c[0];
    if(c[0]==1) rz(1.5*pi) q[0];
    if(c[0]==1) rz(0.0*pi) q[1];
    if(c[0]==1) U1q(3.5*pi,0.5*pi) q[1];
    if(c[0]==1) ZZ q[0],q[1];
    if(c[0]==1) rz(3.5*pi) q[1];
    if(c[0]==1) U1q(3.5*pi,1.5*pi) q[1];
    """

    b = authenticated_quum_backend_qa
    h = b.submit_program(Language.QASM, qasm, n_shots=10)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR, Language.PQIR])
@pytest.mark.timeout(120)
def test_options(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    # Unrecognized options are ignored
    c0 = Circuit(1).H(0).measure_all()
    b = authenticated_quum_backend_qa
    c = b.get_compiled_circuit(c0, 0)
    h = b.process_circuits([c], n_shots=1, options={"ignoreme": 0}, language=language)
    r = b.get_results(h)[0]
    shots = r.get_shots()
    assert len(shots) == 1
    assert len(shots[0]) == 1


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": name} for name in pytest.ALL_SYNTAX_CHECKER_NAMES],  # type: ignore
    indirect=True,
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR, Language.PQIR])
@pytest.mark.timeout(120)
def test_tk2(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    c0 = (
        Circuit(2)
        .XXPhase(0.1, 0, 1)
        .YYPhase(0.2, 0, 1)
        .ZZPhase(0.3, 0, 1)
        .measure_all()
    )
    b = authenticated_quum_backend_qa
    b.set_compilation_config_target_2qb_gate(OpType.TK2)
    c = b.get_compiled_circuit(c0, 2)
    h = b.process_circuit(c, n_shots=1, language=language)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 1
    assert len(shots[0]) == 2


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1SC"}], indirect=True
)
@pytest.mark.timeout(120)
def test_qir_submission(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    b = authenticated_quum_backend_qa

    with open("integration/qir/qat-link_2.ll") as f:
        qir = f.read()

    h = b.submit_program(Language.QIR, prog_from_qir_text(qir), n_shots=10)
    r = b.get_result(h)
    assert set(r.get_bitlist()) == set([Bit("0_t0", 0), Bit("0_t1", 0)])  # noqa: C405
    assert len(r.get_shots()) == 10


@pytest.mark.skipif(skip_remote_tests_prod, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_prod", [{"device_name": "H2-1SC"}], indirect=True
)
@pytest.mark.timeout(120)
def test_qir_entrypoints(authenticated_quum_backend_prod: QuantinuumBackend) -> None:
    b = authenticated_quum_backend_prod

    with open("integration/qir/qat-link.ll") as f:
        qir = f.read()

    h = b.submit_program(Language.QIR, prog_from_qir_text(qir), n_shots=10)
    r = b.get_result(h)
    assert set(r.get_bitlist()) == set([Bit("0_t0", 0), Bit("0_t1", 0)])  # noqa: C405
    assert len(r.get_shots()) == 10


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1SC"}], indirect=True
)
@pytest.mark.timeout(120)
def test_qir_entrypoints_qa(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    b = authenticated_quum_backend_qa

    with open("integration/qir/qat-link.ll") as f:
        qir = f.read()

    h = b.submit_program(Language.QIR, prog_from_qir_text(qir), n_shots=10)
    r = b.get_result(h)
    assert set(r.get_bitlist()) == set([Bit("0_t0", 0), Bit("0_t1", 0)])  # noqa: C405
    assert len(r.get_shots()) == 10


@pytest.mark.skipif(skip_remote_tests_prod, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_prod", [{"device_name": "H2-1SC"}], indirect=True
)
@pytest.mark.timeout(120)
def test_qir_submission_mz_to_reg(
    authenticated_quum_backend_prod: QuantinuumBackend,
) -> None:
    b = authenticated_quum_backend_prod
    with open("integration/qir/test_pytket_qir_6.ll") as f:
        qir = f.read()

    h = b.submit_program(Language.QIR, prog_from_qir_text(qir), n_shots=10)

    r = b.get_result(h)

    assert len(r.get_shots()) == 10
    assert len(r.get_bitlist()) == 20


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1SC"}], indirect=True
)
@pytest.mark.timeout(120)
def test_qir_submission_mz_to_reg_qa(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    b = authenticated_quum_backend_qa
    with open("integration/qir/test_pytket_qir_6.ll") as f:
        qir = f.read()

    h = b.submit_program(Language.QIR, prog_from_qir_text(qir), n_shots=10)
    r = b.get_result(h)
    assert len(r.get_shots()) == 10
    assert len(r.get_bitlist()) == 20


@pytest.mark.parametrize(
    "filename",
    [
        "test_pytket_qir_wasm_5-QIRProfile.ADAPTIVE.ll",
        "test_pytket_qir_wasm_5-QIRProfile.PYTKET.ll",
    ],
)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1E"}], indirect=True
)
@pytest.mark.timeout(120)
def test_qir_submission_64bitwasm5_qa(
    authenticated_quum_backend_qa: QuantinuumBackend, filename: str
) -> None:
    b = authenticated_quum_backend_qa
    with open(f"integration/qir/{filename}") as f:
        qir = f.read()

    wfh = WasmFileHandler(str(Path(__file__).parent.parent / "wasm" / "add1.wasm"))

    h = b.submit_program(
        Language.QIR, prog_from_qir_text(qir), n_shots=10, wasm_file_handler=wfh
    )
    r = b.get_result(h)
    assert len(r.get_shots()) == 10
    # 192 = 3 * 64 , 102 = 64 + 32 + 6
    assert len(r.get_bitlist()) in [102, 192]


@pytest.mark.parametrize(
    "filename",
    [
        "test_pytket_qir_wasm_6-QIRProfile.ADAPTIVE.ll",
        "test_pytket_qir_wasm_6-QIRProfile.PYTKET.ll",
    ],
)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1E"}], indirect=True
)
@pytest.mark.timeout(120)
def test_qir_submission_64bitwasm6_qa(
    authenticated_quum_backend_qa: QuantinuumBackend, filename: str
) -> None:
    b = authenticated_quum_backend_qa
    with open(f"integration/qir/{filename}") as f:
        qir = f.read()

    wfh = WasmFileHandler(str(Path(__file__).parent.parent / "wasm" / "add1.wasm"))

    h = b.submit_program(
        Language.QIR, prog_from_qir_text(qir), n_shots=10, wasm_file_handler=wfh
    )
    with pytest.raises(ValueError):
        # this circuit gives a negative results which we reject
        b.get_result(h)


@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        Language.QIR,
        Language.PQIR,
    ],
)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1E"}], indirect=True
)
@pytest.mark.timeout(120)
def test_qir_submission_64bitwasm_pytket_qa(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    b = authenticated_quum_backend_qa

    wfh = WasmFileHandler(
        str(Path(__file__).parent.parent / "wasm" / "add1.wasm"),
        int_size=32,
        check_file=False,
    )

    c = Circuit(6, 6)
    c.Measure(Qubit(0), Bit(0))
    c.Measure(Qubit(1), Bit(1))
    c.Measure(Qubit(2), Bit(2))
    c0 = c.add_c_register("c0", 63)
    c1 = c.add_c_register("c1", 32)
    c.add_c_setbits([True] + [False] * 62, list(c0))
    c.add_wasm_to_reg("add_one", wfh, [c0], [c1])

    h = b.process_circuit(c, n_shots=10, language=language, wasm_file_handler=wfh)

    r = b.get_result(h)
    assert len(r.get_shots()) == 10
    assert len(r.get_bitlist()) == 101  # 63 + 32 + 6


@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        Language.QIR,
        Language.PQIR,
    ],
)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1E"}], indirect=True
)
@pytest.mark.timeout(120)
def test_qir_submission_64bitwasm_pytket_overflow_qa(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    b = authenticated_quum_backend_qa

    wfh = WasmFileHandler(
        str(Path(__file__).parent.parent / "wasm" / "add1.wasm"),
        int_size=32,
        check_file=False,
    )

    c = Circuit(6, 6)
    c.Measure(Qubit(0), Bit(0))
    c.Measure(Qubit(1), Bit(1))
    c.Measure(Qubit(2), Bit(2))
    c0 = c.add_c_register("c0", 63)
    c1 = c.add_c_register("c1", 32)
    c.add_c_setbits([False] * 29 + [True, True, True, True] + [False] * 30, list(c0))
    c.add_wasm_to_reg("add_one", wfh, [c0], [c1])

    h = b.process_circuit(c, n_shots=10, language=language, wasm_file_handler=wfh)
    with pytest.raises(ValueError):
        # this circuit gives a negative results which we reject
        b.get_result(h)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1SC"}], indirect=True
)
@pytest.mark.timeout(120)
def test_qir_conversion(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    c0 = Circuit(2).H(0).CX(0, 1).measure_all()
    b = authenticated_quum_backend_qa
    c = b.get_compiled_circuit(c0)
    h = b.process_circuit(c, n_shots=10, language=Language.PQIR)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10
    assert all(len(shot) == 2 for shot in shots)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.timeout(120)
def test_old_handle(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    # https://github.com/CQCL/pytket-quantinuum/issues/189
    c = Circuit(2, 2, "test")
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    b0 = QuantinuumBackend(
        "H2-1SC",
        api_handler=QuantinuumAPI(  # type: ignore # pylint: disable=unexpected-keyword-arg
            api_url="https://qapi.quantinuum.com/",
            _QuantinuumAPI__user_name=os.getenv("PYTKET_REMOTE_QUANTINUUM_USERNAME"),
            _QuantinuumAPI__pwd=os.getenv("PYTKET_REMOTE_QUANTINUUM_PASSWORD"),
        ),
    )
    c = b0.get_compiled_circuit(c)
    h0 = b0.process_circuit(c, n_shots=10)
    r0 = b0.get_result(h0)
    b1 = QuantinuumBackend(
        "H2-1SC",
        api_handler=QuantinuumAPI(  # type: ignore # pylint: disable=unexpected-keyword-arg
            api_url="https://qapi.quantinuum.com/",
            _QuantinuumAPI__user_name=os.getenv("PYTKET_REMOTE_QUANTINUUM_USERNAME"),
            _QuantinuumAPI__pwd=os.getenv("PYTKET_REMOTE_QUANTINUUM_PASSWORD"),
        ),
    )
    h1 = h0[:2]
    r1 = b1.get_result(h1)  # type: ignore
    assert (r0.get_shots() == r1.get_shots()).all()


@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        Language.QIR,
        Language.PQIR,
    ],
)
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1SC"}], indirect=True
)
@pytest.mark.timeout(120)
def test_scratch_removal(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    # https://github.com/CQCL/pytket-quantinuum/issues/213
    c = Circuit()
    qb0 = c.add_q_register("qb0", 3)
    qb1 = c.add_q_register("qb1", 1)
    cb0 = c.add_c_register("cb0", 2)
    cb1 = c.add_c_register("cb1", 3)

    c.add_gate(OpType.Reset, qb1)  # type:ignore
    c.CX(qb0[0], qb1[0])
    c.CX(qb0[1], qb1[0])
    c.Measure(qb1[0], cb0[0])
    c.add_gate(OpType.Reset, qb1)  # type:ignore
    c.CX(qb0[1], qb1[0])
    c.CX(qb0[2], qb1[0])
    c.Measure(qb1[0], cb0[1])
    c.X(qb0[0], condition=reg_eq(cb0, 1))
    c.X(qb0[2], condition=reg_eq(cb0, 2))
    c.X(qb0[1], condition=reg_eq(cb0, 3))
    c.Measure(qb0[0], cb1[0])
    c.Measure(qb0[1], cb1[1])
    c.Measure(qb0[2], cb1[2])

    b = authenticated_quum_backend_qa
    c1 = b.get_compiled_circuit(c, optimisation_level=1)
    h = b.process_circuit(c1, n_shots=3, language=language)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 3
    assert all(len(shot) == 5 for shot in shots)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1E"}], indirect=True
)
@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        Language.QIR,
        Language.PQIR,
    ],
)
@pytest.mark.timeout(120)
def test_wasm_collatz(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    wasmfile = WasmFileHandler(
        str(Path(__file__).parent.parent / "wasm" / "collatz.wasm")
    )
    c = Circuit(8)
    a = c.add_c_register("a", 8)
    b = c.add_c_register("b", 8)

    # Use Hadamards to set "a" register to a random value.
    for i in range(8):
        c.H(i)
        c.Measure(Qubit(i), Bit("a", i))
    # Compute the value of the Collatz function on this value.
    c.add_wasm_to_reg("collatz", wasmfile, [a], [b])

    backend = authenticated_quum_backend_qa

    c = backend.get_compiled_circuit(c)
    h = backend.process_circuit(
        c, n_shots=10, wasm_file_handler=wasmfile, language=language
    )

    r = backend.get_result(h)
    shots = r.get_shots()

    def to_int(C: np.ndarray) -> int:
        assert len(C) == 8
        return sum(pow(2, i) * int(C[i]) for i in range(8))

    def collatz(n: int) -> int:
        if n == 0:
            return 0
        m = 0
        while n != 1:
            n = (3 * n + 1) // 2 if n % 2 == 1 else n // 2
            m += 1
        return m

    for shot in shots:
        n, m = to_int(shot[:8]), to_int(shot[8:16])
        assert collatz(n) == m


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1E"}], indirect=True
)
@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        Language.QIR,
        Language.PQIR,
    ],
)
@pytest.mark.timeout(120)
def test_wasm_state(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    wasmfile = WasmFileHandler(
        str(Path(__file__).parent.parent / "wasm" / "state.wasm")
    )
    c = Circuit(8)
    a = c.add_c_register("a", 8).to_list()  # measurement results
    b = c.add_c_register("b", 4)  # final count
    s = c.add_c_register("s", 1)  # scratch bit

    # Use Hadamards to set "a" register to random values.
    for i in range(8):
        c.H(i)
        c.Measure(Qubit(i), a[i])
    # Count the number of 1s in the "a" register and store in the "b" register.
    c.add_wasm_to_reg("set_c", wasmfile, [s], [])  # set c to zero
    for i in range(8):
        # Copy a[i] to s
        c.add_c_copybits([a[i]], [Bit("s", 0)])
        # Conditionally increment the counter
        c.add_wasm_to_reg("conditional_increment_c", wasmfile, [s], [])
    # Put the counter into "b"
    c.add_wasm_to_reg("get_c", wasmfile, [], [b])

    backend = authenticated_quum_backend_qa

    c = backend.get_compiled_circuit(c)
    h = backend.process_circuit(
        c, n_shots=10, wasm_file_handler=wasmfile, language=language
    )

    r = backend.get_result(h)
    shots = r.get_shots()

    def to_int(C: np.ndarray) -> int:
        assert len(C) == 4
        return sum(pow(2, i) * C[i] for i in range(4))

    for shot in shots:
        a_count = sum(shot[:8])
        b_count = to_int(shot[8:12])
        assert a_count == b_count


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.timeout(120)
def test_default_2q_gate(authenticated_quum_handler: QuantinuumAPI) -> None:
    # https://github.com/CQCL/pytket-quantinuum/issues/250
    config = QuantinuumBackendCompilationConfig(allow_implicit_swaps=False)
    b = QuantinuumBackend(
        "H2-1E", api_handler=authenticated_quum_handler, compilation_config=config
    )
    c = Circuit(2).H(0).CX(0, 1).measure_all()
    c1 = b.get_compiled_circuit(c)
    assert any(cmd.op.type == b.default_two_qubit_gate for cmd in c1)


# https://github.com/CQCL/pytket-quantinuum/issues/265
def test_Rz_removal_before_measurements() -> None:
    backend = QuantinuumBackend("H2-1E", machine_debug=True)
    # Circuit will contain an Rz gate if RemoveRedundancies
    #  isn't applied after SquashRzPhasedX
    circuit = Circuit(2).H(0).Rz(0.75, 0).CX(0, 1).measure_all()

    for optimisation_level in (1, 2):
        compiled_circuit = backend.get_compiled_circuit(
            circuit, optimisation_level=optimisation_level
        )
        assert backend.valid_circuit(compiled_circuit)
        assert compiled_circuit.n_gates_of_type(OpType.Rz) == 0


# https://github.com/CQCL/pytket-quantinuum/issues/263
@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1E"}], indirect=True
)
@pytest.mark.parametrize("language", [Language.QASM, Language.QIR, Language.PQIR])
@pytest.mark.timeout(120)
def test_noiseless_emulation(
    authenticated_quum_backend_qa: QuantinuumBackend, language: Language
) -> None:
    backend = authenticated_quum_backend_qa
    c = Circuit(2).H(0).CX(0, 1).measure_all()
    c1 = backend.get_compiled_circuit(c)
    h = backend.process_circuit(
        c1, n_shots=100, language=language, noisy_simulation=False
    )
    r = backend.get_result(h)
    counts = r.get_counts()
    assert all(x0 == x1 for x0, x1 in counts)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.timeout(120)
def test_get_calendar(
    authenticated_quum_handler: QuantinuumAPI,
) -> None:
    machine = "deadhead" if "hqapi" in authenticated_quum_handler.url else "H2-1"

    backend = QuantinuumBackend(
        api_handler=authenticated_quum_handler, device_name=machine
    )
    start_date = datetime.datetime(2024, 1, 8)  # noqa: DTZ001
    end_date = datetime.datetime(2024, 2, 16)  # noqa: DTZ001
    calendar_data = backend.get_calendar(start_date, end_date)
    assert all(isinstance(a, dict) for a in calendar_data)
    assert all(
        isinstance(a.get(attr), datetime.datetime)
        for a in calendar_data
        for attr in ["start-date", "end-date"]
    )


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize("authenticated_quum_backend_qa", [None], indirect=True)
@pytest.mark.timeout(120)
def test_get_calendar_raises_error(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    backend = authenticated_quum_backend_qa
    start_date = datetime.datetime(2024, 2, 8)  # noqa: DTZ001
    end_date = datetime.datetime(2024, 2, 16)  # noqa: DTZ001
    with pytest.raises(RuntimeError):
        backend.get_calendar(start_date, end_date)


@pytest.mark.skipif(skip_mpl_tests, reason=REASON_MPL)
@pytest.mark.timeout(120)
@pytest.mark.mpl_image_compare(tolerance=20)
def test_view_calendar(authenticated_quum_handler: QuantinuumAPI) -> Any:
    backend = QuantinuumBackend(
        api_handler=authenticated_quum_handler, device_name="H2-1"
    )
    return backend.view_calendar(month=2, year=2024)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1E"}], indirect=True
)
def test_optimisation_level_3_compilation(
    authenticated_quum_backend_qa: QuantinuumBackend,
) -> None:
    b = authenticated_quum_backend_qa

    c = Circuit(6)
    c.add_barrier([0, 1, 2, 3, 4, 5])
    for _ in range(6):
        for i in range(4):
            for j in range(i + 1, 4):
                c.CX(i, j)
                c.Rz(0.23, j)
                c.S(j)
            c.H(i)

    compiled_2 = b.get_compiled_circuit(c, 2)
    compiled_3 = b.get_compiled_circuit(c, 3)

    assert compiled_2.n_2qb_gates() == 36
    assert compiled_2.n_gates == 98
    assert compiled_2.depth() == 45
    assert compiled_3.n_2qb_gates() == 31
    assert compiled_3.n_gates == 93
    assert compiled_3.depth() == 49


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa", [{"device_name": "H2-1E"}], indirect=True
)
def test_no_phase_ops(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    b = authenticated_quum_backend_qa

    c = (
        Circuit(3, 3)
        .H(0)
        .Measure(0, 0)
        .H(1)
        .CX(1, 2, condition_bits=[0], condition_value=1)
        .Measure(1, 1)
        .Measure(2, 2)
    )
    c1 = b.get_compiled_circuit(c)
    for cmd in c1.get_commands():
        op = cmd.op
        typ = op.type
        assert typ != OpType.Phase
        if typ == OpType.Conditional:
            assert isinstance(op, Conditional)
            assert op.op.type != OpType.Phase


def test_default_pass_serialization() -> None:
    h11e_backend = QuantinuumBackend("H2-1E", machine_debug=True)

    for opt_level in range(4):
        default_pass = h11e_backend.default_compilation_pass(opt_level)
        original_pass_dict = default_pass.to_dict()
        reconstructed_pass = BasePass.from_dict(
            original_pass_dict,
            {"resize scratch bits": _gen_scratch_transformation(MAX_C_REG_WIDTH)},
        )
        assert isinstance(reconstructed_pass, SequencePass)
        assert original_pass_dict == reconstructed_pass.to_dict()


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": "H2-1E"}, {"device_name": "H2-1SC"}],
    indirect=True,
)
def test_rng(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    circ = Circuit(1, 1)
    seed = circ.add_c_register("seed", 64)
    bound = circ.add_c_register("bound", 32)
    index = circ.add_c_register("index", 32)
    num0 = circ.add_c_register("num0", 32)
    num1 = circ.add_c_register("num1", 32)
    num2 = circ.add_c_register("num2", 32)
    seed_copy = circ.add_c_register("seed_copy", 64)
    circ.add_c_setbits([True, True, True], [seed[1], seed[3], seed[5]])  # seed = 42
    circ.set_rng_seed(seed)
    circ.set_rng_index(index)  # index = 0
    circ.add_c_setbits([True, True], [bound[0], bound[2]])  # bound = 5
    circ.set_rng_bound(bound)
    circ.get_rng_num(num0)
    circ.add_c_setbits([False], [bound[2]])  # bound = 1
    circ.get_rng_num(num1)
    circ.add_c_setbits([True], [bound[3]])  # bound = 9
    circ.get_rng_num(num2)
    # num0[3] should be 0, so the PhasedX should never be applied:
    circ.PhasedX(0.5, 0.5, 0, condition_bits=[num0[3]], condition_value=1)
    circ.Measure(Qubit(0), Bit(0))
    circ.add_c_copyreg(seed, seed_copy)

    b = authenticated_quum_backend_qa
    n_shots = 10

    if b._device_name.endswith("SC"):  # noqa: SLF001
        estimate = b.cost(circ, n_shots)
        if estimate is not None:  # API is flaky, sometimes returns None
            assert isinstance(estimate, float)
    else:
        h = b.process_circuit(circ, n_shots=n_shots, language=Language.QASM)
        r = b.get_result(h)
        counts = r.get_counts()
        assert len(counts) == 1  # circuit is deterministic
        bits = next(iter(counts.keys()))
        num0_startbit = circ.bits.index(Bit("num0", 0))
        num1_startbit = circ.bits.index(Bit("num1", 0))
        num2_startbit = circ.bits.index(Bit("num2", 0))
        num0_val = sum(bits[num0_startbit + i] * pow(2, i) for i in range(32))
        num1_val = sum(bits[num1_startbit + i] * pow(2, i) for i in range(32))
        num2_val = sum(bits[num2_startbit + i] * pow(2, i) for i in range(32))
        seed_copy_startbit = circ.bits.index(Bit("seed_copy", 0))
        seed_copy_val = sum(bits[seed_copy_startbit + i] * pow(2, i) for i in range(64))
        assert num0_val <= 5
        assert num1_val <= 1
        assert num2_val <= 9
        assert seed_copy_val == 42


@pytest.mark.skipif(skip_remote_tests_prod, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_prod", [{"device_name": "H2-1SC"}], indirect=True
)
@pytest.mark.parametrize(
    "language",
    [
        Language.QASM,
        Language.QIR,
        Language.PQIR,
    ],
)
@pytest.mark.timeout(120)
def test_rng_prod(
    authenticated_quum_backend_prod: QuantinuumBackend, language: Language
) -> None:
    circ = Circuit(1, 1)
    seed = circ.add_c_register("seed", 64)
    bound = circ.add_c_register("bound", 32)
    index = circ.add_c_register("index", 32)
    num0 = circ.add_c_register("num0", 32)
    num1 = circ.add_c_register("num1", 32)
    num2 = circ.add_c_register("num2", 32)
    seed_copy = circ.add_c_register("seed_copy", 64)
    circ.add_c_setbits([True, True, True], [seed[1], seed[3], seed[5]])  # seed = 42
    circ.set_rng_seed(seed)
    circ.set_rng_index(index)  # index = 0
    circ.add_c_setbits([True, True], [bound[0], bound[2]])  # bound = 5
    circ.set_rng_bound(bound)
    circ.get_rng_num(num0)
    circ.add_c_setbits([False], [bound[2]])  # bound = 1
    circ.get_rng_num(num1)
    circ.add_c_setbits([True], [bound[3]])  # bound = 9
    circ.get_rng_num(num2)
    # num0[3] should be 0, so the PhasedX should never be applied:
    circ.PhasedX(0.5, 0.5, 0, condition_bits=[num0[3]], condition_value=1)
    circ.Measure(Qubit(0), Bit(0))
    circ.add_c_copyreg(seed, seed_copy)

    b = authenticated_quum_backend_prod
    n_shots = 10

    if b._device_name.endswith("SC"):  # noqa: SLF001
        estimate = b.cost(circ, n_shots)
        if estimate is not None:  # API is flaky, sometimes returns None
            assert isinstance(estimate, float)


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.parametrize(
    "authenticated_quum_backend_qa",
    [{"device_name": "H2-1E"}, {"device_name": "H2-1SC"}],
    indirect=True,
)
def test_shotnum(authenticated_quum_backend_qa: QuantinuumBackend) -> None:
    circ = Circuit(1).PhasedX(0.5, 0.5, 0).measure_all()
    shotnum = circ.add_c_register("shotnum", 32)
    circ.get_job_shot_num(shotnum)

    b = authenticated_quum_backend_qa
    n_shots = 10

    if b._device_name.endswith("SC"):  # noqa: SLF001
        estimate = b.cost(circ, n_shots)
        if estimate is not None:  # API is flaky, sometimes returns None
            assert isinstance(estimate, float)
    else:
        h = b.process_circuit(circ, n_shots=n_shots, language=Language.QASM)
        r = b.get_result(h)
        counts = r.get_counts()
        assert len(counts) == n_shots  # all results should be different
        shotnum_startbit = circ.bits.index(Bit("shotnum", 0))
        shotnums = {
            sum(res[shotnum_startbit + i] * pow(2, i) for i in range(32))
            for res in counts
        }
        assert shotnums == set(range(1, n_shots + 1))


@pytest.mark.skipif(skip_remote_tests, reason=REASON)
@pytest.mark.timeout(120)
def test_pass_from_info(
    authenticated_quum_handler: QuantinuumAPI,
) -> None:
    infos = QuantinuumBackend.available_devices(api_handler=authenticated_quum_handler)
    for info in infos[:5]:
        actual_pass = QuantinuumBackend.pass_from_info(info)

        be = QuantinuumBackend(
            device_name=info.device_name or "", api_handler=authenticated_quum_handler
        )
        expected_pass = be.default_compilation_pass()

        assert actual_pass.to_dict() == expected_pass.to_dict()
