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

import json
import os
from collections import Counter
from collections.abc import Callable  # pylint: disable=unused-import
from pathlib import Path
from typing import Any, cast

import hypothesis.strategies as st
import numpy as np
import pytest
from hypothesis import given
from hypothesis.strategies._internal import SearchStrategy

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
from pytket.extensions.quantinuum import (
    QuantinuumBackend,
    QuantinuumBackendCompilationConfig,
    have_pecos,
    prune_shots_detected_as_leaky,
)
from pytket.extensions.quantinuum.backends.quantinuum import _ALL_GATES, MAX_C_REG_WIDTH
from pytket.passes import BasePass, SequencePass
from pytket.passes.resizeregpass import _gen_scratch_transformation
from pytket.predicates import CompilationUnit
from pytket.wasm import WasmFileHandler


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_bell() -> None:
    b = QuantinuumBackend("H2-1LE")
    c = Circuit(2, 2, "test 2")
    c.H(0)
    c.CX(0, 1)
    c.measure_all()
    c = b.get_compiled_circuit(c)
    n_shots = 10
    shots = b.run_circuit(c, n_shots=n_shots).get_shots()
    assert all(q[0] == q[1] for q in shots)


def test_default_pass() -> None:
    b = QuantinuumBackend("H2-1")
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


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_classical() -> None:
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

    backend = QuantinuumBackend("H2-1LE")

    c = backend.get_compiled_circuit(c)
    assert backend.run_circuit(c, n_shots=10).get_counts()


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_postprocess() -> None:
    b = QuantinuumBackend("H2-1LE")
    assert b.supports_contextual_optimisation
    c = Circuit(2, 2)
    c.add_gate(OpType.PhasedX, [1, 1], [0])
    c.add_gate(OpType.PhasedX, [1, 1], [1])
    c.add_gate(OpType.ZZMax, [0, 1])
    c.measure_all()
    c = b.get_compiled_circuit(c)
    h = b.process_circuit(c, n_shots=10, postprocess=True)
    ppcirc = Circuit.from_dict(json.loads(cast("str", h[1])))
    ppcmds = ppcirc.get_commands()
    assert len(ppcmds) > 0
    assert all(ppcmd.op.type == OpType.ClassicalTransform for ppcmd in ppcmds)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_leakage_detection() -> None:
    b = QuantinuumBackend("H2-1LE")
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
def test_shots_bits_edgecases(n_shots: int, n_bits: int) -> None:
    quantinuum_backend = QuantinuumBackend("H2-1LE", machine_debug=True)
    c = Circuit(n_bits, n_bits)

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


def test_retrieve_available_devices() -> None:
    backend_infos = QuantinuumBackend.available_devices()
    assert len(backend_infos) > 0
    assert all(
        {OpType.TK2, OpType.ZZMax, OpType.ZZPhase} & backend_info.gate_set
        for backend_info in backend_infos
    )


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_submission_with_group() -> None:
    b = QuantinuumBackend("H2-1LE")
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
    ).get_shots()
    assert all(q[0] == q[1] for q in shots)


def test_zzphase_support_opti2() -> None:
    backend = QuantinuumBackend("H2-1")
    c = Circuit(3, 3, "test rzz synthesis")
    c.H(0)
    c.CX(0, 2)
    c.Rz(0.2, 2)
    c.CX(0, 2)
    c.measure_all()
    c0 = backend.get_compiled_circuit(c, 2)

    assert c0.n_gates_of_type(backend.default_two_qubit_gate) == 1


def test_prefer_zzphase() -> None:
    # We should prefer small-angle ZZPhase to alternative ZZMax decompositions
    backend = QuantinuumBackend("H2-1")
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


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_wasm_qa() -> None:
    wasfile = WasmFileHandler(str(Path(__file__).parent.parent / "wasm" / "add1.wasm"))
    c = Circuit(1)
    c.name = "test_wasm"
    a = c.add_c_register("a", 8)
    c.add_wasm_to_reg("add_one", wasfile, [a], [a])
    c.measure_all()

    b = QuantinuumBackend("H2-1LE")

    c = b.get_compiled_circuit(c)
    h = b.process_circuits([c], n_shots=10, wasm_file_handler=wasfile)[0]

    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_wasm() -> None:
    wasfile = WasmFileHandler(str(Path(__file__).parent.parent / "wasm" / "add1.wasm"))
    c = Circuit(1, 1)
    c.name = "test_wasm"
    a = c.add_c_register("a", 8)
    c.add_wasm_to_reg("add_one", wasfile, [a], [a])
    c.measure_all()

    b = QuantinuumBackend("H2-1LE")

    c = b.get_compiled_circuit(c)
    h = b.process_circuits([c], n_shots=10, wasm_file_handler=wasfile)[0]

    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 10


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_options() -> None:
    # Unrecognized options are ignored
    c0 = Circuit(1).H(0).measure_all()
    b = QuantinuumBackend("H2-1LE")
    c = b.get_compiled_circuit(c0, 0)
    h = b.process_circuits([c], n_shots=1, options={"ignoreme": 0})
    r = b.get_results(h)[0]
    shots = r.get_shots()
    assert len(shots) == 1
    assert len(shots[0]) == 1


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_tk2() -> None:
    c0 = (
        Circuit(2)
        .XXPhase(0.1, 0, 1)
        .YYPhase(0.2, 0, 1)
        .ZZPhase(0.3, 0, 1)
        .measure_all()
    )
    b = QuantinuumBackend("H2-1LE")
    b.set_compilation_config_target_2qb_gate(OpType.TK2)
    c = b.get_compiled_circuit(c0, 2)
    h = b.process_circuit(c, n_shots=1)
    r = b.get_result(h)
    shots = r.get_shots()
    assert len(shots) == 1
    assert len(shots[0]) == 2


@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_wasm_collatz() -> None:
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

    backend = QuantinuumBackend("H2-1LE")

    c = backend.get_compiled_circuit(c)
    h = backend.process_circuit(c, n_shots=10, wasm_file_handler=wasmfile)

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


# FIXME: Bug in pecos?
@pytest.mark.xfail
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_wasm_state() -> None:
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

    backend = QuantinuumBackend("H2-1LE")

    c = backend.get_compiled_circuit(c)
    h = backend.process_circuit(c, n_shots=10, wasm_file_handler=wasmfile)

    r = backend.get_result(h)
    shots = r.get_shots()

    def to_int(C: np.ndarray) -> int:
        assert len(C) == 4
        return sum(pow(2, i) * C[i] for i in range(4))

    for shot in shots:
        a_count = sum(shot[:8])
        b_count = to_int(shot[8:12])
        assert a_count == b_count


def test_default_2q_gate() -> None:
    # https://github.com/Quantinuum/pytket-quantinuum/issues/250
    config = QuantinuumBackendCompilationConfig(allow_implicit_swaps=False)
    b = QuantinuumBackend("H2-1", compilation_config=config)
    c = Circuit(2).H(0).CX(0, 1).measure_all()
    c1 = b.get_compiled_circuit(c)
    assert any(cmd.op.type == b.default_two_qubit_gate for cmd in c1)


# https://github.com/Quantinuum/pytket-quantinuum/issues/265
def test_Rz_removal_before_measurements() -> None:
    backend = QuantinuumBackend("H2-1", machine_debug=True)
    # Circuit will contain an Rz gate if RemoveRedundancies
    #  isn't applied after SquashRzPhasedX
    circuit = Circuit(2).H(0).Rz(0.75, 0).CX(0, 1).measure_all()

    for optimisation_level in (1, 2):
        compiled_circuit = backend.get_compiled_circuit(
            circuit, optimisation_level=optimisation_level
        )
        assert backend.valid_circuit(compiled_circuit)
        assert compiled_circuit.n_gates_of_type(OpType.Rz) == 0


# https://github.com/Quantinuum/pytket-quantinuum/issues/263
@pytest.mark.skipif(not have_pecos(), reason="pecos not installed")
def test_noiseless_emulation() -> None:
    backend = QuantinuumBackend("H2-1LE")
    c = Circuit(2).H(0).CX(0, 1).measure_all()
    c1 = backend.get_compiled_circuit(c)
    h = backend.process_circuit(c1, n_shots=100, noisy_simulation=False)
    r = backend.get_result(h)
    counts = r.get_counts()
    assert all(x0 == x1 for x0, x1 in counts)


def test_optimisation_level_3_compilation() -> None:
    b = QuantinuumBackend("H2-1")

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


def test_no_phase_ops() -> None:
    b = QuantinuumBackend("H2-1")

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
    h11e_backend = QuantinuumBackend("H2-1", machine_debug=True)

    for opt_level in range(4):
        default_pass = h11e_backend.default_compilation_pass(opt_level)
        original_pass_dict = default_pass.to_dict()
        reconstructed_pass = BasePass.from_dict(
            original_pass_dict,
            {"resize scratch bits": _gen_scratch_transformation(MAX_C_REG_WIDTH)},
        )
        assert isinstance(reconstructed_pass, SequencePass)
        assert original_pass_dict == reconstructed_pass.to_dict()


def test_pass_from_info() -> None:
    be = QuantinuumBackend("H2-1")
    info = be.backend_info
    assert info is not None
    actual_pass = QuantinuumBackend.pass_from_info(info)
    expected_pass = be.default_compilation_pass()
    assert actual_pass.to_dict() == expected_pass.to_dict()


def test_preserve_qubit_names() -> None:
    b = QuantinuumBackend(
        "H2-1",
        machine_debug=True,
        compilation_config=QuantinuumBackendCompilationConfig(
            preserve_qubit_names=True
        ),
    )
    c = Circuit(1, 1)
    qb = Qubit("a", 3)
    c.add_qubit(qb)
    c.H(qb).measure_all()
    c1 = b.get_compiled_circuit(c)
    assert c.qubits == c1.qubits
