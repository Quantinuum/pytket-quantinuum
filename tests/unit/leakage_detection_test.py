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


from collections import Counter
from typing import TYPE_CHECKING, cast

import pytest

from pytket import Bit, Circuit, OpType, Qubit
from pytket.backends.backendresult import BackendResult
from pytket.circuit import BitRegister
from pytket.extensions.quantinuum.backends.leakage_gadget import (
    LEAKAGE_DETECTION_BIT_NAME_,
    LEAKAGE_DETECTION_QUBIT_NAME_,
    get_detection_circuit,
    get_leakage_gadget_circuit,
    prune_shots_detected_as_leaky,
)
from pytket.utils.outcomearray import OutcomeArray

if TYPE_CHECKING:
    from collections.abc import Sequence


def test_postselection_circuits_1qb_task_gen() -> None:
    comparison_circuit: Circuit = Circuit(1, 1)
    lg_qb = Qubit(LEAKAGE_DETECTION_QUBIT_NAME_, 0)
    lg_b = Bit(LEAKAGE_DETECTION_BIT_NAME_, 0)
    comparison_circuit.add_qubit(lg_qb)
    comparison_circuit.add_bit(lg_b)
    comparison_circuit.add_gate(OpType.Reset, [lg_qb])
    comparison_circuit.X(lg_qb).H(0)
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.H(lg_qb).ZZMax(lg_qb, Qubit(0))
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.ZZMax(lg_qb, Qubit(0)).H(lg_qb).Z(0)
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.Measure(0, 0).Measure(lg_qb, lg_b)
    assert comparison_circuit == get_detection_circuit(
        Circuit(1, 1).H(0).Measure(0, 0), 2
    )


def test_postselection_circuits_2qb_2_spare_task_gen() -> None:
    comparison_circuit = Circuit(2, 2)
    lg_qb0 = Qubit(LEAKAGE_DETECTION_QUBIT_NAME_, 0)
    lg_b0 = Bit(LEAKAGE_DETECTION_BIT_NAME_, 0)
    lg_qb1 = Qubit(LEAKAGE_DETECTION_QUBIT_NAME_, 1)
    lg_b1 = Bit(LEAKAGE_DETECTION_BIT_NAME_, 1)
    comparison_circuit.add_qubit(lg_qb0)
    comparison_circuit.add_bit(lg_b0)
    comparison_circuit.add_qubit(lg_qb1)
    comparison_circuit.add_bit(lg_b1)
    comparison_circuit.add_gate(OpType.Reset, [lg_qb0])
    comparison_circuit.add_gate(OpType.Reset, [lg_qb1])
    comparison_circuit.X(lg_qb0).X(lg_qb1).H(0).CZ(0, 1)
    comparison_circuit.add_barrier([Qubit(0), lg_qb0])
    comparison_circuit.add_barrier([Qubit(1), lg_qb1])
    comparison_circuit.H(lg_qb0).H(lg_qb1).ZZMax(lg_qb0, Qubit(0)).ZZMax(
        lg_qb1, Qubit(1)
    )
    comparison_circuit.add_barrier([Qubit(0), lg_qb0])
    comparison_circuit.add_barrier([Qubit(1), lg_qb1])
    comparison_circuit.ZZMax(lg_qb0, Qubit(0)).ZZMax(lg_qb1, Qubit(1))
    comparison_circuit.H(lg_qb0).H(lg_qb1).Z(0).Z(1)
    comparison_circuit.add_barrier([Qubit(0), lg_qb0])
    comparison_circuit.add_barrier([Qubit(1), lg_qb1])
    comparison_circuit.Measure(0, 0).Measure(1, 1).Measure(lg_qb0, lg_b0).Measure(
        lg_qb1, lg_b1
    )

    assert comparison_circuit == get_detection_circuit(
        Circuit(2, 2).H(0).CZ(0, 1).Measure(0, 0).Measure(1, 1), 4
    )


def test_postselection_circuits_2qb_1_spare_task_gen() -> None:
    comparison_circuit = Circuit(2, 2)
    lg_qb = Qubit(LEAKAGE_DETECTION_QUBIT_NAME_, 0)
    lg_b0 = Bit(LEAKAGE_DETECTION_BIT_NAME_, 0)
    lg_b1 = Bit(LEAKAGE_DETECTION_BIT_NAME_, 1)
    comparison_circuit.add_qubit(lg_qb)
    comparison_circuit.add_bit(lg_b0)
    comparison_circuit.add_bit(lg_b1)
    comparison_circuit.add_gate(OpType.Reset, [lg_qb])

    comparison_circuit.X(lg_qb).H(0).CZ(0, 1)
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.H(lg_qb).ZZMax(lg_qb, Qubit(0))
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.ZZMax(lg_qb, Qubit(0)).H(lg_qb).Z(0)
    comparison_circuit.add_barrier([Qubit(0), lg_qb])
    comparison_circuit.Measure(0, 0).Measure(lg_qb, lg_b0)

    # with reuse of pre-measured qubits, this will now use Qubit 0 from the circuit
    comparison_circuit.add_gate(OpType.Reset, [Qubit(0)])
    comparison_circuit.X(Qubit(0))
    comparison_circuit.add_barrier([Qubit(1), Qubit(0)])
    comparison_circuit.H(Qubit(0)).ZZMax(Qubit(0), Qubit(1))
    comparison_circuit.add_barrier([Qubit(1), Qubit(0)])
    comparison_circuit.ZZMax(Qubit(0), Qubit(1)).H(Qubit(0)).Z(1)
    comparison_circuit.add_barrier([Qubit(1), Qubit(0)])
    comparison_circuit.Measure(1, 1).Measure(Qubit(0), lg_b1)

    assert comparison_circuit == get_detection_circuit(
        Circuit(2, 2).H(0).CZ(0, 1).Measure(0, 0).Measure(1, 1), 3
    )


def test_postselection_no_qubits() -> None:
    with pytest.raises(ValueError):
        get_detection_circuit(Circuit(0), 1)


def test_postselection_not_enough_device_qubits_0() -> None:
    comparison_circuit = Circuit(2, 2)
    comparison_circuit.CX(0, 1).Measure(0, 0)
    comparison_circuit.append(
        get_leakage_gadget_circuit(
            Qubit(1), Qubit(0), Bit(LEAKAGE_DETECTION_BIT_NAME_, 0)
        )
    )
    comparison_circuit.Measure(1, 1)
    assert comparison_circuit == get_detection_circuit(
        Circuit(2, 2).CX(0, 1).measure_all(), 2
    )


def test_postselection_not_enough_device_qubits_1() -> None:
    comparison_circuit = Circuit(4, 4)
    comparison_circuit.CX(0, 1).Measure(0, 0).CX(1, 2)
    comparison_circuit.CX(2, 3)
    comparison_circuit.append(
        get_leakage_gadget_circuit(
            Qubit(1), Qubit(0), Bit(LEAKAGE_DETECTION_BIT_NAME_, 0)
        )
    )
    comparison_circuit.Measure(1, 1)
    comparison_circuit.append(
        get_leakage_gadget_circuit(
            Qubit(2), Qubit(1), Bit(LEAKAGE_DETECTION_BIT_NAME_, 1)
        )
    )
    comparison_circuit.Measure(2, 2)
    comparison_circuit.append(
        get_leakage_gadget_circuit(
            Qubit(3), Qubit(2), Bit(LEAKAGE_DETECTION_BIT_NAME_, 2)
        )
    )
    comparison_circuit.Measure(3, 3)
    assert comparison_circuit == get_detection_circuit(
        Circuit(4, 4).CX(0, 1).CX(1, 2).CX(2, 3).measure_all(), 4
    )


def test_postselection_existing_qubit() -> None:
    lg_qb = Qubit(LEAKAGE_DETECTION_QUBIT_NAME_, 0)
    c = Circuit(1, 2).X(0)
    c.add_qubit(lg_qb)
    c.X(lg_qb)
    c.Measure(0, 0)
    c.Measure(lg_qb, Bit(1))
    with pytest.raises(ValueError):
        get_detection_circuit(c, 2)


def test_postselection_existing_bit() -> None:
    lg_b = Bit(LEAKAGE_DETECTION_BIT_NAME_, 0)
    c = Circuit(2, 1).CX(0, 1)
    c.add_bit(lg_b)
    c.Measure(0, 0)
    c.Measure(Qubit(1), lg_b)

    with pytest.raises(ValueError):
        get_detection_circuit(c, 2)


def test_postselection_discard_0() -> None:
    counter_dict = {(0, 0): 100, (0, 1): 75, (1, 0): 50, (1, 1): 25}
    counts = Counter(
        {OutcomeArray.from_readouts([key]): val for key, val in counter_dict.items()}
    )

    discard_result = prune_shots_detected_as_leaky(
        BackendResult(
            counts=counts,
            c_bits=cast("Sequence[Bit]", [Bit(0), Bit(LEAKAGE_DETECTION_BIT_NAME_, 0)]),
        )
    ).get_counts()
    assert discard_result[(0,)] == 100
    assert discard_result[(1,)] == 50


def test_postselection_discard_1() -> None:
    raw_readouts = (
        [[0, 0, 0, 0]] * 100
        + [[0, 1, 0, 0]] * 75
        + [[0, 1, 0, 1]] * 50
        + [[1, 0, 1, 0]] * 33
    )
    outcomes = OutcomeArray.from_readouts(raw_readouts)

    backres_shots = BackendResult(
        shots=outcomes,
        c_bits=[
            Bit(3),
            Bit(LEAKAGE_DETECTION_BIT_NAME_, 23),
            Bit("a", 7),
            Bit(LEAKAGE_DETECTION_BIT_NAME_, 3),
        ],
    )
    discard_result = prune_shots_detected_as_leaky(backres_shots).get_counts()
    assert discard_result[(0, 0)] == 100
    assert discard_result[(1, 1)] == 33


def test_classical_ops() -> None:
    c = Circuit(2, 2)
    c.add_c_register("reg32", 32)
    c.add_c_register("reg64", 64)
    c.H(0).CX(0, 1)
    c.add_c_setbits([True, False], [Bit(0), Bit(1)])
    c.add_c_copybits([Bit(0)], [Bit(1)])
    c.set_rng_seed(BitRegister("reg64", 64))
    c.set_rng_bound(BitRegister("reg32", 32))
    c.set_rng_index(BitRegister("reg32", 32))
    c.get_rng_num(BitRegister("reg32", 32))
    c.get_job_shot_num(BitRegister("reg32", 32))
    c.measure_all()
    c_d = get_detection_circuit(c, 100)
    assert c_d.n_qubits == 4
