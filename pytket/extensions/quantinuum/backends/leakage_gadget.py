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

from pytket import Bit, Circuit, OpType, Qubit
from pytket.backends.backendresult import BackendResult
from pytket.circuit import BitRegister
from pytket.utils.outcomearray import OutcomeArray

if TYPE_CHECKING:
    from collections.abc import Sequence

LEAKAGE_DETECTION_BIT_NAME_ = "leakage_detection_bit"
LEAKAGE_DETECTION_QUBIT_NAME_ = "leakage_detection_qubit"


def get_leakage_gadget_circuit(
    circuit_qubit: Qubit, postselection_qubit: Qubit, postselection_bit: Bit
) -> Circuit:
    """
    Returns a two qubit Circuit for detecting leakage errors.

    :param circuit_qubit: Generated circuit detects whether leakage errors
        have occurred in this qubit.
    :param postselection_qubit: Measured qubit to detect leakage error.
    :param postselection_bit: Leakage detection result is written to this bit.
    :return: Circuit for detecting leakage errors for specified ids.
    """
    c = Circuit()
    c.add_qubit(circuit_qubit)
    c.add_qubit(postselection_qubit)
    c.add_gate(OpType.Reset, [postselection_qubit])
    c.add_bit(postselection_bit)
    c.X(postselection_qubit)
    c.add_barrier([circuit_qubit, postselection_qubit])
    c.H(postselection_qubit).ZZMax(postselection_qubit, circuit_qubit)
    c.add_barrier([circuit_qubit, postselection_qubit])
    c.ZZMax(postselection_qubit, circuit_qubit).H(postselection_qubit).Z(circuit_qubit)
    c.add_barrier([circuit_qubit, postselection_qubit])
    c.Measure(postselection_qubit, postselection_bit)
    return c


def get_detection_circuit(circuit: Circuit, n_device_qubits: int) -> Circuit:  # noqa: PLR0912 PLR0915
    """
    For a passed circuit, inserts a leakage detection circuit before
    each measurement using spare device qubits or data qubits after their
    final measurement. Measurements are left unchecked if neither is available.
    Waiting cannot free a qubit if its remaining operations depend on this measurement.
    All additional Qubit added for leakage detection are
    written to a new register "leakage_detection_qubit" and all
    additional Bit are written to a new register "leakage_detection_bit".

    :param circuit: Circuit to have leakage detection added.
    :param n_device_qubits: Total number of qubits supported by the device
        being compiled to.

    :return: Circuit with leakage detection circuitry added.
    """
    n_qubits: int = circuit.n_qubits
    if n_qubits == 0:
        raise ValueError(
            "Circuit for Leakage Gadget Postselection must have at least one Qubit."
        )
    n_spare_qubits: int = n_device_qubits - n_qubits
    # N.b. even if n_spare_qubits == 0 , we will reuse measured data qubits

    # construct detection circuit
    detection_circuit: Circuit = Circuit()
    postselection_qubits: list[Qubit] = [
        Qubit(LEAKAGE_DETECTION_QUBIT_NAME_, i) for i in range(n_spare_qubits)
    ]
    for q in circuit.qubits + postselection_qubits:
        detection_circuit.add_qubit(q)
    for b in circuit.bits:
        detection_circuit.add_bit(b)

    # identify final measurements only to decide when data qubits can become ancillas
    # the second pass adds gadgets to both mid-circuit and final measurements
    end_circuit_measures: dict[Qubit, int] = {}
    for i, com in enumerate(circuit):
        if com.op.type == OpType.Barrier:
            continue
        for q in com.qubits:
            # a later use of this qubit means its previous measurement was not final
            end_circuit_measures.pop(q, None)
        if com.op.type == OpType.Measure:
            # use the command index to distinguish repeated identical measurements
            end_circuit_measures[com.qubits[0]] = i

    # we try to use each free architecture qubit as few times as possible
    ps_q_index: int = 0
    ps_b_index: int = 0
    for i, com in enumerate(circuit):
        op, args = com.op, com.args
        if op.type == OpType.Barrier:
            detection_circuit.add_barrier(args)
            continue
        if op.type == OpType.Measure:
            q, b = com.qubits[0], com.bits[0]
            # if there are no spare qubits we wait until a data qubit has its
            # final measurement before using it as an ancilla qubit
            if postselection_qubits:
                if q.reg_name == LEAKAGE_DETECTION_QUBIT_NAME_:
                    raise ValueError(
                        "Leakage Gadget scheme makes a qubit register named "
                        "'leakage_detection_qubit' but this already exists in"
                        " the passed circuit."
                    )
                ps_q_index = (
                    0 if ps_q_index == len(postselection_qubits) else ps_q_index
                )
                leakage_detection_bit: Bit = Bit(
                    LEAKAGE_DETECTION_BIT_NAME_, ps_b_index
                )
                if leakage_detection_bit in circuit.bits:
                    raise ValueError(
                        "Leakage Gadget scheme makes a new Bit named 'leakage_detection_bit'"
                        " but this already exists in the passed circuit."
                    )
                leakage_gadget_circuit: Circuit = get_leakage_gadget_circuit(
                    q, postselection_qubits[ps_q_index], leakage_detection_bit
                )
                detection_circuit.append(leakage_gadget_circuit)
                ps_q_index += 1
                ps_b_index += 1
            detection_circuit.Measure(q, b)
            # only reuse qubits after their final measurement
            if end_circuit_measures.get(q) == i:
                postselection_qubits.append(q)
        elif op.is_gate():
            detection_circuit.add_gate(op.type, op.params, args)
        elif op.type == OpType.SetBits:
            detection_circuit.add_c_setbits(op.values, args)  # type: ignore
        elif op.type == OpType.CopyBits:
            assert len(args) % 2 == 0
            n = len(args) // 2
            detection_circuit.add_c_copybits(args[:n], args[n:])  # type: ignore
        elif op.type == OpType.ClExpr:
            detection_circuit.add_clexpr(op.expr, args)  # type: ignore
        elif op.type == OpType.RNGSeed:
            creg = BitRegister(args[0].reg_name, 64)
            detection_circuit.set_rng_seed(creg)
        elif op.type == OpType.RNGBound:
            creg = BitRegister(args[0].reg_name, 32)
            detection_circuit.set_rng_bound(creg)
        elif op.type == OpType.RNGIndex:
            creg = BitRegister(args[0].reg_name, 32)
            detection_circuit.set_rng_index(creg)
        elif op.type == OpType.RNGNum:
            creg = BitRegister(args[0].reg_name, 32)
            detection_circuit.get_rng_num(creg)
        elif op.type == OpType.JobShotNum:
            creg = BitRegister(args[0].reg_name, 32)
            detection_circuit.get_job_shot_num(creg)
        elif op.type == OpType.ExplicitPredicate:
            match op.get_name():
                case "AND":
                    [arg0_in, arg1_in, arg_out] = args
                    detection_circuit.add_c_and(arg0_in, arg1_in, arg_out)  # type: ignore
                case "OR":
                    [arg0_in, arg1_in, arg_out] = args
                    detection_circuit.add_c_or(arg0_in, arg1_in, arg_out)  # type: ignore
                case "XOR":
                    [arg0_in, arg1_in, arg_out] = args
                    detection_circuit.add_c_xor(arg0_in, arg1_in, arg_out)  # type: ignore
                case "NOT":
                    [arg_in, arg_out] = args
                    detection_circuit.add_c_not(arg_in, arg_out)  # type: ignore
                case _:
                    raise ValueError(
                        f"ExplicitPredicate '{op.get_name()}' not supported in leakage detection circuit."
                    )
        elif op.type == OpType.Conditional:
            # preserve the condition without accessing gate parameters on classical ops
            detection_circuit.add_gate(op, args)
        else:
            raise ValueError(
                f"Operation type {op.type} not supported in leakage detection circuit."
            )

    detection_circuit.remove_blank_wires()
    return detection_circuit


def prune_shots_detected_as_leaky(result: BackendResult) -> BackendResult:
    """
    For all states with a Bit with name "leakage_detection_bit"
    in a state 1 sets the counts to 0.

    :param result: Shots returned from device.
    :return: Shots with leakage cases removed.
    """
    regular_bits: list[Bit] = [
        b for b in result.c_bits if b.reg_name != LEAKAGE_DETECTION_BIT_NAME_
    ]
    leakage_bits: list[Bit] = [
        b for b in result.c_bits if b.reg_name == LEAKAGE_DETECTION_BIT_NAME_
    ]
    received_counts: Counter[tuple[int, ...]] = result.get_counts(
        cbits=regular_bits + leakage_bits
    )
    discarded_counts: Counter[tuple[int, ...]] = Counter(
        {
            tuple(state[: len(regular_bits)]): received_counts[state]
            for state in received_counts
            # start after regular bits: with no leakage bits, state[-0:] is the whole state
            if not any(state[len(regular_bits) :])
        }
    )
    return BackendResult(
        counts=Counter(
            {
                OutcomeArray.from_readouts([key]): val
                for key, val in discarded_counts.items()
            }
        ),
        c_bits=cast("Sequence[Bit]", regular_bits),
    )
