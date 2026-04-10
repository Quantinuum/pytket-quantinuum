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
import re
import warnings
from ast import literal_eval
from base64 import b64encode
from collections import Counter
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from enum import Enum
from functools import cache, cached_property
from typing import Any, Union, cast
from uuid import uuid1

import numpy as np

from pytket.architecture import FullyConnected
from pytket.backends import Backend, CircuitStatus, ResultHandle, StatusEnum
from pytket.backends.backend import KwargTypes
from pytket.backends.backend_exceptions import CircuitNotRunError
from pytket.backends.backendinfo import BackendInfo
from pytket.backends.backendresult import BackendResult
from pytket.backends.resulthandle import _ResultIdTuple
from pytket.circuit import Bit, Circuit, OpType
from pytket.extensions.quantinuum._metadata import __extension_version__
from pytket.extensions.quantinuum.backends.leakage_gadget import get_detection_circuit
from pytket.passes import (
    AutoRebase,
    AutoSquash,
    BasePass,
    DecomposeBoxes,
    DecomposeTK2,
    FlattenRelabelRegistersPass,
    FullPeepholeOptimise,
    GreedyPauliSimp,
    NormaliseTK2,
    RemoveBarriers,
    RemovePhaseOps,
    RemoveRedundancies,
    SequencePass,
    SimplifyInitial,
    SynthesiseTK,
    ZZPhaseToRz,
    scratch_reg_resize_pass,
)
from pytket.predicates import (
    CliffordCircuitPredicate,
    GateSetPredicate,
    MaxNClRegPredicate,
    MaxNQubitsPredicate,
    NoSymbolsPredicate,
    Predicate,
)
from pytket.qasm import circuit_to_qasm_str
from pytket.qir import QIRFormat, QIRProfile, pytket_to_qir
from pytket.qir.conversion.api import ClassicalRegisterWidthError
from pytket.unit_id import _TEMP_BIT_NAME
from pytket.utils import prepare_circuit
from pytket.utils.outcomearray import OutcomeArray
from pytket.wasm import WasmFileHandler

from .api_wrappers import QuantinuumAPI, QuantinuumAPIError
from .data import QuantinuumBackendData

_DEBUG_HANDLE_PREFIX = "_MACHINE_DEBUG_"
MAX_C_REG_WIDTH = 32

_STATUS_MAP = {
    "queued": StatusEnum.QUEUED,
    "running": StatusEnum.RUNNING,
    "completed": StatusEnum.COMPLETED,
    "failed": StatusEnum.ERROR,
    "canceling": StatusEnum.CANCELLED,
    "canceled": StatusEnum.CANCELLED,
    "depleted": StatusEnum.ERROR,
    "terminated": StatusEnum.ERROR,
}

_ADDITIONAL_GATES = {
    OpType.Reset,
    OpType.Measure,
    OpType.Barrier,
    OpType.RangePredicate,
    OpType.MultiBit,
    OpType.ExplicitPredicate,
    OpType.ExplicitModifier,
    OpType.SetBits,
    OpType.CopyBits,
    OpType.ClExpr,
    OpType.WASM,
    OpType.RNGSeed,
    OpType.RNGBound,
    OpType.RNGIndex,
    OpType.RNGNum,
    OpType.JobShotNum,
}

_GATE_MAP = {
    "Rxxyyzz": OpType.TK2,
    "Rz": OpType.Rz,
    "RZZ": OpType.ZZPhase,
    "TK2": OpType.TK2,
    "U1q": OpType.PhasedX,
    "ZZ": OpType.ZZMax,
}

_ALL_GATES = _ADDITIONAL_GATES.copy()
_ALL_GATES.update(_GATE_MAP.values())


def _default_2q_gate(device_name: str) -> OpType:
    # If we change this, we should update the main documentation page and highlight it
    # in the changelog.
    return OpType.ZZPhase


def _get_gateset(gates: list[str]) -> set[OpType]:
    gs = _ADDITIONAL_GATES.copy()
    for gate in gates:
        if gate not in _GATE_MAP:
            warnings.warn(f"Gate {gate} not recognized.")  # noqa: B028
        else:
            gs.add(_GATE_MAP[gate])
    return gs


def _is_scratch(bit: Bit) -> bool:
    reg_name = bit.reg_name
    return bool(reg_name == _TEMP_BIT_NAME) or reg_name.startswith(f"{_TEMP_BIT_NAME}_")


def _used_scratch_registers(qasm: str) -> set[str]:
    # See https://github.com/Quantinuum/tket/blob/e846e8a7bdcc4fa29967d211b7fbf452ec970dfb/
    # pytket/pytket/qasm/qasm.py#L966
    def_matcher = re.compile(rf"creg ({_TEMP_BIT_NAME}\_*\d*)\[\d+\]")
    regs = set()
    for line in qasm.split("\n"):
        if reg := def_matcher.match(line):
            regs.add(reg.group(1))
    return regs


class GetResultFailed(Exception):
    pass


class NoSyntaxChecker(Exception):
    pass


class MaxShotsExceeded(Exception):
    pass


class WasmUnsupported(Exception):
    pass


class BatchingUnsupported(Exception):
    """Batching not supported for this backend."""


class LanguageUnsupported(Exception):
    """Submission language not supported for this backend."""


@dataclass
class DeviceNotAvailable(Exception):
    device_name: str


class Language(Enum):
    """Language used for submission of circuits."""

    QASM = 0  # "OPENQASM 2.0"
    QIR = 1  # pytket qir with classical functions: "QIR 1.0"
    PQIR = 2  # profile QIR: "QIR 1.0"


def _language2str(language: Language) -> str:
    """returns matching string for Language enum"""
    if language == Language.QASM:
        return "OPENQASM 2.0"
    return "QIR 1.0"


# DEFAULT_API_HANDLER provides a global re-usable API handler
# that will persist after this module is imported.
DEFAULT_API_HANDLER = QuantinuumAPI()

QuumKwargTypes = Union[KwargTypes, WasmFileHandler, dict[str, Any], OpType, bool]  # noqa: UP007


@dataclass
class QuantinuumBackendCompilationConfig:
    """
    Options to configure default compilation and rebase passes.

    * ``allow_implicit_swaps``: Whether to allow use of implicit swaps when rebasing.
      The default is to allow implicit swaps.
    * ``target_2qb_gate``: Choice of two-qubit gate. The default is to use the device's
      default.
    """

    allow_implicit_swaps: bool = True
    target_2qb_gate: OpType | None = None


@cache
def have_pecos() -> bool:
    try:
        import pytket_pecos  # type: ignore # noqa # pylint: disable=unused-import

        return True
    except ImportError:
        return False


@dataclass
class _LocalEmulatorConfiguration:
    """Options stored internally when running circuits on the local emulator."""

    circuit: Circuit
    wasm_fh: WasmFileHandler | None
    n_shots: int
    seed: int | None
    multithreading: bool
    noisy_simulation: bool


class BackendOfflineError(Exception):
    """Raised when backend constructed with the `data` parameter is asked to make an
    online API call.
    """


class QuantinuumBackend(Backend):
    """
    Interface to a Quantinuum device.
    More information about the QuantinuumBackend can be found on this page
    https://docs.quantinuum.com/tket/extensions/pytket-quantinuum/index.html
    """

    _supports_shots = True
    _supports_counts = True
    _supports_contextual_optimisation = True
    _persistent_handles = True

    def __init__(  # noqa: PLR0913
        self,
        device_name: str,
        label: str | None = "job",
        simulator: str = "state-vector",
        group: str | None = None,
        provider: str | None = None,
        machine_debug: bool = False,
        api_handler: QuantinuumAPI = DEFAULT_API_HANDLER,
        compilation_config: QuantinuumBackendCompilationConfig | None = None,
        data: QuantinuumBackendData | None = None,
        **kwargs: QuumKwargTypes,
    ):
        """Construct a new Quantinuum backend.

        :param device_name: Name of device, e.g. "H2-1"
        :param label: Job labels used if Circuits have no name, defaults to "job"
        :param simulator: Only applies to simulator devices, options are
            "state-vector" or "stabilizer", defaults to "state-vector"
        :param group: string identifier of a collection of jobs, can be used for usage
          tracking.
        :param provider: select a provider for federated authentication. We currently
            only support 'microsoft', which enables the microsoft Device Flow.
        :param api_handler: Instance of API handler, defaults to DEFAULT_API_HANDLER
        :param compilation_config: Optional compilation configuration
        :param data: Data characterizing the backend. If this is not provided, the data
            are retrieved online using the `device_name` provided. If it is provided,
            then no online queries are made and no online submission is possible.

        Supported kwargs:

        * `options`: items to add to the "options" dictionary of the request body, as a
          json-style dictionary (see :py:meth:`QuantinuumBackend.process_circuits`)
        """

        super().__init__()
        self._device_name = device_name
        self._label = label
        self._group = group

        self._backend_info: BackendInfo | None = None
        self._MACHINE_DEBUG = machine_debug

        self.simulator_type = simulator

        self.api_handler = api_handler

        self._process_circuits_options = cast(
            "dict[str, Any]", kwargs.get("options", {})
        )

        self._local_emulator_handles: dict[
            ResultHandle,
            _LocalEmulatorConfiguration,
        ] = dict()  # noqa: C408

        if compilation_config is None:
            self._compilation_config = QuantinuumBackendCompilationConfig()
        else:
            self._compilation_config = compilation_config

        self._data = data

    @property
    def compilation_config(self) -> QuantinuumBackendCompilationConfig:
        """The current compilation configuration for the Backend.

        Accessing this property will set the target_2qb_gate if it
        has not already been set.
        """
        if self._compilation_config.target_2qb_gate is None:
            self._compilation_config.target_2qb_gate = self.default_two_qubit_gate
        return self._compilation_config

    def get_compilation_config(self) -> QuantinuumBackendCompilationConfig:
        """Get the current compilation configuration."""
        return self.compilation_config

    def set_compilation_config_allow_implicit_swaps(
        self, allow_implicit_swaps: bool
    ) -> None:
        """Set the option to allow or disallow implicit swaps during compilation."""
        self.compilation_config.allow_implicit_swaps = allow_implicit_swaps

    def set_compilation_config_target_2qb_gate(self, target_2qb_gate: OpType) -> None:
        """Set the target two-qubit gate for compilation."""
        if target_2qb_gate not in self.two_qubit_gate_set:
            raise QuantinuumAPIError(
                "Requested target_2qb_gate is not supported by the given Device. "
                "The supported gateset is: " + str(self.two_qubit_gate_set)
            )
        self.compilation_config.target_2qb_gate = target_2qb_gate

    @classmethod
    @cache
    def _available_devices(
        cls,
        api_handler: QuantinuumAPI,
    ) -> list[dict[str, Any]]:
        """List devices available from Quantinuum.

        >>> QuantinuumBackend._available_devices()
        e.g. [{'name': 'H2-1', 'n_qubits': 56}]

        :param api_handler: Instance of API handler
        :return: Dictionaries of machine name and number of qubits.
        """
        return api_handler.get_machine_list()

    @classmethod
    def _dict_to_backendinfo(
        cls, dct: dict[str, Any], local_emulator: bool = False
    ) -> BackendInfo:
        dct1 = copy(dct)
        name: str = dct1.pop("name")
        n_qubits: int = dct1.pop("n_qubits")
        n_cl_reg: int | None = None
        if "n_classical_registers" in dct:
            n_cl_reg = dct1.pop("n_classical_registers")
        gate_set: list[str] = dct1.pop("gateset", [])
        if local_emulator:
            dct1["system_type"] = "local_emulator"
            dct1.pop("emulator", None)
            dct1["batching"] = False
            dct1.pop("noise_specs", None)
        return BackendInfo(
            name=cls.__name__,
            device_name=name + "LE" if local_emulator else name,
            version=__extension_version__,
            architecture=FullyConnected(n_qubits, "q"),
            gate_set=_get_gateset(gate_set),
            n_cl_reg=n_cl_reg,
            supports_fast_feedforward=True,
            supports_midcircuit_measurement=True,
            supports_reset=True,
            misc=dct1,
        )

    @classmethod
    def available_devices(
        cls,
        **kwargs: Any,
    ) -> list[BackendInfo]:
        """
        See :py:meth:`pytket.backends.backend.Backend.available_devices`.

        :param api_handler: Instance of API handler, defaults to DEFAULT_API_HANDLER
        :return: A list of BackendInfo objects for each available Backend.

        """
        api_handler = kwargs.get("api_handler", DEFAULT_API_HANDLER)
        jr = cls._available_devices(api_handler)
        devices = []
        for d in jr:
            devices.append(cls._dict_to_backendinfo(copy(d)))
            if have_pecos() and (d.get("system_type") == "hardware"):
                # Add a local-emulator variant
                devices.append(cls._dict_to_backendinfo(d, local_emulator=True))
        return devices

    def _retrieve_backendinfo(self) -> BackendInfo:
        if self._data is None:
            infos = self.available_devices(api_handler=self.api_handler)
            try:
                info = next(
                    entry for entry in infos if entry.device_name == self._device_name
                )
            except StopIteration:
                raise DeviceNotAvailable(self._device_name)  # noqa: B904
            info.misc["options"] = self._process_circuits_options
        else:
            info = BackendInfo(
                name=self.__class__.__name__,
                device_name=self._device_name,
                version=__extension_version__,
                architecture=FullyConnected(self._data.n_qubits, "q"),
                gate_set=set(self._data.gate_set),
                n_cl_reg=self._data.n_cl_reg,
                supports_fast_feedforward=True,
                supports_midcircuit_measurement=True,
                supports_reset=True,
                misc={},
            )
        return info

    @classmethod
    def device_state(
        cls,
        device_name: str,
        api_handler: QuantinuumAPI = DEFAULT_API_HANDLER,
    ) -> str:
        """Check the status of a device.

        >>> QuantinuumBackend.device_state('H2-1') # e.g. "online"


        :param device_name: Name of the device.
        :param api_handler: Instance of API handler, defaults to DEFAULT_API_HANDLER
        :return: String of state, e.g. "online"
        """
        infos = cls.available_devices(api_handler=api_handler)
        try:
            info = next(entry for entry in infos if entry.device_name == device_name)
        except StopIteration:
            raise DeviceNotAvailable(device_name)  # noqa: B904
        if info.get_misc("system_type") == "local_emulator":
            return "online"
        raise NotImplementedError("Cannot retrieve device state.")

    @property
    def backend_info(self) -> BackendInfo | None:
        if self._backend_info is None and not self._MACHINE_DEBUG:
            self._backend_info = self._retrieve_backendinfo()
        return self._backend_info

    @cached_property
    def _gate_set(self) -> set[OpType]:
        return (
            _ALL_GATES
            if self._MACHINE_DEBUG
            else cast("BackendInfo", self.backend_info).gate_set
        )

    @property
    def required_predicates(self) -> list[Predicate]:
        preds = [
            NoSymbolsPredicate(),
            GateSetPredicate(self._gate_set),
        ]
        if not self._MACHINE_DEBUG:
            assert self.backend_info is not None
            preds.append(MaxNQubitsPredicate(self.backend_info.n_nodes))
            preds.append(MaxNClRegPredicate(cast("int", self.backend_info.n_cl_reg)))
        if self.simulator_type == "stabilizer":
            preds.append(CliffordCircuitPredicate())

        return preds

    @cached_property
    def default_two_qubit_gate(self) -> OpType:
        """Returns the default two-qubit gate for the device."""
        default_2q_gate = _default_2q_gate(self._device_name)

        if default_2q_gate in self.two_qubit_gate_set:
            pass
        elif len(self.two_qubit_gate_set) > 0:
            default_2q_gate = list(self.two_qubit_gate_set)[0]  # noqa: RUF015
        else:
            raise ValueError("The device is not supporting any two qubit gates")

        return default_2q_gate

    @cached_property
    def two_qubit_gate_set(self) -> set[OpType]:
        """Returns the set of supported two-qubit gates.

        Submitted circuits must contain only one of these.
        """
        return self._gate_set & {OpType.ZZPhase, OpType.ZZMax, OpType.TK2}

    @property
    def is_local_emulator(self) -> bool:
        """True if the backend is a local emulator, otherwise False"""
        if self._MACHINE_DEBUG:
            return False
        if self._data is not None:
            return self._data.local_emulator
        info = self.backend_info
        assert info is not None
        return bool(info.get_misc("system_type") == "local_emulator")

    def rebase_pass(self) -> BasePass:
        assert self.compilation_config.target_2qb_gate in self.two_qubit_gate_set
        return AutoRebase(
            (self._gate_set - self.two_qubit_gate_set)
            | {self.compilation_config.target_2qb_gate},
            allow_swaps=self.compilation_config.allow_implicit_swaps,
        )

    def default_compilation_pass(
        self, optimisation_level: int = 2, timeout: int = 300
    ) -> BasePass:
        """
        :param optimisation_level: Allows values of 0, 1, 2 or 3, with higher values
            prompting more computationally heavy optimising compilation that
            can lead to reduced gate count in circuits.
        :param timeout: Only valid for optimisation level 3, gives a maximimum time
            for running a single thread of the pass :py:meth:`~pytket.passes.GreedyPauliSimp`. Increase for
            optimising larger circuits.

        :return: Compilation pass for compiling circuits to Quantinuum devices
        """
        assert optimisation_level in range(4)
        passlist = [
            DecomposeBoxes(),
            scratch_reg_resize_pass(),
        ]
        squash = AutoSquash({OpType.PhasedX, OpType.Rz})
        target_2qb_gate = self.compilation_config.target_2qb_gate
        assert target_2qb_gate is not None
        if target_2qb_gate == OpType.TK2:
            decomposition_passes = []
        elif target_2qb_gate == OpType.ZZPhase:
            decomposition_passes = [
                NormaliseTK2(),
                DecomposeTK2(
                    allow_swaps=self.compilation_config.allow_implicit_swaps,
                    ZZPhase_fidelity=1.0,
                ),
            ]
        elif target_2qb_gate == OpType.ZZMax:
            decomposition_passes = [
                NormaliseTK2(),
                DecomposeTK2(
                    allow_swaps=self.compilation_config.allow_implicit_swaps,
                    ZZMax_fidelity=1.0,
                ),
            ]
        else:
            raise ValueError(
                f"Unrecognized target 2-qubit gate: {target_2qb_gate.name}"
            )
        # If you make changes to the default_compilation_pass,
        # then please update this page accordingly
        # https://docs.quantinuum.com/tket/extensions/pytket-quantinuum/index.html#default-compilation
        # Edit this docs source file -> pytket-quantinuum/docs/intro.txt
        if optimisation_level == 0:
            passlist.append(self.rebase_pass())
        elif optimisation_level == 1:
            passlist.append(SynthesiseTK())
            passlist.extend(decomposition_passes)
            passlist.extend(
                [
                    self.rebase_pass(),
                    ZZPhaseToRz(),
                    RemoveRedundancies(),
                    squash,
                    RemoveRedundancies(),
                ]
            )
        elif optimisation_level == 2:  # noqa: PLR2004
            passlist.append(
                FullPeepholeOptimise(
                    allow_swaps=self.compilation_config.allow_implicit_swaps,
                    target_2qb_gate=OpType.TK2,
                )
            )
            passlist.extend(decomposition_passes)
            passlist.extend(
                [
                    self.rebase_pass(),
                    RemoveRedundancies(),
                    squash,
                    RemoveRedundancies(),
                ]
            )
        else:
            passlist.extend(
                [
                    RemoveBarriers(),
                    AutoRebase(
                        {
                            OpType.Z,
                            OpType.X,
                            OpType.Y,
                            OpType.S,
                            OpType.Sdg,
                            OpType.V,
                            OpType.Vdg,
                            OpType.H,
                            OpType.CX,
                            OpType.CY,
                            OpType.CZ,
                            OpType.SWAP,
                            OpType.Rz,
                            OpType.Rx,
                            OpType.Ry,
                            OpType.T,
                            OpType.Tdg,
                            OpType.ZZMax,
                            OpType.ZZPhase,
                            OpType.XXPhase,
                            OpType.YYPhase,
                            OpType.PhasedX,
                        }
                    ),
                    GreedyPauliSimp(
                        allow_zzphase=True,
                        only_reduce=True,
                        thread_timeout=timeout,
                        trials=10,
                    ),
                ]
            )
            passlist.extend(decomposition_passes)
            passlist.extend(
                [
                    self.rebase_pass(),
                    RemoveRedundancies(),
                    squash,
                    RemoveRedundancies(),
                ]
            )
        passlist.append(RemovePhaseOps())

        # In TKET, a qubit register with N qubits can have qubits
        # indexed with a a value greater than N, i.e. a single
        # qubit register can exist with index "7" or similar.
        # Similarly, a qubit register with N qubits could be defined
        # in a Circuit, but fewer than N qubits in the register
        # have operations.
        # Both of these cases can causes issues when converting to QASM,
        # as the size of the defined "qreg" can be larger than the
        # number of Qubits actually used, or at times larger than the
        # number of device Qubits, even if fewer are really used.
        # By flattening the Circuit qubit registers, we make sure
        # that the produced QASM has one "qreg", with the exact number
        # of qubits actually used in the Circuit.
        # The Circuit qubits attribute is iterated through, with the ith
        # qubit being assigned to the ith qubit of a new "q" register
        passlist.append(FlattenRelabelRegistersPass("q"))
        return SequencePass(passlist)

    @staticmethod
    def pass_from_info(
        backend_info: BackendInfo,
        compilation_config: QuantinuumBackendCompilationConfig | None = None,
        data: QuantinuumBackendData | None = None,
        optimisation_level: int = 2,
        timeout: int = 300,
    ) -> BasePass:
        if backend_info.device_name is None:
            raise ValueError("The provided BackendInfo has no device_name.")

        backend = QuantinuumBackend(
            device_name=backend_info.device_name,
            compilation_config=compilation_config,
            data=data,
        )
        backend._backend_info = backend_info
        return backend.default_compilation_pass(optimisation_level, timeout)

    def get_compiled_circuit(
        self, circuit: Circuit, optimisation_level: int = 2, timeout: int = 300
    ) -> Circuit:
        """
        Return a single circuit compiled with :py:meth:`default_compilation_pass`.

        :param optimisation_level: Allows values of 0, 1, 2 or 3, with higher values
            prompting more computationally heavy optimising compilation that
            can lead to reduced gate count in circuits.
        :param timeout: Only valid for optimisation level 3, gives a maximimum time
            for running a single thread of the pass :py:meth:`~pytket.passes.GreedyPauliSimp`. Increase for
            optimising larger circuits.

        :return: An optimised quantum circuit
        """
        return_circuit = circuit.copy()
        if optimisation_level == 3 and circuit.n_gates_of_type(OpType.Barrier) > 0:  # noqa: PLR2004
            warnings.warn(  # noqa: B028
                "Barrier operations in this circuit will be removed when using "
                "optimisation level 3."
            )
        self.default_compilation_pass(optimisation_level, timeout).apply(return_circuit)
        return return_circuit

    def get_compiled_circuits(
        self,
        circuits: Sequence[Circuit],
        optimisation_level: int = 2,
        timeout: int = 300,
    ) -> list[Circuit]:
        """Compile a sequence of circuits with :py:meth:`default_compilation_pass`
        and return the list of compiled circuits (does not act in place).

        As well as applying a degree of optimisation (controlled by the
        ``optimisation_level`` parameter), this method tries to ensure that the circuits
        can be run on the backend (i.e. successfully passed to
        :py:meth:`process_circuits`), for example by rebasing to the supported gate set,
        or routing to match the connectivity of the device. However, this is not always
        possible, for example if the circuit contains classical operations that are not
        supported by the backend. You may use :py:meth:`~pytket.backends.backend.Backend.valid_circuit` to check whether
        the circuit meets the backend's requirements after compilation. This validity
        check is included in :py:meth:`process_circuits` by default, before any circuits
        are submitted to the backend.

        If the validity check fails, you can obtain more information about the failure
        by iterating through the predicates in the :py:attr:`required_predicates` property of the
        backend, and running the :py:meth:`~pytket.predicates.Predicate.verify` method on each in turn with your
        circuit.

        :param circuits: The circuits to compile.
        :param optimisation_level: The level of optimisation to perform during
            compilation. See :py:meth:`default_compilation_pass` for a description of
            the different levels (0, 1, 2 or 3). Defaults to 2.
        :param timeout: Only valid for optimisation level 3, gives a maximimum time
            for running a single thread of the pass :py:meth:`~pytket.passes.GreedyPauliSimp`. Increase for
            optimising larger circuits.
        :return: Compiled circuits.
        """
        return [
            self.get_compiled_circuit(c, optimisation_level, timeout) for c in circuits
        ]

    @property
    def _result_id_type(self) -> _ResultIdTuple:
        return tuple((str, str, int, str))  # noqa: C409

    @staticmethod
    def _update_result_handle(handle: ResultHandle) -> ResultHandle:
        """Update a legacy handle to be compatible with current format."""
        if len(handle) == 2:  # noqa: PLR2004
            return ResultHandle(handle[0], handle[1], -1, "")
        if len(handle) == 3:  # noqa: PLR2004
            return ResultHandle(handle[0], handle[1], handle[2], "")
        return handle

    @staticmethod
    def get_jobid(handle: ResultHandle) -> str:
        """Return the corresponding Quantinuum Job ID from a ResultHandle.

        :param handle: result handle.
        :return: Quantinuum API Job ID string.
        """
        return cast("str", handle[0])

    @staticmethod
    def get_ppcirc_rep(handle: ResultHandle) -> Any:
        """Return the JSON serialization of the classiocal postprocessing circuit
        attached to a handle, if any.

        :param handle: result handle
        :return: serialized post-processing circuit, if any
        """
        return json.loads(cast("str", handle[1]))

    @staticmethod
    def get_results_width(handle: ResultHandle) -> int | None:
        """Return the truncation width of the results, if any.

        :param handle: result handle
        :return: truncation width of results, if any
        """
        n = cast("int", handle[2])
        if n == -1:
            return None
        assert n >= 0
        return n

    @staticmethod
    def get_results_selection(handle: ResultHandle) -> Any:
        """Return a list of pairs (register name, register index) representing the order
        of the expected results in the response. If None, then all results in the
        response are used, in lexicographic order.
        """
        s = cast("str", handle[3])
        if s == "":
            return None
        bits = json.loads(s)
        if bits is None:
            return None
        assert all(isinstance(name, str) and isinstance(idx, int) for name, idx in bits)
        return bits

    def process_circuits(  # noqa: PLR0912, PLR0915
        self,
        circuits: Sequence[Circuit],
        n_shots: None | int | Sequence[int | None] = None,
        valid_check: bool = True,
        **kwargs: QuumKwargTypes,
    ) -> list[ResultHandle]:
        """
        See :py:meth:`pytket.backends.backend.Backend.process_circuits`.

        Supported kwargs:

        * `postprocess`: apply end-of-circuit simplifications and classical
          postprocessing to improve fidelity of results (bool, default False)
        * `simplify_initial`: apply the pytket :py:meth:`~pytket.passes.SimplifyInitial` pass to improve
          fidelity of results assuming all qubits initialized to zero (bool, default
          False)
        * `noisy_simulation`: boolean flag to specify whether the simulator should
          perform noisy simulation with an error model (default value is `True`).
        * `group`: string identifier of a collection of jobs, can be used for usage
          tracking. Overrides the instance variable ``group``.
        * `wasm_file_handler`: a :py:class:`~pytket.wasm.wasm.WasmFileHandler` object for linked WASM module.
        * `pytketpass`: a :py:class:`pytket.passes.BasePass` intended to be applied
           by the backend (beta feature, may be ignored).
        * `options`: items to add to the "options" dictionary of the request body, as a
          json-style dictionary (in addition to any that were set in the backend
          constructor)
        * `request_options`: extra options to add to the request body as a
          json-style dictionary
        * `language`: languange for submission, of type :py:class:`Language`, default
          QIR.
        * `leakage_detection`: if true, adds additional Qubit and Bit to Circuit
          to detect leakage errors. Run `prune_shots_detected_as_leaky()` on returned
          :py:class:`~pytket.backends.backendresult.BackendResult` to get counts with leakage errors removed.
        * `n_leakage_detection_qubits`: if set, sets an upper bound on the number
          of additional qubits to be used when adding leakage detection
        * `seed`: for local emulators only, PRNG seed for reproduciblity (int)
        * `multithreading`: for local emulators only, boolean to indicate
          whether to use multithreading for emulation (defaults to False)
        * `max_cost`: if set, the maximum amount in HQC to be spent on running the job.
          Ignored for local emulator.
        """

        if self._data is not None and not self.is_local_emulator:
            raise BackendOfflineError(
                "process_circuits() not available for this backend"
            )

        circuits = list(circuits)
        n_shots_list = Backend._get_n_shots_as_list(  # noqa: SLF001
            n_shots,
            len(circuits),
            optional=False,
        )

        if kwargs.get("leakage_detection", False):
            n_device_nodes: int = cast("int", self.backend_info.n_nodes)  # type: ignore
            n_leakage_detection_qubits: int = kwargs.get(  # type: ignore
                "n_leakage_detection_qubits", n_device_nodes
            )
            if n_leakage_detection_qubits > n_device_nodes:
                raise ValueError(
                    "Number of qubits specified for leakage detection is larger than "
                    "the number of qubits on the device."
                )
            circuits = [
                self.get_compiled_circuit(
                    get_detection_circuit(c, n_leakage_detection_qubits),
                    optimisation_level=0,
                )
                for c in circuits
            ]

        if valid_check:
            self._check_all_circuits(circuits)

        postprocess = cast("bool", kwargs.get("postprocess", False))
        simplify_initial = kwargs.get("simplify_initial", False)
        noisy_simulation = cast("bool", kwargs.get("noisy_simulation", True))

        wasm_fh = cast("WasmFileHandler | None", kwargs.get("wasm_file_handler"))

        language: Language = cast("Language", kwargs.get("language", Language.QIR))

        handle_list = []

        max_shots = self.backend_info.misc.get("n_shots") if self.backend_info else None
        seed = kwargs.get("seed")
        if seed is not None and not isinstance(seed, int):
            raise ValueError("seed must be an integer or None")
        multithreading = bool(kwargs.get("multithreading"))
        for circ, n_shots in zip(circuits, n_shots_list, strict=False):  # noqa: PLR1704
            if max_shots is not None and n_shots > max_shots:
                raise MaxShotsExceeded(
                    f"Number of shots {n_shots} exceeds maximum {max_shots}"
                )
            if postprocess:
                c0, ppcirc = prepare_circuit(circ, allow_classical=False, xcirc=_xcirc)
                ppcirc_rep = ppcirc.to_dict()
            else:
                c0, ppcirc_rep = circ, None
            if simplify_initial:
                SimplifyInitial(
                    allow_classical=False, create_all_qubits=True, xcirc=_xcirc
                ).apply(c0)
            if self.is_local_emulator:
                jobid = str(uuid1())
                results_selection = []
                for name, count in Counter(bit.reg_name for bit in c0.bits).items():
                    for i in range(count):
                        results_selection.append((name, i))  # noqa: PERF401
                handle = ResultHandle(
                    jobid,
                    json.dumps(ppcirc_rep),
                    len(results_selection),
                    json.dumps(results_selection),
                )
                handle_list.append(handle)
                self._local_emulator_handles[handle] = _LocalEmulatorConfiguration(
                    c0,
                    wasm_fh,
                    n_shots,
                    seed,
                    multithreading,
                    noisy_simulation,
                )
                if seed is not None:
                    seed += 1
            else:
                results_selection = []
                if language == Language.QASM:
                    quantinuum_circ = circuit_to_qasm_str(
                        c0,
                        header="hqslib1",
                        # Override the reported value of max_classical_register_width
                        # (which in August 2025 is 63 for all devices), to enable RNG
                        # seeding which requires a 64-bit register.
                        maxwidth=64,
                    )

                    used_scratch_regs = _used_scratch_registers(quantinuum_circ)
                    for name, count in Counter(
                        bit.reg_name
                        for bit in c0.bits
                        if not _is_scratch(bit) or bit.reg_name in used_scratch_regs
                    ).items():
                        for i in range(count):
                            results_selection.append((name, i))

                else:
                    assert language == Language.QIR or language == Language.PQIR  # noqa: PLR1714
                    if language == Language.QIR:
                        profile = QIRProfile.PYTKET
                    else:
                        profile = QIRProfile.ADAPTIVE
                    try:
                        qir = pytket_to_qir(
                            c0,
                            "circuit generated by pytket-qir",
                            QIRFormat.BINARY,
                            profile=profile,
                        )
                    except ClassicalRegisterWidthError as e:
                        raise ValueError(
                            "Unable to convert pytket circuit to QIR as "
                            f"it contains a classical register of width {e.width}: "
                            f"maximum allowed width is {e.max_width}."
                        ) from None
                    quantinuum_circ = b64encode(cast("bytes", qir)).decode("utf-8")

                    for name, count in Counter(
                        bit.reg_name for bit in c0.bits if not _is_scratch(bit)
                    ).items():
                        for i in range(count):
                            results_selection.append((name, i))
                if self._MACHINE_DEBUG:
                    handle_list.append(
                        ResultHandle(
                            _DEBUG_HANDLE_PREFIX + str((circ.n_qubits, n_shots)),
                            json.dumps(ppcirc_rep),
                            len(results_selection),
                            json.dumps(results_selection),
                        )
                    )
                else:
                    raise NotImplementedError("Cannot obtain result handles.")

        return handle_list

    def _check_batchable(self) -> None:
        if self.backend_info and not self.backend_info.misc.get("batching", False):
            raise BatchingUnsupported

    def _update_cache_result(self, handle: ResultHandle, res: BackendResult) -> None:
        rescache = {"result": res}

        if handle in self._cache:
            self._cache[handle].update(rescache)
        else:
            self._cache[handle] = rescache

    def circuit_status(
        self, handle: ResultHandle, **kwargs: KwargTypes
    ) -> CircuitStatus:
        raise NotImplementedError("Cannot retrieve circuit status.")

    def get_result(self, handle: ResultHandle, **kwargs: KwargTypes) -> BackendResult:
        """
        See :py:meth:`pytket.backends.backend.Backend.get_result`.
        Supported kwargs: ``timeout``, ``wait``, ``use_websocket``.
        """
        handle = self._update_result_handle(handle)
        try:
            return super().get_result(handle)
        except CircuitNotRunError:
            jobid = self.get_jobid(handle)
            ppcirc_rep = self.get_ppcirc_rep(handle)
            n_bits = self.get_results_width(handle)
            results_selection = self.get_results_selection(handle)

            ppcirc = Circuit.from_dict(ppcirc_rep) if ppcirc_rep is not None else None

            if self._MACHINE_DEBUG or jobid.startswith(_DEBUG_HANDLE_PREFIX):
                debug_handle_info = jobid[len(_DEBUG_HANDLE_PREFIX) :]
                n_qubits, shots = literal_eval(debug_handle_info)
                return _convert_result(
                    {"c": (["0" * n_qubits] * shots)}, ppcirc, n_bits, results_selection
                )
            if self.is_local_emulator:
                if not have_pecos():
                    raise RuntimeError(  # noqa: B904
                        "Local emulator not available: \
try installing with the `pecos` option."
                    )
                from pytket_pecos import Emulator

                configuration = self._local_emulator_handles[handle]
                # workaround for https://github.com/Quantinuum/pytket-quantinuum/issues/473
                # add redundant SetBits so unused bits won't be omitted during
                # pytket to phir conversion
                circ = configuration.circuit.copy()
                unused_bits = set(circ.bits)
                for cmd in circ.get_commands():
                    unused_bits = unused_bits - set(cmd.args)
                for bit in unused_bits:
                    circ.add_c_setbits([False], [bit])

                emu = Emulator(
                    circ,
                    wasm=configuration.wasm_fh,
                    qsim="state-vector",
                    seed=configuration.seed,
                )
                res = emu.run(
                    n_shots=configuration.n_shots,
                    multithreading=configuration.multithreading,
                )
                backres = BackendResult(c_bits=circ.bits, shots=res, ppcirc=ppcirc)
            else:
                raise NotImplementedError(
                    "Cannot get result from remote backend."
                ) from None
            self._update_cache_result(handle, backres)
            return backres


_xcirc = Circuit(1).add_gate(OpType.PhasedX, [1, 0], [0])
_xcirc.add_phase(0.5)


def _convert_result(
    resultdict: dict[str, list[str]],
    ppcirc: Circuit | None = None,
    n_bits: int | None = None,
    results_selection: list[tuple[str, int]] | None = None,
) -> BackendResult:
    for creg, reslist in resultdict.items():
        if any(["-" in res for res in reslist]):  # noqa: C419
            raise ValueError(
                f"found negative value for creg: {creg}. \
This could indicate a problem with the circuit submitted"
            )

    if results_selection is None:
        found_int_res = any(
            re.findall("[23456789]", res)
            for reslist in resultdict.values()
            for res in reslist
        )

        if found_int_res:
            # this is only a temporary solution and not fully working
            # see issue https://github.com/Quantinuum/pytket-quantinuum/issues/501

            def conv_int(res: str) -> list:
                long_res = bin(int(res)).replace(
                    "0b",
                    "0000000000000000000000000000000000000\
00000000000000000000000000",  # 0 * 63
                )
                return list(long_res[len(long_res) - 64 : len(long_res)])

            array_dict = {
                creg: np.array([conv_int(a) for a in reslist]).astype(np.uint8)
                for creg, reslist in resultdict.items()
            }
        else:
            array_dict = {
                creg: np.array([list(a) for a in reslist]).astype(np.uint8)
                for creg, reslist in resultdict.items()
            }

        reversed_creg_names = sorted(array_dict.keys(), reverse=True)
        c_bits = [
            Bit(name, ind)
            for name in reversed_creg_names
            for ind in range(array_dict[name].shape[-1] - 1, -1, -1)
        ]
        if n_bits is not None:
            assert n_bits >= 0 and n_bits <= len(c_bits)
            c_bits = c_bits[:n_bits]
            for creg in array_dict:
                array_dict[creg] = array_dict[creg][:, :n_bits]

        stacked_array = cast(
            "Sequence[Sequence[int]]",
            np.hstack([array_dict[name] for name in reversed_creg_names]),
        )
    else:
        assert n_bits == len(results_selection)

        # Figure out the number of shots and sanity-check the results list.
        n_shots_per_reg = [len(reslist) for reslist in resultdict.values()]
        n_shots = n_shots_per_reg[0] if n_shots_per_reg else 0
        assert all(n == n_shots for n in n_shots_per_reg)

        c_bits = [Bit(name, ind) for name, ind in results_selection]

        # Construct the shots table
        try:
            stacked_array = [
                [int(resultdict[name][i][-1 - ind]) for name, ind in results_selection]
                for i in range(n_shots)
            ]
        except IndexError:
            # this is only a temporary solution and not fully working
            # see issue https://github.com/Quantinuum/pytket-quantinuum/issues/501
            stacked_array = [
                [
                    int(
                        bin(int(resultdict[name][i])).replace(
                            "0b",
                            "0000000000000000000000000000000000000\
00000000000000000000000000",  # 0 * 63
                        )[-1 - ind]
                    )
                    for name, ind in results_selection
                ]
                for i in range(n_shots)
            ]

    return BackendResult(
        c_bits=c_bits,
        shots=OutcomeArray.from_readouts(stacked_array),
        ppcirc=ppcirc,
    )


def _parse_status(response: dict) -> CircuitStatus:
    h_status = response["status"]
    msgdict = {
        k: response.get(k)
        for k in (
            "name",
            "submit-date",
            "start-date",
            "result-date",
            "queue-position",
            "cost",
            "error",
            "cost-confidence",
            "last-shot",
            "qubits",
            "priority",
        )
    }
    message = json.dumps(msgdict)
    return CircuitStatus(_STATUS_MAP[h_status], message)
