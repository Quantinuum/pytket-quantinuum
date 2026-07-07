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

import os
from typing import Any

import jwt
import pytest
from _pytest.fixtures import SubRequest
from requests_mock.mocker import Mocker

from pytket.extensions.quantinuum import QuantinuumBackend
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    MemoryCredentialStorage,
)

ALL_QUANTUM_HARDWARE_NAMES = []

if not os.getenv("PYTKET_REMOTE_QUANTINUUM_EMULATORS_ONLY", 0):  # noqa: PLW1508
    ALL_QUANTUM_HARDWARE_NAMES.extend(
        [
            "H2-1",
        ]
    )

ALL_SIMULATOR_NAMES = [
    "H2-1E",
]

ALL_SYNTAX_CHECKER_NAMES = [
    "H2-1SC",
]

ALL_LOCAL_SIMULATOR_NAMES = [
    "H2-1LE",
]

ALL_DEVICE_NAMES = [
    *ALL_QUANTUM_HARDWARE_NAMES,
    *ALL_SIMULATOR_NAMES,
    *ALL_SYNTAX_CHECKER_NAMES,
]


def pytest_configure() -> None:
    """Define global symbols used by the tests.

    Note: we need to do this as part of the pytest_configure as these symbols
    are used while parametrizing the tests and not as fixtures."""

    pytest.ALL_DEVICE_NAMES = ALL_DEVICE_NAMES  # type: ignore
    pytest.ALL_SYNTAX_CHECKER_NAMES = ALL_SYNTAX_CHECKER_NAMES  # type: ignore
    pytest.ALL_SIMULATOR_NAMES = ALL_SIMULATOR_NAMES  # type: ignore
    pytest.ALL_QUANTUM_HARDWARE_NAMES = ALL_QUANTUM_HARDWARE_NAMES  # type: ignore
    pytest.ALL_LOCAL_SIMULATOR_NAMES = ALL_LOCAL_SIMULATOR_NAMES  # type: ignore


def pytest_make_parametrize_id(
    config: pytest.Config, val: object, argname: str
) -> str | None:
    """Custom ids for the parametrized tests."""
    if isinstance(val, QuantinuumBackend):
        return val._device_name  # noqa: SLF001
    if isinstance(val, dict):
        return val.get("device_name")
    return None


@pytest.fixture()
def mock_credentials() -> tuple[str, str]:
    username = "mark.quantinuum@mail.com"
    pwd = "1906"
    return (username, pwd)


@pytest.fixture()
def mock_token() -> str:
    # A mock token that expires in 2073
    token_payload = {"exp": 3278815149.143694}
    mock_token = jwt.encode(token_payload, key="0", algorithm="HS256")
    return str(mock_token)


@pytest.fixture()
def mock_mfa_code() -> str:
    return "mfa code"


@pytest.fixture()
def mock_ms_provider_token() -> str:
    return "ms token"


@pytest.fixture()
def mock_machine_info() -> dict[str, Any]:
    return {
        "wasm": True,
        "batching": True,
        "benchmarks": {"qv": {"date": "2024-04-04", "value": 1048576.0}},
        "max_classical_register_width": 32,
        "gateset": ["RZZ", "Rxxyyzz", "Rz", "U1q", "ZZ"],
        "name": "H9-27",
        "syntax_checker": "H9-27SC",
        "n_gate_zones": "5",
        "noise_specs": {
            "date": "2024-02-04",
            "spam_error": {
                "p_meas_1_unc": 0.000199,
                "p_meas_0": 0.00095,
                "p_meas_1": 0.00397,
                "p_meas_0_unc": 9.74e-05,
            },
            "crosstalk_error": {
                "p_crosstalk_meas_unc": 1.02e-06,
                "p_crosstalk_meas": 1.453e-05,
            },
            "memory_error": {
                "memory_error_unc": 2.52e-05,
                "memory_error": 0.000208,
            },
            "1q_gate_error": {"p1": 2.08e-05, "p1_unc": 2.77e-06},
            "2q_gate_error": {"p2_unc": 2.85e-05, "p2": 0.000882},
        },
        "max_n_shots": 10000,
        "n_qubits": 20,
        "n_classical_registers": 120,
        "system_type": "hardware",
        "connectivity": "all-to-all",
        "emulator": "H9-27E",
    }


@pytest.fixture()
def sample_machine_infos() -> list[dict[str, Any]]:
    return [
        {
            "wasm": True,
            "batching": True,
            "n_gate_zones": "4",
            "noise_specs": {
                "crosstalk_probability": {
                    "p_crosstalk_meas": 7.4e-06,
                    "p_crosstalk_init": 9.6e-06,
                },
                "1q_fault_probability": {"p1": 2.9e-05},
                "bit_flip_measurement_probability": {
                    "p_meas_0": 0.0005,
                    "p_meas_1": 0.0025,
                },
                "dephasing_rates": {
                    "linear_dephasing_rate": 0.0028,
                    "quadratic_dephasing_rate": 0.043,
                },
                "2q_fault_probability": {"p2": 0.00128},
                "ratio_spontaneous_emission": {
                    "p1_emission_ratio": 0.32,
                    "p2_emission_ratio": 0.48,
                },
                "fit_parameters": {
                    "przz_c": 1.518,
                    "przz_d": 0.241,
                    "przz_a": 1.518,
                    "przz_b": 0.241,
                    "przz_power": 1.0,
                },
                "coherent_to_incoherent_factor": 2.0,
                "init_fault_probability": {"p_init": 4e-05},
            },
            "max_n_shots": 10000,
            "n_qubits": 56,
            "n_classical_registers": 50,
            "system_type": "emulator",
            "max_classical_register_width": 32,
            "gateset": ["RZZ", "Rxxyyzz", "Rz", "U1q", "ZZ"],
            "name": "H2-1E",
            "connectivity": "all-to-all",
        },
        {
            "wasm": True,
            "max_n_shots": 10000,
            "n_qubits": 56,
            "n_classical_registers": 50,
            "system_type": "syntax checker",
            "max_classical_register_width": 63,
            "gateset": ["RZZ", "Rxxyyzz", "Rz", "TK2", "U1q", "ZZ"],
            "name": "H2-1SC",
        },
        {
            "wasm": True,
            "batching": True,
            "benchmarks": {"qv": {"date": "2024-05-31", "value": 262144.0}},
            "max_classical_register_width": 63,
            "gateset": ["RZZ", "Rxxyyzz", "Rz", "U1q", "ZZ"],
            "name": "H2-1",
            "syntax_checker": "H2-1SC",
            "n_gate_zones": "4",
            "noise_specs": {
                "date": "2024-05-20",
                "spam_error": {
                    "p_meas_1_unc": 0.0002,
                    "p_meas_0": 0.0005,
                    "p_meas_1": 0.0025,
                    "p_meas_0_unc": 8e-05,
                },
                "crosstalk_error": {
                    "p_crosstalk_meas_unc": 8e-07,
                    "p_crosstalk_meas": 7.4e-06,
                },
                "memory_error": {"memory_error_unc": 2e-05, "memory_error": 0.0005},
                "1q_gate_error": {"p1": 2.9e-05, "p1_unc": 4e-06},
                "2q_gate_error": {"p2_unc": 8e-05, "p2": 0.00128},
            },
            "max_n_shots": 10000,
            "n_qubits": 56,
            "n_classical_registers": 50,
            "system_type": "hardware",
            "connectivity": "all-to-all",
            "emulator": "H2-1E",
        },
    ]


@pytest.fixture(name="mock_quum_api_handler", params=[True, False])
def fixture_mock_quum_api_handler(
    request: SubRequest,
    requests_mock: Mocker,
    mock_credentials: tuple[str, str],
    mock_token: str,
) -> QuantinuumAPI:
    """A logged-in QuantinuumQAPI fixture.
    After using this fixture in a test, call:
        mock_quum_api_handler.delete_authentication()
    To remove mock tokens from memory.
    """

    username, pwd = mock_credentials

    mock_url = "https://qapi.quantinuum.com/v1/login"

    requests_mock.register_uri(
        "POST",
        mock_url,
        json={
            "id-token": mock_token,
            "refresh-token": mock_token,
        },
        headers={"Content-Type": "application/json"},
    )

    cred_store = MemoryCredentialStorage()
    cred_store.save_user_name(username)
    cred_store._password = pwd  # noqa: SLF001

    # Construct QuantinuumQAPI and login
    api_handler = QuantinuumAPI()

    # Add the credential storage seperately in line with fixture parameters
    api_handler._cred_store = cred_store  # noqa: SLF001
    api_handler.login()

    return api_handler


@pytest.fixture(scope="module", name="authenticated_quum_handler")
def fixture_authenticated_quum() -> QuantinuumAPI:
    # Authenticated QuantinuumAPI used for the remote tests
    # The credentials are taken from the env variables:
    # PYTKET_REMOTE_QUANTINUUM_USERNAME and PYTKET_REMOTE_QUANTINUUM_PASSWORD
    return QuantinuumAPI(  # type: ignore # pylint: disable=unexpected-keyword-arg
        api_url="https://qapi.quantinuum.com/",
        _QuantinuumAPI__user_name=os.getenv("PYTKET_REMOTE_QUANTINUUM_USERNAME"),
        _QuantinuumAPI__pwd=os.getenv("PYTKET_REMOTE_QUANTINUUM_PASSWORD"),
    )


@pytest.fixture(name="authenticated_quum_backend_prod")
def fixture_authenticated_quum_backend_prod(
    request: SubRequest,
    authenticated_quum_handler: QuantinuumAPI,
) -> QuantinuumBackend:
    # Authenticated QuantinuumBackend used for the remote tests
    # The credentials are taken from the env variables:
    # PYTKET_REMOTE_QUANTINUUM_USERNAME and PYTKET_REMOTE_QUANTINUUM_PASSWORD
    # Note: this fixture should only be used in tests where PYTKET_RUN_REMOTE_TESTS
    #       is true, by marking it with @parametrize, using the
    #       "authenticated_quum_backend_prod" as parameter and `indirect=True`

    # By default, the backend is created with device_name="H2-1SC" only,
    # but other params can be specified when parametrizing the
    # authenticated_quum_backend_prod
    if (not hasattr(request, "param")) or request.param is None:
        backend = QuantinuumBackend("H2-1SC", api_handler=authenticated_quum_handler)
    else:
        backend = QuantinuumBackend(
            api_handler=authenticated_quum_handler, **request.param
        )
    # In case machine_debug was specified by mistake in the params
    backend._MACHINE_DEBUG = False  # noqa: SLF001

    return backend


@pytest.fixture(scope="module", name="authenticated_quum_handler_qa")
def fixture_authenticated_quum_qa() -> QuantinuumAPI:
    # Authenticated QA QuantinuumAPI used for the remote tests
    # The credentials are taken from the env variables:
    # PYTKET_REMOTE_QUANTINUUM_USERNAME_QA and PYTKET_REMOTE_QUANTINUUM_PASSWORD_QA
    return QuantinuumAPI(  # type: ignore # pylint: disable=unexpected-keyword-arg
        api_url="https://hqapi.quantinuum.com/",
        _QuantinuumAPI__user_name=os.getenv("PYTKET_REMOTE_QUANTINUUM_USERNAME_QA"),
        _QuantinuumAPI__pwd=os.getenv("PYTKET_REMOTE_QUANTINUUM_PASSWORD_QA"),
    )


@pytest.fixture(name="authenticated_quum_backend_qa")
def fixture_authenticated_quum_backend_qa(
    request: SubRequest,
    authenticated_quum_handler_qa: QuantinuumAPI,
) -> QuantinuumBackend:
    # Authenticated QuantinuumBackend used for the remote tests
    # The credentials are taken from the env variables:
    # PYTKET_REMOTE_QUANTINUUM_USERNAME and PYTKET_REMOTE_QUANTINUUM_PASSWORD
    # Note: this fixture should only be used in tests where PYTKET_RUN_REMOTE_TESTS
    #       is true, by marking it with @parametrize, using the
    #       "authenticated_quum_backend_qa" as parameter and `indirect=True`

    # By default, the backend is created with device_name="H2-1SC" only,
    # but other params can be specified when parametrizing the
    # authenticated_quum_backend_qa
    if (not hasattr(request, "param")) or request.param is None:
        backend = QuantinuumBackend("H2-1SC", api_handler=authenticated_quum_handler_qa)
    else:
        backend = QuantinuumBackend(
            api_handler=authenticated_quum_handler_qa, **request.param
        )
    # In case machine_debug was specified by mistake in the params
    backend._MACHINE_DEBUG = False  # noqa: SLF001

    return backend
