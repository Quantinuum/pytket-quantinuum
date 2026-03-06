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

# NB: This test has been placed in a separate file from api_test.py to work around an
# issue on the MacOS CI, whereby pytest would hang indefinitely after the collection
# phase.

from http import HTTPStatus
from io import StringIO
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import requests
from pytket.architecture import FullyConnected
from pytket.backends import ResultHandle, StatusEnum
from pytket.circuit import Circuit
from pytket.extensions.quantinuum._metadata import __extension_version__
from requests_mock.mocker import Mocker

from pytket.extensions.quantinuum.backends import (
    Language,
    QuantinuumBackend,
    have_pecos,
)
from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    QuantinuumConfigCredentialStorage,
)
from pytket.extensions.quantinuum.backends.quantinuum import DEFAULT_API_HANDLER


def test_default_login_flow(
    requests_mock: Mocker,
    mock_credentials: tuple[str, str],
    mock_token: str,
    mock_machine_info: dict[str, Any],
    monkeypatch: Any,
) -> None:
    """Test that when an api_handler is not provided to
    QuantinuumBackend we use the DEFAULT_API_HANDLER.

    Demonstrate that the login endpoint is only called one time
    for the session when not providing an api_handler argument to
    QuantinuumBackend.
    """

    DEFAULT_API_HANDLER.delete_authentication()

    fake_device = mock_machine_info["name"]
    fake_job_id = "abc-123"

    login_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/login",
        json={
            "id-token": mock_token,
            "refresh-token": mock_token,
        },
        headers={"Content-Type": "application/json"},
    )

    job_submit_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/job",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    job_status_route = requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/job/{fake_job_id}?websocket=true",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )
    requests_mock.register_uri(
        "GET",
        "https://qapi.quantinuum.com/v1/machine/?config=true",
        json=[mock_machine_info],
        headers={"Content-Type": "application/json"},
    )

    username, pwd = mock_credentials
    # fake user input from stdin
    monkeypatch.setattr("sys.stdin", StringIO(username + "\n"))
    monkeypatch.setattr("getpass.getpass", lambda prompt: pwd)

    backend_1 = QuantinuumBackend(
        device_name=fake_device,
    )
    backend_2 = QuantinuumBackend(
        device_name=fake_device,
    )

    circ = Circuit(2, name="default_login_flow_test").H(0).CX(0, 1).measure_all()
    circ = backend_1.get_compiled_circuit(circ)
    circ = backend_2.get_compiled_circuit(circ)

    backend_1.process_circuits(
        circuits=[circ, circ],
        n_shots=10,
        valid_check=False,
    )
    backend_2.process_circuits(
        circuits=[circ, circ],
        n_shots=10,
        valid_check=False,
    )

    # We expect /login to be called once globally.
    assert login_route.called_once
    assert job_submit_route.call_count == 4
    assert job_status_route.call_count == 0


def test_custom_login_flow(
    requests_mock: Mocker,
    mock_credentials: tuple[str, str],
    mock_token: str,
    mock_machine_info: dict[str, Any],
    monkeypatch: Any,
) -> None:
    """Test that when an api_handler is provided to
    QuantinuumBackend we use that handler and acquire
    tokens for each.
    """

    DEFAULT_API_HANDLER.delete_authentication()

    fake_device = mock_machine_info["name"]
    fake_job_id = "abc-123"

    login_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/login",
        json={
            "id-token": mock_token,
            "refresh-token": mock_token,
        },
        headers={"Content-Type": "application/json"},
    )

    job_submit_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/job",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    job_status_route = requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/job/{fake_job_id}?websocket=true",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )
    requests_mock.register_uri(
        "GET",
        "https://qapi.quantinuum.com/v1/machine/?config=true",
        json=[mock_machine_info],
        headers={"Content-Type": "application/json"},
    )

    backend_1 = QuantinuumBackend(
        device_name=fake_device,
        api_handler=QuantinuumAPI(  # type: ignore # pylint: disable=unexpected-keyword-arg
            _QuantinuumAPI__user_name="user1",
            _QuantinuumAPI__pwd="securepassword",
        ),
    )
    backend_2 = QuantinuumBackend(
        device_name=fake_device,
        api_handler=QuantinuumAPI(  # type: ignore # pylint: disable=unexpected-keyword-arg
            _QuantinuumAPI__user_name="user2",
            _QuantinuumAPI__pwd="insecurepassword",
        ),
    )

    qcc = QuantinuumConfigCredentialStorage()
    qcc.save_user_name("user3")
    qcc.save_refresh_token("token")

    backend_3 = QuantinuumBackend(
        device_name=fake_device,
        api_handler=QuantinuumAPI(qcc),
    )

    circ = Circuit(2, name="default_login_flow_test").H(0).CX(0, 1).measure_all()
    circ = backend_1.get_compiled_circuit(circ)
    circ = backend_2.get_compiled_circuit(circ)

    username, pwd = mock_credentials
    # fake user input from stdin
    monkeypatch.setattr("sys.stdin", StringIO(username + "\n"))
    monkeypatch.setattr("getpass.getpass", lambda prompt: pwd)
    circ = backend_3.get_compiled_circuit(circ)

    backend_1.process_circuits(
        circuits=[circ, circ],
        n_shots=10,
        valid_check=False,
    )
    backend_2.process_circuits(
        circuits=[circ, circ],
        n_shots=10,
        valid_check=False,
    )
    backend_3.process_circuits(
        circuits=[circ, circ],
        n_shots=10,
        valid_check=False,
    )

    # QuantinuumConfigCredentialStorage doesn't require re-login
    backend_4 = QuantinuumBackend(
        device_name=fake_device,
        api_handler=QuantinuumAPI(QuantinuumConfigCredentialStorage()),
    )
    circ = backend_4.get_compiled_circuit(circ)
    backend_4.process_circuits(
        circuits=[circ, circ],
        n_shots=10,
        valid_check=False,
    )
    # We expect /login to be called for each api_handler except backend_4.
    assert login_route.call_count == 3
    assert job_submit_route.call_count == 8
    assert job_status_route.call_count == 0

    # Remove tokens stored with QuantinuumConfig
    backend_3.api_handler.delete_authentication()


def test_mfa_login_flow(  # noqa: PLR0913
    requests_mock: Mocker,
    mock_credentials: tuple[str, str],
    mock_token: str,
    mock_mfa_code: str,
    mock_machine_info: dict[str, Any],
    monkeypatch: Any,
) -> None:
    """Test that the MFA authentication works as expected"""

    DEFAULT_API_HANDLER.delete_authentication()

    fake_device = mock_machine_info["name"]

    def match_mfa_request(request: requests.PreparedRequest) -> bool:
        return "code" in request.body  # type: ignore

    def match_normal_request(request: requests.PreparedRequest) -> bool:
        return "code" not in request.body  # type: ignore

    mfa_login_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/login",
        json={
            "id-token": mock_token,
            "refresh-token": mock_token,
        },
        headers={"Content-Type": "application/json"},
        additional_matcher=match_mfa_request,  # type: ignore
    )
    normal_login_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/login",
        json={
            "error": {"code": 73},
        },
        headers={"Content-Type": "application/json"},
        additional_matcher=match_normal_request,  # type: ignore
        status_code=HTTPStatus.UNAUTHORIZED,
    )

    username, pwd = mock_credentials
    # fake user input from stdin
    inputs = iter([username + "\n", mock_mfa_code + "\n"])
    monkeypatch.setattr("builtins.input", lambda msg: next(inputs))
    monkeypatch.setattr("getpass.getpass", lambda prompt: pwd)

    backend = QuantinuumBackend(
        device_name=fake_device,
    )
    backend.login()

    assert normal_login_route.called_once
    # Check that the mfa login has been invoked
    assert mfa_login_route.called_once
    assert backend.api_handler._cred_store.id_token is not None  # noqa: SLF001
    assert backend.api_handler._cred_store.refresh_token is not None  # noqa: SLF001


@patch("pytket.extensions.quantinuum.backends.api_wrappers.microsoft_login")
def test_federated_login(  # noqa: PLR0913
    mock_microsoft_login: MagicMock,
    requests_mock: Mocker,
    mock_credentials: tuple[str, str],
    mock_token: str,
    mock_ms_provider_token: str,
    mock_machine_info: dict[str, Any],
) -> None:
    """Test that the federated authentication works as expected"""
    DEFAULT_API_HANDLER.delete_authentication()

    fake_device = mock_machine_info["name"]

    backend = QuantinuumBackend(
        device_name=fake_device,
        provider="microsoft",
    )
    login_route = requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/login",
        json={
            "id-token": mock_token,
            "refresh-token": mock_token,
        },
        headers={"Content-Type": "application/json"},
    )
    mock_microsoft_login.return_value = (mock_credentials[0], mock_ms_provider_token)
    backend.login()

    mock_microsoft_login.assert_called_once()
    assert login_route.called_once
    assert backend.api_handler._cred_store.id_token is not None  # noqa: SLF001
    assert backend.api_handler._cred_store.refresh_token is not None  # noqa: SLF001


def test_federated_login_wrong_provider(
    mock_machine_info: dict[str, Any],
) -> None:
    """Test that the federated authentication works as expected"""
    DEFAULT_API_HANDLER.delete_authentication()

    fake_device = mock_machine_info["name"]

    backend = QuantinuumBackend(
        device_name=fake_device,
        provider="wrong provider",
    )
    with pytest.raises(RuntimeError) as e:
        backend.login()
        err_msg = "Unsupported provider for login"
        assert err_msg in str(e.value)


def test_resumed_batching(
    requests_mock: Mocker,
    mock_quum_api_handler: QuantinuumAPI,
    sample_machine_infos: dict[str, Any],
) -> None:
    """Test that you can resume using a batch."""

    fake_job_id = "abc-123"

    requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/job",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/job/{fake_job_id}?websocket=true",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )
    requests_mock.register_uri(
        "GET",
        "https://qapi.quantinuum.com/v1/machine/?config=true",
        json=sample_machine_infos,
        headers={"Content-Type": "application/json"},
    )

    backend = QuantinuumBackend(
        device_name="H2-1E",
        api_handler=mock_quum_api_handler,
    )

    circ = Circuit(2, name="batching_test").H(0).CX(0, 1).measure_all()
    circ = backend.get_compiled_circuit(circ)

    h1 = backend.start_batch(500, circ, n_shots=10, valid_check=False)

    submitted_json = {}
    if requests_mock.last_request:
        # start batch makes two requests
        submitted_json = requests_mock.request_history[-2].json()

    assert "batch-exec" in submitted_json
    assert submitted_json["batch-exec"] == 500
    assert "batch-end" not in submitted_json

    _ = backend.add_to_batch(h1, circ, n_shots=10, valid_check=False, batch_end=True)

    if requests_mock.last_request:
        submitted_json = requests_mock.last_request.json()
    assert submitted_json["batch-exec"] == backend.get_jobid(h1)
    assert "batch-end" in submitted_json


def test_available_devices(
    requests_mock: Mocker,
    mock_quum_api_handler: QuantinuumAPI,
    mock_machine_info: dict[str, Any],
) -> None:
    """Test that we can get a list of available devices."""
    requests_mock.register_uri(
        "GET",
        "https://qapi.quantinuum.com/v1/machine/?config=true",
        json=[mock_machine_info],
        headers={"Content-Type": "application/json"},
    )

    devices = QuantinuumBackend.available_devices(api_handler=mock_quum_api_handler)
    backinfo0 = devices[0]

    assert backinfo0.device_name == mock_machine_info["name"]
    assert backinfo0.architecture == FullyConnected(mock_machine_info["n_qubits"], "q")
    assert backinfo0.version == __extension_version__
    assert backinfo0.supports_fast_feedforward
    assert backinfo0.supports_midcircuit_measurement
    assert backinfo0.supports_reset
    assert backinfo0.n_cl_reg == 120
    assert (
        backinfo0.misc.items()
        >= {
            "wasm": True,
            "batching": True,
            "max_classical_register_width": 32,
            "syntax_checker": "H9-27SC",
            "n_gate_zones": "5",
            "max_n_shots": 10000,
            "system_type": "hardware",
            "connectivity": "all-to-all",
            "emulator": "H9-27E",
        }.items()
    )

    assert backinfo0.name == "QuantinuumBackend"

    if have_pecos():
        backinfo1 = devices[1]
        assert backinfo1.device_name == mock_machine_info["name"] + "LE"
        assert backinfo1.architecture == FullyConnected(
            mock_machine_info["n_qubits"], "q"
        )
        assert backinfo1.version == __extension_version__
        assert backinfo1.supports_fast_feedforward
        assert backinfo1.supports_midcircuit_measurement
        assert backinfo1.supports_reset
        assert backinfo1.n_cl_reg == 120
        assert (
            backinfo1.misc.items()
            >= {
                "wasm": True,
                "batching": False,
                "max_classical_register_width": 32,
                "syntax_checker": "H9-27SC",
                "n_gate_zones": "5",
                "max_n_shots": 10000,
                "system_type": "local_emulator",
                "connectivity": "all-to-all",
            }.items()
        )
        assert backinfo1.name == "QuantinuumBackend"


def test_submit_qasm_api(
    requests_mock: Mocker,
    mock_quum_api_handler: QuantinuumAPI,
    sample_machine_infos: dict[str, Any],
) -> None:
    """Test that we can submit a QASM program."""

    fake_job_id = "abc-123"

    requests_mock.register_uri(
        "POST",
        "https://qapi.quantinuum.com/v1/job",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )

    requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/job/{fake_job_id}?websocket=true",
        json={"job": fake_job_id},
        headers={"Content-Type": "application/json"},
    )
    requests_mock.register_uri(
        "GET",
        "https://qapi.quantinuum.com/v1/machine/?config=true",
        json=sample_machine_infos,
        headers={"Content-Type": "application/json"},
    )

    backend = QuantinuumBackend(
        device_name="H2-1SC",
        api_handler=mock_quum_api_handler,
    )

    qasm = """
    OPENQASM 2.0;
    include "hqslib1.inc";
    """
    h1 = backend.submit_program(Language.QASM, qasm, n_shots=10)

    assert h1[0] == fake_job_id

    submitted_json = {}
    if requests_mock.last_request:
        # start batch makes two requests
        submitted_json = requests_mock.last_request.json()

    assert submitted_json["program"] == qasm
    assert submitted_json["count"] == 10


def test_get_partial_result(
    requests_mock: Mocker,
    mock_quum_api_handler: QuantinuumAPI,
    mock_machine_info: dict[str, Any],
) -> None:
    """Test that we can get partial results."""
    queued_job_id = "abc-123"
    requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/job/{queued_job_id}?websocket=true",
        json={"job": "abc-123", "name": "job", "status": "queued"},
        headers={"Content-Type": "application/json"},
    )
    running_job_id = "abc-456"
    requests_mock.register_uri(
        "GET",
        f"https://qapi.quantinuum.com/v1/job/{running_job_id}?websocket=true",
        json={
            "job": "abc-123",
            "name": "job",
            "status": "running",
            "results": {"c": ["10110", "10000", "10110", "01100", "10000"]},
        },
        headers={"Content-Type": "application/json"},
    )
    requests_mock.register_uri(
        "GET",
        "https://qapi.quantinuum.com/v1/machine/?config=true",
        json=[mock_machine_info],
        headers={"Content-Type": "application/json"},
    )
    backend = QuantinuumBackend(device_name="H9-27", api_handler=mock_quum_api_handler)
    h1 = ResultHandle(queued_job_id, "null", -1)
    res, status = backend.get_partial_result(h1)
    assert res is None
    assert status.status == StatusEnum.QUEUED

    h2 = ResultHandle(running_job_id, "null", -1)
    res, status = backend.get_partial_result(h2)
    assert res is not None
    assert status.status == StatusEnum.RUNNING
