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

from io import StringIO
from typing import Any

from requests_mock.mocker import Mocker

from pytket.extensions.quantinuum.backends.api_wrappers import QuantinuumAPI
from pytket.extensions.quantinuum.backends.credential_storage import (
    MemoryCredentialStorage,
)


def test_quum_login(
    mock_quum_api_handler: QuantinuumAPI,
    mock_credentials: tuple[str, str],
    mock_token: str,
) -> None:
    """Test that credentials are storable and deletable using
    the QuantinuumQAPI handler."""

    _, pwd = mock_credentials

    assert isinstance(mock_quum_api_handler._cred_store, MemoryCredentialStorage)  # noqa: SLF001
    # Check credentials are retrievable
    assert mock_quum_api_handler._cred_store._password == pwd  # noqa: SLF001
    assert mock_quum_api_handler._cred_store.refresh_token == mock_token  # noqa: SLF001
    assert mock_quum_api_handler._cred_store.id_token == mock_token  # noqa: SLF001

    # Delete authentication and verify
    mock_quum_api_handler.delete_authentication()
    assert mock_quum_api_handler._cred_store.id_token is None  # noqa: SLF001
    assert mock_quum_api_handler._cred_store._password is None  # type: ignore  # noqa: SLF001
    assert mock_quum_api_handler._cred_store.refresh_token is None  # noqa: SLF001


def test_machine_status(
    requests_mock: Mocker,
    mock_quum_api_handler: QuantinuumAPI,
) -> None:
    """Test that we can retrieve the machine state via Quantinuum endpoint."""

    machine_name = "quum-LT-S1-APIVAL"
    mock_machine_state = "online"

    mock_url = f"https://qapi.quantinuum.com/v1/machine/{machine_name}"

    requests_mock.register_uri(
        "GET",
        mock_url,
        json={"state": mock_machine_state},
        headers={"Content-Type": "application/json"},
    )

    assert mock_quum_api_handler.status(machine_name) == mock_machine_state

    # Delete authentication tokens to clean them from memory
    mock_quum_api_handler.delete_authentication()


def test_full_login(
    requests_mock: Mocker,
    mock_credentials: tuple[str, str],
    mock_token: str,
    monkeypatch: Any,
) -> None:
    """Test that we can perform the login flow."""
    username, pwd = mock_credentials

    mock_url = "https://qapi.quantinuum.com/v1/login"

    requests_mock.register_uri(
        "POST",
        mock_url,
        json={
            "id-token": mock_token,
            "refresh-token": "refresh" + mock_token,
        },
        headers={"Content-Type": "application/json"},
    )

    # fake user input from stdin
    monkeypatch.setattr("sys.stdin", StringIO(username + "\n"))
    monkeypatch.setattr("getpass.getpass", lambda prompt: pwd)

    api_handler = QuantinuumAPI()
    # emulate no pytket config stored email address
    api_handler.full_login()

    assert isinstance(api_handler._cred_store, MemoryCredentialStorage)  # noqa: SLF001
    assert api_handler._cred_store.id_token == mock_token  # noqa: SLF001
    assert api_handler._cred_store.refresh_token == "refresh" + mock_token  # noqa: SLF001
    assert api_handler._cred_store._id_token_timeout is not None  # noqa: SLF001
    assert api_handler._cred_store._refresh_token_timeout is not None  # noqa: SLF001

    assert api_handler._cred_store._password is None  # noqa: SLF001
    assert api_handler._cred_store._user_name is None  # noqa: SLF001

    api_handler.delete_authentication()

    assert all(
        val is None
        for val in (
            api_handler._cred_store.id_token,  # noqa: SLF001
            api_handler._cred_store.refresh_token,  # noqa: SLF001
            api_handler._cred_store._id_token_timeout,  # noqa: SLF001
            api_handler._cred_store._refresh_token_timeout,  # noqa: SLF001
        )
    )
