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

import asyncio
import contextlib
import getpass
import json
import time
from http import HTTPStatus
from typing import Any, cast

import nest_asyncio  # type: ignore
from requests import Session
from requests.models import Response
from websockets import connect, exceptions

from .config import QuantinuumConfig
from .credential_storage import CredentialStorage, MemoryCredentialStorage
from .federated_login import microsoft_login

# This is necessary for use in Jupyter notebooks to allow for nested asyncio loops
# May fail in some cloud environments: ignore.
with contextlib.suppress(RuntimeError, ValueError):
    nest_asyncio.apply()


class QuantinuumAPIError(Exception):
    pass


class _OverrideManager:
    def __init__(
        self,
        api_handler: "QuantinuumAPI",
        timeout: int | None = None,
        retry_timeout: int | None = None,
    ):
        self._timeout = timeout
        self._retry = retry_timeout
        self.api_handler = api_handler
        self._orig_timeout = api_handler.timeout
        self._orig_retry = api_handler.retry_timeout

    def __enter__(self) -> None:
        if self._timeout is not None:
            self.api_handler.timeout = self._timeout
        if self._retry is not None:
            self.api_handler.retry_timeout = self._retry

    def __exit__(self, exc_type, exc_value, traceback) -> None:  # type: ignore
        self.api_handler.timeout = self._orig_timeout
        self.api_handler.retry_timeout = self._orig_retry


class QuantinuumAPI:
    """
    Interface to the Quantinuum online remote API.
    """

    JOB_DONE = ["failed", "completed", "canceled"]  # noqa: RUF012

    DEFAULT_API_URL = "https://qapi.quantinuum.com/"

    AZURE_PROVIDER = "microsoft"

    # Quantinuum API error codes
    # mfa verification code is required during login
    ERROR_CODE_MFA_REQUIRED = 73

    def __init__(  # noqa: PLR0913
        self,
        token_store: CredentialStorage | None = None,
        api_url: str | None = None,
        api_version: int = 1,
        use_websocket: bool = True,
        provider: str | None = None,
        support_mfa: bool = True,
        session: Session | None = None,
        __user_name: str | None = None,
        __pwd: str | None = None,
    ):
        """Initialize Quantinuum API client.

        :param token_store: JWT Token store, defaults to None
            A new :py:class:`~.MemoryCredentialStorage` will be initialised
            if None is provided.
        :param api_url: _description_, defaults to DEFAULT_API_URL
        :param api_version: API version, defaults to 1
        :param use_websocket: Whether to use websocket to retrieve, defaults to True
        :param support_mfa: Whether to wait for the user to input the auth code,
            defaults to True
        :param session: Session for HTTP requests, defaults to None
            A new ``requests.Session`` will be initialised if None
            is provided
        """
        self.online = True

        self.url = f"{api_url if api_url else self.DEFAULT_API_URL}v{api_version}/"  # noqa: FURB110

        if session is None:
            self.session = Session()
        else:
            self.session = session

        self._cred_store: CredentialStorage
        if token_store is None:
            self._cred_store = MemoryCredentialStorage()
        else:
            self._cred_store = token_store

        # if __user_name is None and MemoryCredentialStorage is used
        # and there is a cached username in the config file,
        # load that username into memory
        if __user_name is None and isinstance(
            self._cred_store, MemoryCredentialStorage
        ):
            config = QuantinuumConfig.from_default_config_file()
            if config.username is not None:
                self._cred_store.save_user_name(config.username)
        elif __user_name is not None:
            # username will be cached if persistent storage is used,
            # otherwise it will be stored in memory
            self._cred_store.save_user_name(__user_name)
        if __pwd is not None and isinstance(self._cred_store, MemoryCredentialStorage):
            self._cred_store._password = __pwd  # noqa: SLF001

        self.api_version = api_version
        self.use_websocket = use_websocket
        self.provider = provider
        self.support_mfa = support_mfa

        self.ws_timeout = 180
        self.retry_timeout = 5
        self.timeout: int | None = None  # don't timeout by default

    def _override_timeouts(
        self, timeout: int | None = None, retry_timeout: int | None = None
    ) -> _OverrideManager:
        return _OverrideManager(self, timeout=timeout, retry_timeout=retry_timeout)

    def _request_tokens(self, user: str, pwd: str) -> None:
        """Method to send login request to machine api and save tokens."""
        body = {"email": user, "password": pwd}
        try:
            # send request to login
            response = self.session.post(
                f"{self.url}login",
                json.dumps(body),
            )

            # handle mfa verification
            if response.status_code == HTTPStatus.UNAUTHORIZED:
                error_code = response.json()["error"]["code"]
                if error_code == self.ERROR_CODE_MFA_REQUIRED:
                    if not self.support_mfa:
                        raise QuantinuumAPIError(
                            "This API instance does not support MFA login."
                        )
                    # get a mfa code from user input
                    mfa_code = input("Enter your MFA verification code: ")
                    body["code"] = mfa_code

                    # resend request to login
                    response = self.session.post(
                        f"{self.url}login",
                        json.dumps(body),
                    )

            self._response_check(response, "Login")
            resp_dict = response.json()
            self._cred_store.save_tokens(
                resp_dict["id-token"], resp_dict["refresh-token"]
            )

        finally:
            del user
            del pwd
            del body

    def _request_tokens_federated(self) -> None:
        """Method to perform federated login and save tokens."""

        if self.provider is not None and self.provider.lower() == self.AZURE_PROVIDER:
            _, token = microsoft_login()
        else:
            raise RuntimeError(
                "Unsupported provider for login", HTTPStatus.UNAUTHORIZED
            )

        body = {"provider-token": token}

        try:
            response = self.session.post(
                f"{self.url}login",
                json.dumps(body),
            )
            self._response_check(response, "Login")
            resp_dict = response.json()
            self._cred_store.save_tokens(
                resp_dict["id-token"], resp_dict["refresh-token"]
            )
        finally:
            del body

    def _refresh_id_token(self, refresh_token: str) -> None:
        """Method to refresh ID token using a refresh token."""
        body = {"refresh-token": refresh_token}
        try:
            # send request to login
            response = self.session.post(
                f"{self.url}login",
                json.dumps(body),
            )

            message = response.json()

            if (
                response.status_code == HTTPStatus.BAD_REQUEST
                and message is not None
                and "Invalid Refresh Token" in message["error"]["text"]
            ):
                # ask user for credentials to login again
                self.full_login()

            else:
                self._response_check(response, "Token Refresh")
                self._cred_store.save_tokens(
                    message["id-token"], message["refresh-token"]
                )

        finally:
            del refresh_token
            del body

    def _get_credentials(self) -> tuple[str, str]:
        """Method to ask for user's credentials"""
        user_name = self._cred_store.user_name
        pwd = None
        if isinstance(self._cred_store, MemoryCredentialStorage):
            pwd = self._cred_store._password  # noqa: SLF001

        if not user_name:
            user_name = input("Enter your Quantinuum email: ")

        if not pwd:
            pwd = getpass.getpass(prompt="Enter your Quantinuum password: ")

        return user_name, pwd

    def full_login(self) -> None:
        """Ask for user credentials from std input and update JWT tokens"""
        if self.provider is None:
            self._request_tokens(*self._get_credentials())
        else:
            self._request_tokens_federated()

    def login(self) -> str:
        """This methods checks if we have a valid (non-expired) id-token
        and returns it, otherwise it gets a new one with refresh-token.
        If refresh-token doesn't exist, it asks user for credentials.

        :return: (str) login token
        """
        # check if refresh_token exists
        refresh_token = self._cred_store.refresh_token
        if refresh_token is None:
            self.full_login()
            refresh_token = self._cred_store.refresh_token

        if refresh_token is None:
            raise QuantinuumAPIError(
                "Unable to retrieve refresh token or authenticate."
            )

        # check if id_token exists
        id_token = self._cred_store.id_token
        if id_token is None:
            self._refresh_id_token(refresh_token)
            id_token = self._cred_store.id_token

        if id_token is None:
            raise QuantinuumAPIError("Unable to retrieve id token or refresh or login.")

        return id_token

    def delete_authentication(self) -> None:
        """Remove stored credentials and tokens"""
        self._cred_store.delete_credential()

    def _submit_job(self, body: dict) -> Response:
        id_token = self.login()
        # send job request
        return self.session.post(
            f"{self.url}job",
            json.dumps(body),
            headers={"Authorization": id_token},
        )

    def _response_check(self, res: Response, description: str) -> None:
        """Consolidate as much error-checking of response"""
        # check if token has expired or is generally unauthorized
        if res.status_code == HTTPStatus.UNAUTHORIZED:
            jr = res.json()
            raise QuantinuumAPIError(
                f"Authorization failure attempting: {description}."
                f"\n\nServer Response: {jr}"
            )
        if res.status_code != HTTPStatus.OK:
            jr = res.json()
            raise QuantinuumAPIError(
                f"HTTP error attempting: {description}.\n\nServer Response: {jr}"
            )

    def retrieve_job_status(
        self,
        job_id: str,
        use_websocket: bool | None = None,
        request_raw_results: bool | None = None,
    ) -> dict | None:
        """
        Retrieves job status from device.

        :param job_id: unique id of job
        :param use_websocket: use websocket to minimize interaction

        :return: (dict) output from API

        """
        job_url = f"{self.url}job/{job_id}"
        # Using the login wrapper we will automatically try to refresh token
        id_token = self.login()
        params = {}
        if use_websocket or (use_websocket is None and self.use_websocket):
            params.update({"websocket": "true"})
        if request_raw_results:
            params.update({"results_format": "raw"})
        res = self.session.get(
            job_url, headers={"Authorization": id_token}, params=params
        )

        jr: dict | None = None
        # Check for invalid responses, and raise an exception if so
        self._response_check(res, "job status")
        # if we successfully got status return the decoded details
        if res.status_code == HTTPStatus.OK:
            jr = res.json()
        return jr

    def retrieve_job(
        self, job_id: str, use_websocket: bool | None = None
    ) -> dict | None:
        """
        Retrieves job from device.

        :param job_id: unique id of job
        :param use_websocket: use websocket to minimize interaction

        :return: (dict) output from API

        """
        jr = self.retrieve_job_status(job_id, use_websocket)
        if not jr:
            raise QuantinuumAPIError(f"Unable to retrieve job {job_id}")
        if "status" in jr and jr["status"] in self.JOB_DONE:
            return jr

        if "websocket" in jr:
            # wait for job completion using websocket
            try:
                loop = asyncio.get_event_loop()
                jr = loop.run_until_complete(self._wait_results(job_id))
            except RuntimeError:
                # no event loop in thread, call asyncio.run to use a new loop
                jr = asyncio.run(self._wait_results(job_id))

        else:
            # poll for job completion
            jr = self._poll_results(job_id)
        return jr

    def _poll_results(self, job_id: str) -> dict | None:
        jr = None
        start_time = time.time()
        while True:
            if self.timeout is not None and time.time() > (start_time + self.timeout):
                break
            self.login()
            try:
                jr = self.retrieve_job_status(job_id)

                # If we are failing to retrieve status of any kind, then fail out.
                if jr is None:
                    break
                if "status" in jr and jr["status"] in self.JOB_DONE:
                    return jr
                time.sleep(self.retry_timeout)
            except KeyboardInterrupt:
                raise RuntimeError("Keyboard Interrupted")  # noqa: B904
        return jr

    async def _wait_results(self, job_id: str) -> dict | None:
        start_time = time.time()
        while True:
            if self.timeout is not None and time.time() > (start_time + self.timeout):
                break
            self.login()
            jr = self.retrieve_job_status(job_id, True)
            if jr is None or ("status" in jr and jr["status"] in self.JOB_DONE):
                return jr
            task_token = jr["websocket"]["task_token"]
            execution_arn = jr["websocket"]["executionArn"]
            websocket_uri = self.url.replace("https://", "wss://ws.")
            async with connect(websocket_uri) as websocket:
                body = {
                    "action": "OpenConnection",
                    "task_token": task_token,
                    "executionArn": execution_arn,
                    "partial": False,
                }
                await websocket.send(json.dumps(body))
                while True:
                    try:
                        res = await asyncio.wait_for(
                            websocket.recv(), timeout=self.ws_timeout
                        )
                        jr = json.loads(res)
                        if not isinstance(jr, dict):
                            raise RuntimeError("Unable to decode response.")
                        if "status" in jr and jr["status"] in self.JOB_DONE:
                            return jr
                    except (TimeoutError, exceptions.ConnectionClosed):  # noqa: PERF203
                        try:
                            # Try to keep the connection alive...
                            pong = await websocket.ping()
                            await asyncio.wait_for(pong, timeout=10)
                            continue
                        except TimeoutError:
                            # If we are failing, wait a little while,
                            #  then start from the top
                            await asyncio.sleep(self.retry_timeout)
                            break
                    except KeyboardInterrupt:
                        raise RuntimeError("Keyboard Interrupted")  # noqa: B904

    def status(self, machine: str) -> str:
        """
        Check status of machine.

        :param machine: machine name

        :return: (str) status of machine

        """
        id_token = self.login()
        res = self.session.get(
            f"{self.url}machine/{machine}",
            headers={"Authorization": id_token},
        )
        self._response_check(res, "get machine status")
        jr = res.json()

        return str(jr["state"])

    def cancel(self, job_id: str) -> dict:
        """
        Cancels job.

        :param job_id: job ID to cancel

        :return: (dict) output from API

        """

        id_token = self.login()
        res = self.session.post(
            f"{self.url}job/{job_id}/cancel", headers={"Authorization": id_token}
        )
        self._response_check(res, "job cancel")
        jr = res.json()

        return jr  # type: ignore  # noqa: RET504

    def get_calendar(self, start_date: str, end_date: str) -> list[dict[str, str]]:
        """
        Retrieves calendar data using L4 API. All dates and times
        are in the UTC timezone.

        :param start_date: String formatted start date (YYYY-MM-DD)
        :param end_date: String formatted end date (YYYY-MM-DD)

        :return: (dict) output from API
        """
        id_token = self.login()

        base_url = self.url.replace("https://", "https://ui.").replace("v1", "beta")
        url = f"{base_url}reservation?mode=user&start={start_date}&end={end_date}"
        res = self.session.get(
            url,
            headers={"Authorization": id_token},
        )
        self._response_check(res, "get calendar events")
        jr: list[dict[str, str]] = res.json()
        return jr

    def get_machine_list(self) -> list[dict[str, Any]]:
        """Returns a given list of the available machines
        :return: list of machines
        """
        id_token = self.login()
        res = self.session.get(
            f"{self.url}machine/?config=true",
            headers={"Authorization": id_token},
        )
        self._response_check(res, "get machine list")
        jr = res.json()

        return cast("list[dict[str, Any]]", jr)


OFFLINE_MACHINE_LIST = [
    {
        "wasm": True,
        "batching": True,
        "supported_languages": ["OPENQASM 2.0", "QIR 1.0"],
        "benchmarks": {"qv": {"date": "2024-04-04", "value": 1048576.0}},
        "max_classical_register_width": 63,
        "gateset": ["RZZ", "Rxxyyzz", "Rz", "U1q", "ZZ"],
        "name": "H1-1",
        "syntax_checker": "H1-1SC",
        "n_gate_zones": "5",
        "noise_specs": {
            "date": "2025-05-02",
            "spam_error": {
                "p_meas_1_unc": 0.000176,
                "p_meas_0": 0.00122,
                "p_meas_1": 0.00343,
                "p_meas_0_unc": 0.000105,
            },
            "crosstalk_error": {
                "p_crosstalk_meas_unc": 1.7e-06,
                "p_crosstalk_meas": 4.1e-05,
            },
            "memory_error": {
                "memory_error_unc": 9.16e-05,
                "memory_error": 0.000222,
            },
            "1q_gate_error": {"p1": 1.8e-05, "p1_unc": 2.93e-06},
            "2q_gate_error": {"p2_unc": 6.23e-05, "p2": 0.000973},
        },
        "max_n_shots": 10000,
        "n_qubits": 20,
        "n_classical_registers": 4000,
        "system_type": "hardware",
        "connectivity": "all-to-all",
        "emulator": "H1-1E",
    },
    {
        "wasm": True,
        "batching": True,
        "supported_languages": ["OPENQASM 2.0", "QIR 1.0"],
        "benchmarks": {"qv": {"date": "2024-08-11", "value": 2097152.0}},
        "max_classical_register_width": 63,
        "gateset": ["RZZ", "Rxxyyzz", "Rz", "U1q", "ZZ"],
        "name": "H2-1",
        "syntax_checker": "H2-1SC",
        "n_gate_zones": "4",
        "noise_specs": {
            "date": "2025-04-30",
            "spam_error": {
                "p_meas_1_unc": 0.000124,
                "p_meas_0": 0.0006,
                "p_meas_1": 0.00139,
                "p_meas_0_unc": 8.16e-05,
            },
            "crosstalk_error": {
                "p_crosstalk_meas_unc": 8.98e-07,
                "p_crosstalk_meas": 6.65e-06,
            },
            "memory_error": {
                "memory_error_unc": 2.34e-05,
                "memory_error": 0.000203,
            },
            "1q_gate_error": {"p1": 1.89e-05, "p1_unc": 4.23e-06},
            "2q_gate_error": {"p2_unc": 8.08e-05, "p2": 0.00105},
        },
        "max_n_shots": 10000,
        "n_qubits": 56,
        "n_classical_registers": 4000,
        "system_type": "hardware",
        "connectivity": "all-to-all",
        "emulator": "H2-1E",
    },
    {
        "wasm": True,
        "batching": True,
        "supported_languages": ["OPENQASM 2.0", "QIR 1.0"],
        "benchmarks": {"qv": {"date": "2025-09-08", "value": 33554432.0}},
        "max_classical_register_width": 63,
        "gateset": ["RZZ", "Rxxyyzz", "Rz", "U1q", "ZZ"],
        "name": "H2-2",
        "syntax_checker": "H2-2SC",
        "n_gate_zones": "4",
        "noise_specs": {
            "date": "2025-08-28",
            "spam_error": {
                "p_meas_1_unc": 0.00011,
                "p_meas_0": 0.00067,
                "p_meas_1": 0.0012,
                "p_meas_0_unc": 8.7e-05,
            },
            "crosstalk_error": {
                "p_crosstalk_meas_unc": 5.3e-07,
                "p_crosstalk_meas": 2.2e-05,
            },
            "memory_error": {"memory_error_unc": 2e-05, "memory_error": 0.00012},
            "1q_gate_error": {"p1": 2.8e-05, "p1_unc": 3.6e-06},
            "2q_gate_error": {"p2_unc": 4.8e-05, "p2": 0.00083},
        },
        "max_n_shots": 10000,
        "n_qubits": 56,
        "n_classical_registers": 4000,
        "system_type": "hardware",
        "connectivity": "all-to-all",
        "emulator": "H2-2E",
    },
]


class QuantinuumAPIOffline:
    """
    Offline copy of the interface to the Quantinuum remote API.
    """

    def __init__(self, machine_list: list | None = None):
        """Initialize offline API client.

        Tries to allow all the operations of the :py:class:`QuantinuumAPI` without
        any interaction with the remote device.

        All jobs that are submitted to this offline API are stored
        and can be requested again later.

        :param machine_list: List of dictionaries each containing device information.
            The format of should match what a real backend would return.
            One short example:
            {
            "name": "H1-1",
            "n_qubits": 20,
            "gateset": ["RZZ", "Riswap", "TK2"],
            "n_shots": 10000,
            "batching": True,
            }
        """
        if machine_list is None:
            machine_list = OFFLINE_MACHINE_LIST
        self.provider = ""
        self.url = ""
        self.online = False
        self.machine_list = machine_list
        self._cred_store = None
        self.submitted: list = []

    def get_machine_list(self) -> list[dict[str, Any]]:
        """Returns a given list of the available machines
        :return: list of machines
        """

        return self.machine_list

    def full_login(self) -> None:
        """No login offline with the offline API"""

        return

    def login(self) -> str:
        """No login offline with the offline API, this function will always
        return an empty api token"""
        return ""

    def _submit_job(self, body: dict) -> None:
        """The function will take the submitted job and store it for later

        :param body: submitted job

        :return: None
        """
        self.submitted.append(body)

    def get_jobs(self) -> list | None:
        """The function will return all the jobs that have been submitted

        :return: List of all the submitted jobs
        """
        return self.submitted

    def _response_check(self, res: Response, description: str) -> None:
        """No _response_check offline"""

        jr = res.json()
        raise QuantinuumAPIError(
            f"Reponse can't be checked offline: {description}.\n\nServer Response: {jr}"
        )

    def retrieve_job_status(
        self, job_id: str, use_websocket: bool | None = None
    ) -> None:
        """No retrieve_job_status offline"""
        raise QuantinuumAPIError(
            f"Can't retrieve job status offline: job_id {job_id}."
            f"\n use_websocket {use_websocket}"
        )

    def retrieve_job(self, job_id: str, use_websocket: bool | None = None) -> None:
        """No retrieve_job_status offline"""
        raise QuantinuumAPIError(
            f"Can't retrieve job status offline: job_id {job_id}."
            f"\n use_websocket {use_websocket}"
        )

    def status(self, machine: str) -> str:
        """No retrieve_job_status offline"""

        return "unclear"

    def cancel(self, job_id: str) -> dict:
        """No cancel offline"""
        raise QuantinuumAPIError(f"Can't cancel job offline: job_id {job_id}.")
