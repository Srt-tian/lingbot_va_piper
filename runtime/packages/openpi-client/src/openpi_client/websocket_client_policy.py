"""远程 Policy 的同步 WebSocket 客户端。

对上层 Runtime 暴露与本地模型一致的 ``infer(obs)`` 接口；调用内部完成
MsgPack/NumPy 编解码、请求编号和耗时采集。
"""

import contextlib
import itertools
import logging
import time
from typing import Dict, Optional, Tuple

from typing_extensions import override
from websockets.exceptions import ConnectionClosed
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    _SESSION_BUSY_REASON = "Stateful EAPN policy already has an active session"
    _SESSION_BUSY_MAX_RETRIES = 5
    _SESSION_BUSY_RETRY_DELAY = 0.2

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._request_counter = itertools.count(1)
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def close(self) -> None:
        """Release this episode's connection and the server's stateful session."""
        if self._ws is not None:
            self._ws.close()
            self._ws = None

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        """持续重试连接，并读取服务端在握手后发送的模型 metadata。"""
        logging.info(f"Waiting for server at {self._uri}...")
        session_busy_retries = 0
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                try:
                    metadata = msgpack_numpy.unpackb(conn.recv())
                except BaseException:
                    # A failed handshake never becomes an owned policy client.
                    # Close it here, including when startup is interrupted.
                    with contextlib.suppress(Exception):
                        conn.close()
                    raise
                return conn, metadata
            except ConnectionClosed as exc:
                # Closing a socket can finish just before the server releases
                # its session lock. Allow a short grace period, but never evict
                # an active client or retry unrelated connection failures.
                if (
                    exc.rcvd is None
                    or exc.rcvd.code != 1013
                    or exc.rcvd.reason != self._SESSION_BUSY_REASON
                    or session_busy_retries >= self._SESSION_BUSY_MAX_RETRIES
                ):
                    raise
                session_busy_retries += 1
                logging.info(
                    "Waiting for previous EAPN session to close (%s/%s)...",
                    session_busy_retries,
                    self._SESSION_BUSY_MAX_RETRIES,
                )
                time.sleep(self._SESSION_BUSY_RETRY_DELAY)
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """发送一次阻塞式推理 RPC，返回服务端产生的 action chunk。"""
        ws = self._ws
        if ws is None:
            raise RuntimeError("Websocket policy client is closed")
        request_id = next(self._request_counter)
        roundtrip_start = time.monotonic()
        pack_start = time.monotonic()
        data = self._packer.pack(
            {
                "type": "infer",
                "request_id": request_id,
                "payload": obs,
            }
        )
        pack_ms = (time.monotonic() - pack_start) * 1000
        logging.debug("Sending websocket inference request_id=%s", request_id)
        send_start = time.monotonic()
        ws.send(data)
        send_ms = (time.monotonic() - send_start) * 1000
        wait_start = time.monotonic()
        # Runtime 把本方法放在独立推理线程中，因此这里阻塞不会阻塞控制循环。
        response = ws.recv()
        wait_response_ms = (time.monotonic() - wait_start) * 1000
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        unpack_start = time.monotonic()
        message = msgpack_numpy.unpackb(response)
        unpack_ms = (time.monotonic() - unpack_start) * 1000
        roundtrip_ms = (time.monotonic() - roundtrip_start) * 1000
        timing = {
            "transport": "websocket",
            "request_id": request_id,
            "pack_ms": pack_ms,
            "send_ms": send_ms,
            "wait_response_ms": wait_response_ms,
            "unpack_ms": unpack_ms,
            "roundtrip_ms": roundtrip_ms,
            "request_bytes": len(data),
            "response_bytes": len(response),
        }
        if isinstance(message, dict) and message.get("type") == "error":
            raise RuntimeError(
                f"Error in inference server for request_id={message.get('request_id', message.get('request_index'))}:\n"
                f"{message.get('traceback', '')}"
            )
        if isinstance(message, dict) and message.get("type") == "result" and "payload" in message:
            response_id = message.get("request_id", message.get("request_index"))
            if response_id != request_id:
                logging.warning(
                    "Websocket inference request id mismatch: sent=%s received=%s server_request_id=%s",
                    request_id,
                    response_id,
                    message.get("server_request_id"),
                )
            result = message["payload"]
            if not isinstance(result, dict):
                raise RuntimeError(f"Expected inference result dict, got {type(result).__name__}")
            result["client_timing"] = timing
            logging.debug(
                "Received websocket inference response request_id=%s server_request_id=%s "
                "pack=%.3fms send=%.3fms wait_response=%.3fms unpack=%.3fms roundtrip=%.3fms "
                "request_bytes=%s response_bytes=%s",
                response_id,
                message.get("server_request_id"),
                pack_ms,
                send_ms,
                wait_response_ms,
                unpack_ms,
                roundtrip_ms,
                len(data),
                len(response),
            )
            return result
        if not isinstance(message, dict):
            raise RuntimeError(f"Expected inference result dict, got {type(message).__name__}")
        message["client_timing"] = timing
        return message

    @override
    def reset(self) -> None:
        pass
