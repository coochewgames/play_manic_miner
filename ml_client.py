#!/usr/bin/env python3
"""Low-level client for the Fuse ML UNIX socket protocol."""

from __future__ import annotations

import socket
from typing import Dict


class FuseMLClient:
    """Simple line-based client for the Fuse ML bridge."""

    def __init__(self, socket_path: str = "/tmp/fuse-ml.sock", timeout_s: float = 30.0):
        self.socket_path = socket_path
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._socket.settimeout(timeout_s)
        self._buffer = b""
        self._socket.connect(socket_path)
        banner = self._readline()
        if not banner.startswith("OK READY"):
            raise RuntimeError(f"Unexpected Fuse banner: {banner!r}")

    def close(self) -> None:
        self._socket.close()

    def _readline(self) -> str:
        while b"\n" not in self._buffer:
            chunk = self._socket.recv(65536)
            if not chunk:
                raise RuntimeError("Fuse ML socket closed")
            self._buffer += chunk
        line, self._buffer = self._buffer.split(b"\n", 1)
        return line.decode("utf-8", errors="replace").strip()

    def cmd(self, text: str) -> str:
        self._socket.sendall((text + "\n").encode("utf-8"))
        response = self._readline()
        if response.startswith("ERR "):
            raise RuntimeError(f"Fuse command failed for {text!r}: {response}")
        return response

    def ping(self) -> str:
        return self.cmd("PING")

    def reset(self) -> str:
        return self.cmd("RESET")

    def mode_headless(self) -> str:
        return self.cmd("MODE HEADLESS")

    def mode_visual(self, pace_ms: int = 0) -> str:
        return self.cmd(f"MODE VISUAL {pace_ms}")

    def game(self) -> str:
        return self.cmd("GAME")

    def get_info(self) -> Dict[str, int]:
        parts = self.cmd("GETINFO").split()
        # INFO <frame_count> <tstates> <width> <height>
        return {
            "frame_count": int(parts[1]),
            "tstates": int(parts[2]),
            "width": int(parts[3]),
            "height": int(parts[4]),
        }

    def read_bytes(self, address: int, length: int) -> bytes:
        parts = self.cmd(f"READ {address} {length}").split(maxsplit=1)
        # DATA <hex>
        if len(parts) != 2 or parts[0] != "DATA":
            raise RuntimeError(f"Unexpected READ response: {' '.join(parts)!r}")
        return bytes.fromhex(parts[1])

    def episode_step(self, action: int, frames: int, auto_reset: int = 0) -> Dict[str, int]:
        parts = self.cmd(f"EPISODE_STEP {action} {frames} {auto_reset}").split()
        # EPISODE <frame_count> <tstates> <width> <height> <reward> <done> <reset>
        if len(parts) != 8 or parts[0] != "EPISODE":
            raise RuntimeError(f"Unexpected EPISODE response: {' '.join(parts)!r}")
        return {
            "frame_count": int(parts[1]),
            "tstates": int(parts[2]),
            "width": int(parts[3]),
            "height": int(parts[4]),
            "reward": int(parts[5]),
            "done": int(parts[6]),
            "reset": int(parts[7]),
        }

    def episode_step_keys(self, key_chord: str, frames: int, auto_reset: int = 0) -> Dict[str, int]:
        parts = self.cmd(f"EPISODE_STEP_KEYS {key_chord} {frames} {auto_reset}").split()
        # EPISODE <frame_count> <tstates> <width> <height> <reward> <done> <reset>
        if len(parts) != 8 or parts[0] != "EPISODE":
            raise RuntimeError(f"Unexpected EPISODE response: {' '.join(parts)!r}")
        return {
            "frame_count": int(parts[1]),
            "tstates": int(parts[2]),
            "width": int(parts[3]),
            "height": int(parts[4]),
            "reward": int(parts[5]),
            "done": int(parts[6]),
            "reset": int(parts[7]),
        }
