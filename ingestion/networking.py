import math
import os
import platform
import re
import subprocess
import time

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal


POLL_INTERVAL_SECONDS = 1.0
CONSECUTIVE_FAILURE_LIMIT = 5
RSSI_MIN_DBM = -95
RSSI_MAX_DBM = -25
INFO_POLL_INTERVAL = 10.0


def _read_proc_net_wireless(interface: str) -> float | None:
    try:
        with open("/proc/net/wireless") as f:
            for line in f:
                if interface not in line:
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    level_str = parts[3].replace(".", "")
                    return float(level_str)
    except (FileNotFoundError, PermissionError, OSError, ValueError):
        return None
    return None


def _run_iw_dev_link(interface: str) -> float | None:
    try:
        result = subprocess.run(
            ["iw", "dev", interface, "link"],
            capture_output=True, text=True, timeout=5,
        )
        match = re.search(r"signal:\s*(-?\d+)\s*dBm", result.stdout)
        if match:
            return float(match.group(1))
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return None


def _run_iwconfig(interface: str) -> float | None:
    try:
        result = subprocess.run(
            ["iwconfig", interface],
            capture_output=True, text=True, timeout=5,
        )
        match = re.search(r"Signal level\s*=\s*(-?\d+)\s*dBm", result.stdout)
        if match:
            return float(match.group(1))
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return None


def _detect_linux_interfaces() -> list[str]:
    interfaces = []
    try:
        with open("/proc/net/wireless") as f:
            for i, line in enumerate(f):
                if i < 2:
                    continue
                iface = line.split(":")[0].strip()
                if iface:
                    interfaces.append(iface)
    except (FileNotFoundError, PermissionError, OSError):
        pass
    return interfaces


def _rssi_linux() -> float | None:
    interfaces = _detect_linux_interfaces()
    if not interfaces:
        return None
    iface = interfaces[0]

    rssi = _read_proc_net_wireless(iface)
    if rssi is not None:
        return rssi
    rssi = _run_iw_dev_link(iface)
    if rssi is not None:
        return rssi
    rssi = _run_iwconfig(iface)
    if rssi is not None:
        return rssi
    return None


def _rssi_windows() -> float | None:
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, text=True, timeout=5,
        )
        match = re.search(r"Signal\s*:\s*(\d+)%", result.stdout)
        if match:
            percent = float(match.group(1))
            return (percent / 2.0) - 100.0
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return None


def _get_ssid_from_iw(interface: str) -> str | None:
    try:
        result = subprocess.run(
            ["iw", "dev", interface, "link"],
            capture_output=True, text=True, timeout=5,
        )
        match = re.search(r"SSID:\s*(.+)", result.stdout)
        if match:
            return match.group(1).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return None


def _get_ssid_from_iwconfig(interface: str) -> str | None:
    try:
        result = subprocess.run(
            ["iwconfig", interface],
            capture_output=True, text=True, timeout=5,
        )
        match = re.search(r'ESSID:"(.+?)"', result.stdout)
        if match:
            return match.group(1)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return None


def _get_ssid_windows() -> str | None:
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "interfaces"],
            capture_output=True, text=True, timeout=5,
        )
        match = re.search(r"SSID\s*:\s*(.+)", result.stdout)
        if match:
            return match.group(1).strip()
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None
    return None


def _get_ssid_macos() -> str | None:
    path = (
        "/System/Library/PrivateFrameworks/Apple80211.framework/"
        "Versions/Current/Resources/airport"
    )
    if not os.path.exists(path):
        return None
    try:
        result = subprocess.run(
            [path, "-I"],
            capture_output=True, text=True, timeout=5,
        )
        match = re.search(r"\s+SSID:\s*(.+)", result.stdout)
        if match:
            return match.group(1).strip()
    except (subprocess.SubprocessError, OSError):
        return None
    return None


def _get_connected_device_names_linux(interface: str) -> list[str]:
    try:
        result = subprocess.run(
            ["iw", "dev", interface, "station", "dump"],
            capture_output=True, text=True, timeout=5,
        )
        macs = []
        for line in result.stdout.splitlines():
            if line.startswith("Station "):
                mac = line.split()[1]
                macs.append(mac)
        name_map: dict[str, str] = {}
        try:
            with open("/proc/net/arp") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 4 and parts[3] != "00:00:00:00:00:00":
                        name_map[parts[3].lower()] = parts[0]
        except OSError:
            pass
        names = []
        for mac in macs:
            ip = name_map.get(mac.lower())
            if ip:
                try:
                    import socket
                    hostname, _, _ = socket.gethostbyaddr(ip)
                    names.append(hostname.split(".")[0])
                except Exception:
                    names.append(mac[-8:])
            else:
                names.append(mac[-8:])
        return names
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []


def _get_connected_device_names_windows() -> list[str]:
    try:
        result = subprocess.run(
            ["netsh", "wlan", "show", "hostednetwork"],
            capture_output=True, text=True, timeout=5,
        )
        match = re.search(r"Number of clients\s*:\s*(\d+)", result.stdout)
        count = int(match.group(1)) if match else 0
        return [f"Device-{i + 1}" for i in range(count)]
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return []


def _rssi_macos() -> float | None:
    path = (
        "/System/Library/PrivateFrameworks/Apple80211.framework/"
        "Versions/Current/Resources/airport"
    )
    if not os.path.exists(path):
        return None
    try:
        result = subprocess.run(
            [path, "-I"],
            capture_output=True, text=True, timeout=5,
        )
        match = re.search(r"agrCtlRSSI:\s*(-?\d+)", result.stdout)
        if match:
            return float(match.group(1))
    except (subprocess.SubprocessError, OSError):
        return None
    return None


class RSSIProvider:
    def __init__(self):
        self.system = platform.system()

    def read(self) -> float | None:
        if self.system == "Linux":
            return _rssi_linux()
        elif self.system == "Windows":
            return _rssi_windows()
        elif self.system == "Darwin":
            return _rssi_macos()
        return None

    @staticmethod
    def source_description() -> str:
        system = platform.system()
        if system == "Linux":
            interfaces = _detect_linux_interfaces()
            if interfaces:
                iface = interfaces[0]
                if _read_proc_net_wireless(iface) is not None:
                    return f"Linux /proc/net/wireless ({iface})"
                if _run_iw_dev_link(iface) is not None:
                    return f"Linux iw dev ({iface})"
                if _run_iwconfig(iface) is not None:
                    return f"Linux iwconfig ({iface})"
            return "Linux (no interface found)"
        elif system == "Windows":
            return "Windows netsh wlan"
        elif system == "Darwin":
            return "macOS airport"
        return f"Unknown OS ({system})"

    def get_ssid(self) -> str | None:
        if self.system == "Linux":
            interfaces = _detect_linux_interfaces()
            if interfaces:
                iface = interfaces[0]
                ssid = _get_ssid_from_iw(iface)
                if ssid is not None:
                    return ssid
                ssid = _get_ssid_from_iwconfig(iface)
                if ssid is not None:
                    return ssid
        elif self.system == "Windows":
            return _get_ssid_windows()
        elif self.system == "Darwin":
            return _get_ssid_macos()
        return None

    def get_connected_devices(self) -> list[str]:
        if self.system == "Linux":
            interfaces = _detect_linux_interfaces()
            if interfaces:
                return _get_connected_device_names_linux(interfaces[0])
        elif self.system == "Windows":
            return _get_connected_device_names_windows()
        return []


class SimulatedRSSIProvider:
    def __init__(self):
        self._t = 0.0
        self._walk = 0.0
        self._last_value = -55.0
        self._rng = np.random.default_rng()

    def read(self) -> float:
        self._t += POLL_INTERVAL_SECONDS
        drift = 15.0 * math.sin(2.0 * math.pi * self._t / 120.0)
        self._walk += self._rng.normal(0, 0.5)
        self._walk = max(-10.0, min(10.0, self._walk))
        noise = self._rng.normal(0, 1.0)
        value = -55.0 + drift + self._walk + noise
        value = max(RSSI_MIN_DBM, min(RSSI_MAX_DBM, value))
        self._last_value = value
        return value

    def get_ssid(self) -> str:
        return "SIMULATED_NETWORK"

    def get_connected_devices(self) -> list[str]:
        count = int(abs(np.random.default_rng().normal(3, 1.5))) + 1
        return [f"Device-{i + 1}" for i in range(count)]

    @staticmethod
    def source_description() -> str:
        return "Simulated (no Wi-Fi source available)"


class RSSIWorker(QThread):
    rssi_updated = pyqtSignal(float, str)
    source_changed = pyqtSignal(str)
    ssid_updated = pyqtSignal(str)
    devices_updated = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._interval = POLL_INTERVAL_SECONDS
        self._running = False
        self._real_provider = RSSIProvider()
        self._sim_provider = SimulatedRSSIProvider()
        self._current_provider = self._real_provider
        self._consecutive_failures = 0
        self._failure_limit = CONSECUTIVE_FAILURE_LIMIT
        self._last_good_value: float | None = None
        self._is_simulated = False
        self._info_counter = 0.0

    @property
    def is_simulated(self) -> bool:
        return self._is_simulated

    @property
    def last_good_value(self) -> float | None:
        return self._last_good_value

    def run(self):
        self._running = True
        while self._running:
            try:
                rssi = self._current_provider.read()
            except Exception:
                rssi = None

            if rssi is not None:
                self._consecutive_failures = 0
                self._last_good_value = rssi
                if self._is_simulated:
                    self._is_simulated = False
                    self._current_provider = self._real_provider
                    self.source_changed.emit(
                        RSSIProvider.source_description()
                    )
                self.rssi_updated.emit(
                    rssi, "real" if not self._is_simulated else "simulated"
                )
            else:
                self._consecutive_failures += 1
                if self._consecutive_failures >= self._failure_limit:
                    if not self._is_simulated:
                        self._is_simulated = True
                        self._current_provider = self._sim_provider
                        self._consecutive_failures = 0
                        sim_value = self._sim_provider.read()
                        self._last_good_value = sim_value
                        self.source_changed.emit(
                            "SIMULATED — real Wi-Fi source unavailable"
                        )
                        self.rssi_updated.emit(sim_value, "simulated")
                else:
                    if self._last_good_value is not None:
                        self.rssi_updated.emit(
                            self._last_good_value, "real (held)"
                        )

            self._info_counter += self._interval
            if self._info_counter >= INFO_POLL_INTERVAL:
                self._info_counter = 0.0
                try:
                    ssid = self._current_provider.get_ssid()
                    if ssid is not None:
                        self.ssid_updated.emit(ssid)
                except Exception:
                    pass
                try:
                    devices = self._current_provider.get_connected_devices()
                    self.devices_updated.emit(devices)
                except Exception:
                    pass

            for _ in range(int(self._interval / 0.1)):
                if not self._running:
                    return
                time.sleep(0.1)

    def stop(self):
        self._running = False
