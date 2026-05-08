"""Hardware detection and device assignment for DK-2500."""

import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass
class DeviceConfig:
    vit_device: str = "GPU"
    llm_prefill_device: str = "GPU"
    llm_generate_device: str = "NPU"
    fallback_device: str = "GPU"


def detect_devices() -> list[str]:
    """Detect available OpenVINO compute devices."""
    try:
        import openvino as ov
        core = ov.Core()
        devices = core.available_devices
        log.info(f"Available devices: {devices}")
        return devices
    except ImportError:
        log.warning("OpenVINO not installed, returning CPU only")
        return ["CPU"]


def get_device_config(target: str = "dk2500") -> DeviceConfig:
    """Get device assignment config based on target hardware."""
    available = detect_devices()

    if target == "dk2500":
        has_npu = "NPU" in available
        has_gpu = "GPU" in available
        return DeviceConfig(
            vit_device="GPU" if has_gpu else "CPU",
            llm_prefill_device="GPU" if has_gpu else "CPU",
            llm_generate_device="NPU" if has_npu else ("GPU" if has_gpu else "CPU"),
            fallback_device="GPU" if has_gpu else "CPU",
        )

    # Default: CPU-only
    return DeviceConfig(
        vit_device="CPU",
        llm_prefill_device="CPU",
        llm_generate_device="CPU",
        fallback_device="CPU",
    )
