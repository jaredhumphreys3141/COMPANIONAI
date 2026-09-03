from companionai import hardware


def test_detect_returns_a_known_device():
    info = hardware.detect()
    assert info.device in (hardware.PC, hardware.RPI5, hardware.JETSON)
    assert info.cpu_count >= 1


def test_jetson_profile_offloads_to_the_gpu():
    info = hardware.HardwareInfo(
        device=hardware.JETSON, label="Jetson", machine="aarch64", system="Linux",
        cpu_count=6, total_ram_gb=8.0, cuda=True,
    )
    profile = hardware.profile_for(info)
    assert profile.llm_gpu_layers == -1
    assert profile.torch_device == "cuda"
    assert profile.asr_compute_type == "float16"


def test_pi_profile_stays_on_cpu_and_picks_small_models():
    info = hardware.HardwareInfo(
        device=hardware.RPI5, label="Raspberry Pi 5", machine="aarch64", system="Linux",
        cpu_count=4, total_ram_gb=8.0, cuda=False,
    )
    profile = hardware.profile_for(info)
    assert profile.llm_gpu_layers == 0
    assert profile.asr_model == "tiny.en"
    assert profile.image_size <= 512


def test_summary_is_markdown():
    assert "Platform:" in hardware.summary()
