"""Pure-data defaults for the portable managed runtime."""

LOCAL_RUNTIME_DEFAULTS = {
    # Off = detection-only (Hermes still finds an external llama-server you run).
    "enabled": False,
    # Engine versions and every dependent library are pinned by pm/lock.json.
    # auto = CUDA on NVIDIA, Metal on macOS, Vulkan on other GPUs, else CPU. Explicit:
    # cuda|metal|vulkan|hip|cpu.
    "backend": "auto",
    "request_scheduler": {
        "enabled": True,
        "max_waiters": 64,
        "queue_timeout_seconds": 3600,
        "foreground_advantage_seconds": 30,
    },
    "adaptive_resources": {
        "prompt_cache_max_mib": 8192,
        "prompt_cache_room_fraction": 1.0,
        "pause_on_memory_pressure": True,
        "physical_reserve_fraction": 0.05,
        "growth_stability_seconds": 30,
        "refit_cooldown_seconds": 60,
    },
    "models_max": 4,  # Router process: how many models may be resident at once.
    "port": 0,  # Port for the managed server. 0 = pick a free port at spawn.
    # Extra ports detection probes for an external llama-server (besides 8080).
    "detect_ports": [],
}
