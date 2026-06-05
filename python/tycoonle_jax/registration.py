from __future__ import annotations


def register() -> None:
    import jumanji

    try:
        jumanji.register(id="TycoonLE-v0", entry_point="tycoonle_jax.env:TycoonLE")
    except ValueError:
        pass
