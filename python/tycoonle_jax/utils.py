from __future__ import annotations

import jax
import jax.numpy as jnp

from tycoonle_jax.constants import MAX_HEIGHT, MAX_PATH, MAX_WIDTH


def hash01(seed: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray, salt: int | jnp.ndarray) -> jnp.ndarray:
    """Small deterministic integer hash returning floats in [0, 1)."""

    h = (seed.astype(jnp.uint32) + jnp.uint32(0x9E3779B9)) * jnp.uint32(374761393)
    h = h ^ ((x.astype(jnp.uint32) + jnp.asarray(salt, dtype=jnp.uint32) * jnp.uint32(17)) * jnp.uint32(668265263))
    h = h ^ ((y.astype(jnp.uint32) + jnp.asarray(salt, dtype=jnp.uint32) * jnp.uint32(31)) * jnp.uint32(2246822519))
    h = h ^ jnp.right_shift(h, jnp.uint32(13))
    h = h * jnp.uint32(1274126177)
    h = h ^ jnp.right_shift(h, jnp.uint32(16))
    return h.astype(jnp.float32) / jnp.float32(4294967296.0)


def split_seed(split: int, seed: jnp.ndarray) -> jnp.ndarray:
    offset = jnp.array([10_000, 20_000, 30_000], dtype=jnp.int32)[split]
    return offset + seed.astype(jnp.int32) * jnp.int32(9973)


def round_to(value: jnp.ndarray, digits: int) -> jnp.ndarray:
    factor = jnp.asarray(10**digits, dtype=jnp.float32)
    return jnp.round((value + 1e-6) * factor) / factor


def direct_template_path(source_x: jnp.ndarray, source_y: jnp.ndarray, dest_x: jnp.ndarray, dest_y: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return a fixed-width Manhattan route: horizontal first, then vertical."""

    idx = jnp.arange(MAX_PATH, dtype=jnp.int32)
    dx = dest_x - source_x
    dy = dest_y - source_y
    abs_dx = jnp.abs(dx)
    abs_dy = jnp.abs(dy)
    length = jnp.minimum(MAX_PATH, abs_dx + abs_dy + 1)
    x_step = jnp.sign(dx).astype(jnp.int32)
    y_step = jnp.sign(dy).astype(jnp.int32)

    horizontal = idx <= abs_dx
    x = jnp.where(horizontal, source_x + idx * x_step, dest_x)
    y = jnp.where(horizontal, source_y, source_y + (idx - abs_dx) * y_step)
    x = jnp.clip(x, 0, MAX_WIDTH - 1)
    y = jnp.clip(y, 0, MAX_HEIGHT - 1)
    valid = idx < length
    path = jnp.stack([jnp.where(valid, x, 0), jnp.where(valid, y, 0)], axis=-1).astype(jnp.int32)
    return path, length.astype(jnp.int32)


def path_valid_mask(length: jnp.ndarray) -> jnp.ndarray:
    return jnp.arange(MAX_PATH, dtype=jnp.int32) < length


def gather_path_terrain(terrain: jnp.ndarray, path: jnp.ndarray) -> jnp.ndarray:
    x = path[:, 0]
    y = path[:, 1]
    return terrain[y, x]


def build_occupancy(
    route_mask: jnp.ndarray,
    route_path: jnp.ndarray,
    route_path_length: jnp.ndarray,
    route_mode: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    flat_path = route_path.reshape((-1, 2))
    route_ids = jnp.repeat(jnp.arange(route_mask.shape[0], dtype=jnp.int32), MAX_PATH)
    tile_ids = jnp.tile(jnp.arange(MAX_PATH, dtype=jnp.int32), route_mask.shape[0])
    valid = route_mask[route_ids] & (tile_ids < route_path_length[route_ids])
    x = flat_path[:, 0]
    y = flat_path[:, 1]
    road = valid & (route_mode[route_ids] == 0)
    rail = valid & (route_mode[route_ids] == 1)
    road_grid = jnp.zeros((MAX_HEIGHT, MAX_WIDTH), dtype=jnp.float32).at[y, x].add(road.astype(jnp.float32))
    rail_grid = jnp.zeros((MAX_HEIGHT, MAX_WIDTH), dtype=jnp.float32).at[y, x].add(rail.astype(jnp.float32))
    return road_grid, rail_grid


def first_true(mask: jnp.ndarray) -> jnp.ndarray:
    return jnp.argmax(mask.astype(jnp.int32)).astype(jnp.int32)


def one_hot(index: jnp.ndarray, size: int) -> jnp.ndarray:
    return jax.nn.one_hot(index, size, dtype=jnp.float32)
