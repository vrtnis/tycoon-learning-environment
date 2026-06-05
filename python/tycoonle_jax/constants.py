from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

MAX_WIDTH = 54
MAX_HEIGHT = 38
MAX_NODES = 6
MAX_ROUTES = 12
MAX_CANDIDATES = 24
MAX_PATH = 128
MAX_TRANSIT = 24
CARGO_COUNT = 10
MODE_COUNT = 2
TERRAIN_COUNT = 4
FAMILY_COUNT = 5
SPLIT_COUNT = 3
METRIC_COUNT = 16
SCORE_BREAKDOWN_COUNT = 9
CANDIDATE_FEATURES = 12
NODE_FEATURES = 16
ROUTE_FEATURES = 18

SPLIT_TRAIN = 0
SPLIT_DEV = 1
SPLIT_TEST = 2

FAMILY_SINGLE_ROUTE = 0
FAMILY_LOW_CASH = 1
FAMILY_CHAIN = 2
FAMILY_MIXED_NETWORK = 3
FAMILY_TERRAIN_GAP = 4

TERRAIN_GRASS = 0
TERRAIN_ROUGH = 1
TERRAIN_WATER = 2
TERRAIN_TOWN = 3

NODE_PRODUCER = 0
NODE_PROCESSOR = 1
NODE_CONSUMER = 2
NODE_TOWN = 3
NODE_PORT = 4

MODE_ROAD = 0
MODE_RAIL = 1

ACTION_INVALID = 0
ACTION_BUILD_ROUTE = 1
ACTION_ADD_VEHICLE = 2
ACTION_WAIT = 3
ACTION_TAKE_LOAN = 4
ACTION_REPAY_LOAN = 5

CARGO_COAL = 0
CARGO_IRON_ORE = 1
CARGO_STEEL = 2
CARGO_WOOD = 3
CARGO_LUMBER = 4
CARGO_GRAIN = 5
CARGO_FOOD = 6
CARGO_GOODS = 7
CARGO_PASSENGERS = 8
CARGO_MAIL = 9

SPLIT_NAMES = ("train", "dev", "test")
FAMILY_NAMES = ("single_route", "low_cash", "chain", "mixed_network", "terrain_gap")
TERRAIN_NAMES = ("grass", "rough", "water", "town")
NODE_KIND_NAMES = ("producer", "processor", "consumer", "town", "port")
MODE_NAMES = ("road", "rail")
CARGO_NAMES = (
    "coal",
    "iron_ore",
    "steel",
    "wood",
    "lumber",
    "grain",
    "food",
    "goods",
    "passengers",
    "mail",
)
CARGO_LABELS = (
    "Coal",
    "Iron Ore",
    "Steel",
    "Wood",
    "Lumber",
    "Grain",
    "Food",
    "Goods",
    "Passengers",
    "Mail",
)

CARGO_VALUES = jnp.array([42.0, 46.0, 84.0, 34.0, 62.0, 30.0, 70.0, 92.0, 26.0, 44.0], dtype=jnp.float32)
MODE_TRACK_COST = jnp.array([780.0, 1850.0], dtype=jnp.float32)
MODE_STATION_COST = jnp.array([7000.0, 22000.0], dtype=jnp.float32)
MODE_VEHICLE_COST = jnp.array([16000.0, 44000.0], dtype=jnp.float32)
MODE_CAPACITY = jnp.array([26.0, 78.0], dtype=jnp.float32)
MODE_MAINTENANCE = jnp.array([520.0, 1180.0], dtype=jnp.float32)
MODE_SPEED = jnp.array([0.78, 1.2], dtype=jnp.float32)


@dataclass(frozen=True)
class ScenarioConfig:
    split: int = SPLIT_DEV
    family: int = FAMILY_CHAIN
    max_candidates: int = MAX_CANDIDATES


def split_id(value: str | int) -> int:
    if isinstance(value, int):
        if 0 <= value < SPLIT_COUNT:
            return value
        raise ValueError(f"unsupported split id {value}")
    try:
        return SPLIT_NAMES.index(value)
    except ValueError as exc:
        raise ValueError(f"unsupported split '{value}'") from exc


def family_id(value: str | int) -> int:
    if isinstance(value, int):
        if 0 <= value < FAMILY_COUNT:
            return value
        raise ValueError(f"unsupported family id {value}")
    try:
        return FAMILY_NAMES.index(value)
    except ValueError as exc:
        raise ValueError(f"unsupported family '{value}'") from exc
