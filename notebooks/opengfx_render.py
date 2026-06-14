from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from tycoonle_jax.replay import decode_observation

TILE_W = 64
TILE_H = 31
TILE_Y_STEP = 16


@dataclass(frozen=True)
class SpriteDef:
    sheet: str
    sx: int
    sy: int
    sw: int
    sh: int
    x_rel: int
    y_rel: int


@dataclass(frozen=True)
class Camera:
    origin_x: int
    origin_y: int


class SpriteStore:
    def __init__(self, repo_root: Path) -> None:
        self.root = repo_root / "public" / "assets" / "opengfx"
        self.sheets = {
            "grass": self._load_sheet("grass-temperate.png"),
            "bare": self._load_sheet("bare-temperate.png"),
            "rough": self._load_sheet("terrain-rough-temperate.png"),
            "rocks": self._load_sheet("terrain-rocks-temperate.png"),
            "water": self._load_sheet("terrain-river-water.png"),
            "field": self._load_sheet("terrain-field-48.png"),
            "houses_temperate": self._load_sheet("houses-temperate.png"),
            "houses_buildings": self._load_sheet("houses-buildings.png"),
            "houses_parks": self._load_sheet("houses-parks.png"),
            "coal_mine": self._load_sheet("industry-coalmine-base.png"),
            "factory": self._load_sheet("industry-factory.png"),
            "farm": self._load_sheet("industry-farm-temperate.png"),
            "sawmill": self._load_sheet("industry-sawmill.png"),
            "steel_mill": self._load_sheet("industry-steelmill.png"),
            "tree_leaf_01": self._load_sheet("trees-leaf-01.png"),
            "tree_leaf_02": self._load_sheet("trees-leaf-02.png"),
            "tree_conifer_03": self._load_sheet("trees-conifer-03.png"),
            "tree_leaf_05": self._load_sheet("trees-leaf-05.png"),
            "tree_leaf_13": self._load_sheet("trees-leaf-13.png"),
        }
        self.crops: dict[SpriteDef, Image.Image] = {}

    def crop(self, sprite: SpriteDef) -> Image.Image:
        cached = self.crops.get(sprite)
        if cached is not None:
            return cached
        image = self.sheets[sprite.sheet].crop((sprite.sx, sprite.sy, sprite.sx + sprite.sw, sprite.sy + sprite.sh))
        self.crops[sprite] = image
        return image

    def _load_sheet(self, name: str) -> Image.Image:
        image = Image.open(self.root / name).convert("RGBA")
        pixels = np.array(image)
        blue_key = (pixels[:, :, 2] > 200) & (pixels[:, :, 0] < 18) & (pixels[:, :, 1] < 18)
        pixels[blue_key, 3] = 0
        return Image.fromarray(pixels, mode="RGBA")


GRASS_TILES = [SpriteDef("grass", 1, 1, 64, 31, -31, 0)]
BARE_TILES = [SpriteDef("bare", 1, 1, 64, 31, -31, 0)]
ROUGH_TILES = [SpriteDef("rough", 1, 1, 64, 31, -31, 0)]
ROCK_TILES = [SpriteDef("rocks", 1, 1, 64, 31, -31, 0)]
FIELD_TILES = [SpriteDef("field", 1, 1, 64, 31, -31, 0)]
WATER_TILES = [SpriteDef("water", 1, 1, 64, 31, -31, 0)]

HOUSE_SPRITES = [
    SpriteDef("houses_temperate", 642, 8, 64, 67, -31, -39),
    SpriteDef("houses_temperate", 498, 104, 52, 61, -25, -33),
    SpriteDef("houses_temperate", 562, 104, 52, 61, -25, -33),
    SpriteDef("houses_temperate", 706, 104, 54, 115, -25, -88),
    SpriteDef("houses_temperate", 258, 360, 62, 48, -30, -20),
    SpriteDef("houses_temperate", 418, 360, 62, 83, -30, -55),
    SpriteDef("houses_buildings", 178, 2856, 36, 28, -12, -9),
    SpriteDef("houses_buildings", 578, 2696, 32, 27, -17, -11),
    SpriteDef("houses_buildings", 674, 2696, 32, 27, -17, -11),
]

PARK_SPRITES = [
    SpriteDef("houses_parks", 11, 0, 64, 80, -31, -49),
    SpriteDef("houses_parks", 91, 10, 64, 70, -31, -39),
]

TREE_SPRITES = [
    SpriteDef("tree_leaf_01", 150, 0, 45, 80, -24, -73),
    SpriteDef("tree_leaf_02", 200, 0, 45, 80, -24, -73),
    SpriteDef("tree_conifer_03", 250, 0, 45, 80, -24, -73),
    SpriteDef("tree_leaf_05", 150, 0, 45, 80, -24, -73),
    SpriteDef("tree_leaf_13", 200, 0, 45, 80, -24, -73),
]

INDUSTRY = {
    "coal_ground": SpriteDef("coal_mine", 690, 8, 64, 31, -31, 0),
    "coal_shaft": SpriteDef("coal_mine", 114, 8, 36, 50, -16, -33),
    "coal_loader": SpriteDef("coal_mine", 162, 8, 58, 50, -16, -33),
    "coal_tipple": SpriteDef("coal_mine", 418, 8, 48, 41, -29, -18),
    "farm_ground": SpriteDef("farm", 170, 10, 64, 31, -31, 0),
    "farm_house": SpriteDef("farm", 10, 60, 32, 64, -17, -28),
    "farm_barn": SpriteDef("farm", 170, 60, 57, 29, -25, -5),
    "farm_silo": SpriteDef("farm", 330, 60, 45, 48, -6, -34),
    "factory_ground": SpriteDef("factory", 402, 10, 64, 31, -31, 0),
    "factory_hall": SpriteDef("factory", 104, 2, 57, 62, -28, -37),
    "factory_stack": SpriteDef("factory", 718, 90, 64, 91, -31, -60),
    "sawmill_shed": SpriteDef("sawmill", 66, 80, 51, 44, -24, -18),
    "sawmill_logs": SpriteDef("sawmill", 354, 80, 54, 38, -27, -12),
    "steel_ground": SpriteDef("steel_mill", 2, 10, 64, 31, -31, 0),
    "steel_mill": SpriteDef("steel_mill", 642, 10, 64, 64, -31, -33),
    "steel_yard": SpriteDef("steel_mill", 82, 138, 64, 39, -31, -8),
}


def render_opengfx_map(state: Any, repo_root: Path | str) -> Image.Image:
    observation = decode_observation(state)
    world = observation["world"]
    width = max(960, int((world["width"] + world["height"]) * (TILE_W / 2) + 360))
    height = max(640, int((world["width"] + world["height"]) * TILE_Y_STEP + 360))
    camera = Camera(origin_x=180 + (world["height"] - 1) * (TILE_W // 2), origin_y=120)
    assets = _sprite_store(str(Path(repo_root).resolve()))
    canvas = Image.new("RGBA", (width, height), (31, 77, 29, 255))
    draw = ImageDraw.Draw(canvas)

    _draw_terrain(canvas, observation, assets, camera)
    _draw_routes(draw, observation, camera)
    _draw_objects(canvas, observation, assets, camera)
    _draw_labels(draw, observation, camera)
    return canvas.convert("RGB")


@lru_cache(maxsize=4)
def _sprite_store(repo_root: str) -> SpriteStore:
    return SpriteStore(Path(repo_root))


def _draw_terrain(canvas: Image.Image, observation: dict[str, Any], assets: SpriteStore, camera: Camera) -> None:
    world = observation["world"]
    terrain = world["terrain"]
    seed = world["seed"]
    for y in range(world["height"]):
        for x in range(world["width"]):
            point = _world_to_iso(x, y, camera)
            tile = _visual_terrain(terrain, x, y)
            _draw_sprite(canvas, assets, _terrain_sprite(tile, seed, x, y), point[0], point[1])


def _draw_routes(draw: ImageDraw.ImageDraw, observation: dict[str, Any], camera: Camera) -> None:
    for route in observation["routes"]:
        path = route.get("path") or _route_fallback_path(route, observation["nodes"])
        if len(path) < 2:
            continue
        points = [_tile_center(tile["x"], tile["y"], camera) for tile in path]
        if route["mode"] == "rail":
            draw.line(points, fill=(38, 30, 24), width=10, joint="curve")
            draw.line(points, fill=(18, 18, 18), width=4, joint="curve")
            draw.line(points, fill=(145, 145, 145), width=2, joint="curve")
        else:
            draw.line(points, fill=(34, 34, 31), width=12, joint="curve")
            draw.line(points, fill=(98, 103, 102), width=8, joint="curve")
            draw.line(points, fill=(157, 160, 160), width=2, joint="curve")


def _draw_objects(canvas: Image.Image, observation: dict[str, Any], assets: SpriteStore, camera: Camera) -> None:
    drawables = []
    world = observation["world"]
    nodes = observation["nodes"]
    for y in range(world["height"]):
        for x in range(world["width"]):
            tile = world["terrain"][y][x]
            if tile in {"water", "town"} or _is_near_node(nodes, x, y, 3):
                continue
            density = 0.12 if tile == "rough" else 0.17
            if _hash01(world["seed"], x, y, 31) > density:
                continue
            sprite = _pick(TREE_SPRITES, world["seed"], x, y, 32)
            point = _world_to_iso(x, y, camera)
            drawables.append((x + y + 0.2, sprite, point[0] + int((_hash01(world["seed"], x, y, 33) - 0.5) * 22), point[1] + 15))
    for node in nodes:
        if node["kind"] == "town":
            _add_town_drawables(drawables, node, world["seed"], camera)
        else:
            _add_industry_drawables(drawables, node, camera)
    for _, sprite, x, y in sorted(drawables, key=lambda item: item[0]):
        _draw_sprite(canvas, assets, sprite, x, y)


def _add_town_drawables(drawables: list[tuple[float, SpriteDef, int, int]], node: dict[str, Any], seed: int, camera: Camera) -> None:
    radius = 3 if (node.get("population") or 0) > 1800 else 2
    sprites = HOUSE_SPRITES + PARK_SPRITES
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if abs(dx) + abs(dy) > radius + 1:
                continue
            x = node["x"] + dx
            y = node["y"] + dy
            sprite = _pick(sprites, seed, x, y, 41)
            point = _world_to_iso(x, y, camera)
            drawables.append((x + y + 0.55, sprite, point[0], point[1]))


def _add_industry_drawables(drawables: list[tuple[float, SpriteDef, int, int]], node: dict[str, Any], camera: Camera) -> None:
    produced = " ".join((node.get("produces") or {}).keys())
    accepted = " ".join((node.get("accepts") or {}).keys())
    cargo_text = f"{produced} {accepted} {node['name']}".lower()
    x = node["x"]
    y = node["y"]
    if "grain" in cargo_text or "food" in cargo_text or "farm" in cargo_text:
        _add_sprite(drawables, camera, x, y, INDUSTRY["farm_ground"], 0.1)
        _add_sprite(drawables, camera, x, y, INDUSTRY["farm_house"], 0.5)
        _add_sprite(drawables, camera, x + 1, y, INDUSTRY["farm_barn"], 0.5)
        _add_sprite(drawables, camera, x + 1, y + 1, INDUSTRY["farm_silo"], 0.5)
    elif "wood" in cargo_text or "lumber" in cargo_text:
        _add_sprite(drawables, camera, x, y, INDUSTRY["sawmill_shed"], 0.5)
        _add_sprite(drawables, camera, x + 1, y, INDUSTRY["sawmill_logs"], 0.5)
    elif "steel" in cargo_text or "iron" in cargo_text:
        _add_sprite(drawables, camera, x, y, INDUSTRY["steel_ground"], 0.1)
        _add_sprite(drawables, camera, x, y, INDUSTRY["steel_mill"], 0.5)
        _add_sprite(drawables, camera, x + 1, y, INDUSTRY["steel_yard"], 0.5)
    elif node["kind"] == "consumer" or "goods" in cargo_text:
        _add_sprite(drawables, camera, x, y, INDUSTRY["factory_ground"], 0.1)
        _add_sprite(drawables, camera, x, y, INDUSTRY["factory_hall"], 0.5)
        _add_sprite(drawables, camera, x + 1, y, INDUSTRY["factory_stack"], 0.6)
    else:
        _add_sprite(drawables, camera, x, y, INDUSTRY["coal_ground"], 0.1)
        _add_sprite(drawables, camera, x, y, INDUSTRY["coal_shaft"], 0.5)
        _add_sprite(drawables, camera, x + 1, y, INDUSTRY["coal_loader"], 0.5)
        _add_sprite(drawables, camera, x, y + 1, INDUSTRY["coal_tipple"], 0.5)


def _add_sprite(drawables: list[tuple[float, SpriteDef, int, int]], camera: Camera, x: int, y: int, sprite: SpriteDef, depth: float) -> None:
    point = _world_to_iso(x, y, camera)
    drawables.append((x + y + depth, sprite, point[0], point[1]))


def _draw_labels(draw: ImageDraw.ImageDraw, observation: dict[str, Any], camera: Camera) -> None:
    for node in observation["nodes"]:
        x, y = _world_to_iso(node["x"], node["y"], camera)
        label_y = y - (44 if node["kind"] == "town" else 38)
        label = node["name"]
        for ox, oy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            draw.text((x + ox, label_y + oy), label, anchor="mm", fill=(0, 0, 0))
        draw.text((x, label_y), label, anchor="mm", fill=(255, 255, 255) if node["kind"] == "town" else (244, 227, 166))


def _draw_sprite(canvas: Image.Image, assets: SpriteStore, sprite: SpriteDef, x: int, y: int) -> None:
    image = assets.crop(sprite)
    canvas.alpha_composite(image, (round(x + sprite.x_rel), round(y + sprite.y_rel)))


def _terrain_sprite(terrain: str, seed: int, x: int, y: int) -> SpriteDef:
    if terrain == "water":
        return _pick(WATER_TILES, seed, x, y, 3)
    if terrain == "rough":
        return _pick(ROCK_TILES, seed, x, y, 5) if _hash01(seed, x, y, 4) > 0.68 else _pick(ROUGH_TILES, seed, x, y, 6)
    if _hash01(seed, x, y, 7) < 0.035:
        return _pick(FIELD_TILES, seed, x, y, 8)
    if _hash01(seed, x, y, 9) < 0.07:
        return _pick(BARE_TILES, seed, x, y, 10)
    return _pick(GRASS_TILES, seed, x, y, 11)


def _visual_terrain(terrain: list[list[str]], x: int, y: int) -> str:
    tile = terrain[y][x]
    if tile != "water":
        return tile
    neighbors = sum(1 for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)] if 0 <= y + dy < len(terrain) and 0 <= x + dx < len(terrain[y + dy]) and terrain[y + dy][x + dx] == "water")
    return "grass" if neighbors == 0 else "water"


def _route_fallback_path(route: dict[str, Any], nodes: list[dict[str, Any]]) -> list[dict[str, int]]:
    source = next((node for node in nodes if node["id"] == route["sourceId"]), None)
    destination = next((node for node in nodes if node["id"] == route["destinationId"]), None)
    if source is None or destination is None:
        return []
    path = [{"x": source["x"], "y": source["y"]}]
    x = source["x"]
    y = source["y"]
    while x != destination["x"]:
        x += 1 if destination["x"] > x else -1
        path.append({"x": x, "y": y})
    while y != destination["y"]:
        y += 1 if destination["y"] > y else -1
        path.append({"x": x, "y": y})
    return path


def _world_to_iso(x: int, y: int, camera: Camera) -> tuple[int, int]:
    return (camera.origin_x + (x - y) * (TILE_W // 2), camera.origin_y + (x + y) * TILE_Y_STEP)


def _tile_center(x: int, y: int, camera: Camera) -> tuple[int, int]:
    iso_x, iso_y = _world_to_iso(x, y, camera)
    return (iso_x, iso_y + TILE_H // 2)


def _is_near_node(nodes: list[dict[str, Any]], x: int, y: int, radius: int) -> bool:
    return any(math.hypot(node["x"] - x, node["y"] - y) <= radius for node in nodes)


def _hash01(seed: int, x: int, y: int, salt: int) -> float:
    value = ((seed + 0x9E3779B9) * 374761393) & 0xFFFFFFFF
    value ^= ((x + salt * 17) * 668265263) & 0xFFFFFFFF
    value ^= ((y + salt * 31) * 2246822519) & 0xFFFFFFFF
    value ^= value >> 13
    value = (value * 1274126177) & 0xFFFFFFFF
    return ((value ^ (value >> 16)) & 0xFFFFFFFF) / 4294967296


def _pick(items: list[SpriteDef], seed: int, x: int, y: int, salt: int) -> SpriteDef:
    return items[int(_hash01(seed, x, y, salt) * len(items)) % len(items)]
