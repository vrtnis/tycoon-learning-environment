from __future__ import annotations

import jax
import jax.numpy as jnp

from tycoonle_jax.constants import (
    ACTION_INVALID,
    CARGO_COAL,
    CARGO_FOOD,
    CARGO_GOODS,
    CARGO_GRAIN,
    CARGO_IRON_ORE,
    CARGO_LUMBER,
    CARGO_MAIL,
    CARGO_PASSENGERS,
    CARGO_STEEL,
    CARGO_WOOD,
    FAMILY_CHAIN,
    FAMILY_LOW_CASH,
    FAMILY_MIXED_NETWORK,
    FAMILY_TERRAIN_GAP,
    MAX_CANDIDATES,
    MAX_HEIGHT,
    MAX_NODES,
    MAX_PATH,
    MAX_ROUTES,
    MAX_TRANSIT,
    MAX_WIDTH,
    METRIC_COUNT,
    MODE_COUNT,
    NODE_CONSUMER,
    NODE_PROCESSOR,
    NODE_PRODUCER,
    NODE_TOWN,
    SCORE_BREAKDOWN_COUNT,
    TERRAIN_GRASS,
    TERRAIN_ROUGH,
    TERRAIN_TOWN,
    TERRAIN_WATER,
)
from tycoonle_jax.types import CandidateTable, State
from tycoonle_jax.utils import direct_template_path, gather_path_terrain, hash01, path_valid_mask, split_seed


def initial_state(key: jax.Array, *, split: int, family: int) -> State:
    key, seed_key, world_key, terrain_key = jax.random.split(key, 4)
    seed = jax.random.randint(seed_key, (), 1, 1_000_000, dtype=jnp.int32)
    scenario_seed = split_seed(split, seed)
    width, height, max_steps, max_months, starting_cash, max_loan, interest_rate = _budget_and_size(world_key, family)
    nodes = _build_nodes(world_key, family, width, height)
    terrain_mask = _terrain_mask(width, height)
    terrain = _mark_settlement_terrain(
        _generate_terrain(terrain_key, scenario_seed, family, width, height, terrain_mask),
        nodes,
    )
    templates = _route_templates(terrain, nodes)
    objective_cargo, delivered_target, profit_target, route_target, max_debt_ratio = _objective(family, nodes)

    return State(
        key=key,
        split=jnp.asarray(split, dtype=jnp.int32),
        family=jnp.asarray(family, dtype=jnp.int32),
        seed=seed,
        width=width,
        height=height,
        terrain=terrain,
        terrain_mask=terrain_mask,
        node_mask=nodes["mask"],
        node_kind=nodes["kind"],
        node_x=nodes["x"],
        node_y=nodes["y"],
        node_produces=nodes["produces"],
        node_accepts=nodes["accepts"],
        node_storage=nodes["storage"],
        node_base_production=nodes["base_production"],
        node_convert_from=nodes["convert_from"],
        node_convert_to=nodes["convert_to"],
        node_population=nodes["population"],
        node_production_index=jnp.where(nodes["mask"], 1.0, 0.0).astype(jnp.float32),
        node_service_months=jnp.zeros((MAX_NODES,), dtype=jnp.int32),
        node_last_served_month=jnp.full((MAX_NODES,), -999, dtype=jnp.int32),
        node_rating=jnp.where(nodes["kind"] == NODE_TOWN, 0.68, 0.72).astype(jnp.float32),
        objective_cargo=objective_cargo,
        delivered_target=delivered_target,
        profit_target=profit_target,
        route_target=route_target,
        max_debt_ratio=max_debt_ratio,
        max_steps=max_steps,
        max_months=max_months,
        max_loan=max_loan,
        interest_rate=interest_rate,
        step=jnp.array(0, dtype=jnp.int32),
        month=jnp.array(0, dtype=jnp.int32),
        cash=starting_cash,
        loan=jnp.array(0.0, dtype=jnp.float32),
        done=jnp.array(False),
        invalid_actions=jnp.array(0, dtype=jnp.int32),
        first_delivery_month=jnp.array(-1, dtype=jnp.int32),
        route_mask=jnp.zeros((MAX_ROUTES,), dtype=jnp.bool_),
        route_source=jnp.zeros((MAX_ROUTES,), dtype=jnp.int32),
        route_destination=jnp.zeros((MAX_ROUTES,), dtype=jnp.int32),
        route_cargo=jnp.zeros((MAX_ROUTES,), dtype=jnp.int32),
        route_mode=jnp.zeros((MAX_ROUTES,), dtype=jnp.int32),
        route_path=jnp.zeros((MAX_ROUTES, MAX_PATH, 2), dtype=jnp.int32),
        route_path_length=jnp.zeros((MAX_ROUTES,), dtype=jnp.int32),
        route_distance=jnp.zeros((MAX_ROUTES,), dtype=jnp.float32),
        route_terrain_cost=jnp.zeros((MAX_ROUTES,), dtype=jnp.float32),
        route_build_cost=jnp.zeros((MAX_ROUTES,), dtype=jnp.float32),
        route_vehicle_cost=jnp.zeros((MAX_ROUTES,), dtype=jnp.float32),
        route_vehicles=jnp.zeros((MAX_ROUTES,), dtype=jnp.float32),
        route_delivered=jnp.zeros((MAX_ROUTES,), dtype=jnp.float32),
        route_revenue=jnp.zeros((MAX_ROUTES,), dtype=jnp.float32),
        route_operating_cost=jnp.zeros((MAX_ROUTES,), dtype=jnp.float32),
        route_profit=jnp.zeros((MAX_ROUTES,), dtype=jnp.float32),
        route_utilization=jnp.zeros((MAX_ROUTES,), dtype=jnp.float32),
        route_age_months=jnp.zeros((MAX_ROUTES,), dtype=jnp.int32),
        route_travel_time_months=jnp.zeros((MAX_ROUTES,), dtype=jnp.int32),
        route_last_delivered=jnp.zeros((MAX_ROUTES,), dtype=jnp.float32),
        route_last_delay=jnp.zeros((MAX_ROUTES,), dtype=jnp.int32),
        route_reliability=jnp.ones((MAX_ROUTES,), dtype=jnp.float32),
        route_congestion=jnp.zeros((MAX_ROUTES,), dtype=jnp.float32),
        route_station_rating=jnp.full((MAX_ROUTES,), 0.72, dtype=jnp.float32),
        route_transit_amount=jnp.zeros((MAX_ROUTES, MAX_TRANSIT), dtype=jnp.float32),
        route_transit_remaining=jnp.zeros((MAX_ROUTES, MAX_TRANSIT), dtype=jnp.int32),
        template_path=templates["path"],
        template_path_length=templates["length"],
        template_weighted_distance=templates["weighted_distance"],
        template_water_tiles=templates["water_tiles"],
        template_rough_tiles=templates["rough_tiles"],
        template_town_tiles=templates["town_tiles"],
        candidate=empty_candidate_table(),
        action_mask=jnp.zeros((MAX_CANDIDATES,), dtype=jnp.bool_),
        metrics=jnp.zeros((METRIC_COUNT,), dtype=jnp.float32),
        score_breakdown=jnp.zeros((SCORE_BREAKDOWN_COUNT,), dtype=jnp.float32),
    )


def empty_candidate_table() -> CandidateTable:
    zeros_i = jnp.zeros((MAX_CANDIDATES,), dtype=jnp.int32)
    zeros_f = jnp.zeros((MAX_CANDIDATES,), dtype=jnp.float32)
    zeros_b = jnp.zeros((MAX_CANDIDATES,), dtype=jnp.bool_)
    return CandidateTable(
        kind=jnp.full((MAX_CANDIDATES,), ACTION_INVALID, dtype=jnp.int32),
        source=zeros_i,
        destination=zeros_i,
        cargo=zeros_i,
        mode=zeros_i,
        route=zeros_i,
        vehicles=zeros_i,
        months=zeros_i,
        amount=zeros_f,
        total_cost=zeros_f,
        monthly_profit=zeros_f,
        monthly_delivered=zeros_f,
        rank_score=zeros_f,
        requires_loan=zeros_f,
        feasible=zeros_b,
        directly_executable=zeros_b,
        path=jnp.zeros((MAX_CANDIDATES, MAX_PATH, 2), dtype=jnp.int32),
        path_length=zeros_i,
        terrain_cost=zeros_f,
        congestion=zeros_f,
        diagnostics=jnp.zeros((MAX_CANDIDATES, 8), dtype=jnp.bool_),
    )


def _budget_and_size(key: jax.Array, family: int) -> tuple[jnp.ndarray, ...]:
    low_cash_cash = jax.random.randint(key, (), 58_000, 82_001, dtype=jnp.int32).astype(jnp.float32)
    if family == FAMILY_MIXED_NETWORK:
        return (
            jnp.array(54, dtype=jnp.int32),
            jnp.array(38, dtype=jnp.int32),
            jnp.array(28, dtype=jnp.int32),
            jnp.array(64, dtype=jnp.int32),
            jnp.array(265_000.0, dtype=jnp.float32),
            jnp.array(540_000.0, dtype=jnp.float32),
            jnp.array(0.049, dtype=jnp.float32),
        )
    if family == FAMILY_CHAIN:
        return (
            jnp.array(46, dtype=jnp.int32),
            jnp.array(32, dtype=jnp.int32),
            jnp.array(26, dtype=jnp.int32),
            jnp.array(60, dtype=jnp.int32),
            jnp.array(245_000.0, dtype=jnp.float32),
            jnp.array(520_000.0, dtype=jnp.float32),
            jnp.array(0.052, dtype=jnp.float32),
        )
    if family == FAMILY_LOW_CASH:
        return (
            jnp.array(46, dtype=jnp.int32),
            jnp.array(32, dtype=jnp.int32),
            jnp.array(18, dtype=jnp.int32),
            jnp.array(48, dtype=jnp.int32),
            low_cash_cash,
            jnp.array(310_000.0, dtype=jnp.float32),
            jnp.array(0.062, dtype=jnp.float32),
        )
    return (
        jnp.array(46, dtype=jnp.int32),
        jnp.array(32, dtype=jnp.int32),
        jnp.array(18, dtype=jnp.int32),
        jnp.array(46, dtype=jnp.int32),
        jnp.array(185_000.0, dtype=jnp.float32),
        jnp.array(360_000.0, dtype=jnp.float32),
        jnp.array(0.045, dtype=jnp.float32),
    )


def _build_nodes(key: jax.Array, family: int, width: jnp.ndarray, height: jnp.ndarray) -> dict[str, jnp.ndarray]:
    keys = jax.random.split(key, 12)
    mask = jnp.zeros((MAX_NODES,), dtype=jnp.bool_)
    kind = jnp.zeros((MAX_NODES,), dtype=jnp.int32)
    x = jnp.zeros((MAX_NODES,), dtype=jnp.int32)
    y = jnp.zeros((MAX_NODES,), dtype=jnp.int32)
    produces = jnp.zeros((MAX_NODES, 10), dtype=jnp.float32)
    accepts = jnp.zeros((MAX_NODES, 10), dtype=jnp.float32)
    storage = jnp.zeros((MAX_NODES, 10), dtype=jnp.float32)
    base_production = jnp.zeros((MAX_NODES, 10), dtype=jnp.float32)
    convert_from = jnp.full((MAX_NODES,), -1, dtype=jnp.int32)
    convert_to = jnp.full((MAX_NODES,), -1, dtype=jnp.int32)
    population = jnp.zeros((MAX_NODES,), dtype=jnp.float32)

    def put_node(idx: int, node_kind: int, nx: jnp.ndarray, ny: jnp.ndarray, prod: dict[int, jnp.ndarray], acc: dict[int, jnp.ndarray]):
        nonlocal mask, kind, x, y, produces, accepts, storage, base_production, population
        mask = mask.at[idx].set(True)
        kind = kind.at[idx].set(node_kind)
        x = x.at[idx].set(nx.astype(jnp.int32))
        y = y.at[idx].set(ny.astype(jnp.int32))
        for cargo, amount in prod.items():
            resolved = jnp.asarray(amount, dtype=jnp.float32)
            produces = produces.at[idx, cargo].set(resolved)
            storage = storage.at[idx, cargo].set(resolved)
            base_production = base_production.at[idx, cargo].set(resolved)
        for cargo, amount in acc.items():
            accepts = accepts.at[idx, cargo].set(jnp.asarray(amount, dtype=jnp.float32))
        pop = 700.0 + jnp.abs((nx.astype(jnp.float32) * 83.0 + ny.astype(jnp.float32) * 41.0) % 1900.0)
        population = population.at[idx].set(jnp.where(node_kind == NODE_TOWN, pop, 0.0))

    def town_accepts(nx: jnp.ndarray, ny: jnp.ndarray) -> dict[int, jnp.ndarray]:
        pop = 700.0 + jnp.abs((nx.astype(jnp.float32) * 83.0 + ny.astype(jnp.float32) * 41.0) % 1900.0)
        return {
            CARGO_PASSENGERS: jnp.round(pop / 22.0),
            CARGO_MAIL: jnp.round(pop / 65.0),
            CARGO_GOODS: jnp.round(pop / 55.0),
        }

    def town_produces(nx: jnp.ndarray, ny: jnp.ndarray) -> dict[int, jnp.ndarray]:
        pop = 700.0 + jnp.abs((nx.astype(jnp.float32) * 83.0 + ny.astype(jnp.float32) * 41.0) % 1900.0)
        return {CARGO_PASSENGERS: jnp.round(pop / 24.0), CARGO_MAIL: jnp.round(pop / 72.0)}

    if family == FAMILY_CHAIN:
        raw = CARGO_IRON_ORE
        mid = CARGO_STEEL
        producer_y = jax.random.randint(keys[1], (), 7, height - 7)
        processor_x = jax.random.randint(keys[2], (), 19, width - 16)
        processor_y = jax.random.randint(keys[3], (), 6, height - 7)
        sink_y = jax.random.randint(keys[4], (), 8, height - 7)
        raw_amount = jax.random.randint(keys[5], (), 95, 146).astype(jnp.float32)
        mid_amount = jax.random.randint(keys[6], (), 62, 106).astype(jnp.float32)
        accept_raw = jax.random.randint(keys[7], (), 100, 151).astype(jnp.float32)
        accept_mid = jax.random.randint(keys[8], (), 80, 131).astype(jnp.float32)
        put_node(0, NODE_PRODUCER, jnp.array(7), producer_y, {raw: raw_amount}, {})
        put_node(1, NODE_PROCESSOR, processor_x, processor_y, {mid: mid_amount}, {raw: accept_raw})
        storage = storage.at[1].set(jnp.zeros((10,), dtype=jnp.float32))
        convert_from = convert_from.at[1].set(raw)
        convert_to = convert_to.at[1].set(mid)
        put_node(2, NODE_TOWN, width - 8, sink_y, town_produces(width - 8, sink_y), {mid: accept_mid})
        put_node(3, NODE_TOWN, jax.random.randint(keys[9], (), 13, width - 13), jax.random.randint(keys[10], (), 8, height - 7), {}, {})
        put_node(4, NODE_TOWN, jax.random.randint(keys[11], (), 12, width - 12), jax.random.randint(keys[0], (), 7, height - 6), {}, {})
    else:
        cargo = CARGO_GRAIN if family == FAMILY_LOW_CASH else CARGO_COAL
        source_x = jax.random.randint(keys[1], (), 5, 14)
        source_y = jax.random.randint(keys[2], (), 6, height - 6)
        dest_x = width - jax.random.randint(keys[3], (), 7, 13)
        dest_y = jax.random.randint(keys[4], (), 6, height - 6)
        source_amount = jax.random.randint(keys[5], (), 90, 146).astype(jnp.float32)
        dest_amount = jax.random.randint(keys[6], (), 95, 151).astype(jnp.float32)
        put_node(0, NODE_PRODUCER, source_x, source_y, {cargo: source_amount}, {})
        if cargo == CARGO_PASSENGERS:
            put_node(1, NODE_TOWN, dest_x, dest_y, town_produces(dest_x, dest_y), {CARGO_PASSENGERS: dest_amount, CARGO_MAIL: 35.0})
        else:
            put_node(1, NODE_CONSUMER, dest_x, dest_y, {}, {cargo: dest_amount})
        next_town_index = 2
        if family == FAMILY_MIXED_NETWORK:
            put_node(2, NODE_PRODUCER, jax.random.randint(keys[7], (), 7, 19), jax.random.randint(keys[8], (), 20, height - 4), {CARGO_WOOD: 95.0}, {})
            put_node(3, NODE_CONSUMER, width - jax.random.randint(keys[9], (), 9, 19), jax.random.randint(keys[10], (), 20, height - 5), {}, {CARGO_WOOD: 100.0})
            next_town_index = 4
        put_node(next_town_index, NODE_TOWN, jax.random.randint(keys[9], (), 13, width - 13), jax.random.randint(keys[10], (), 8, height - 7), {}, {})
        put_node(next_town_index + 1, NODE_TOWN, jax.random.randint(keys[11], (), 12, width - 12), jax.random.randint(keys[0], (), 7, height - 6), {}, {})

    # Fill default town production/acceptance for towns created without explicit maps.
    for idx in range(MAX_NODES):
        is_empty_town = mask[idx] & (kind[idx] == NODE_TOWN) & (jnp.sum(accepts[idx]) == 0)
        pop = population[idx]
        accepts = accepts.at[idx, CARGO_PASSENGERS].set(jnp.where(is_empty_town, jnp.round(pop / 22.0), accepts[idx, CARGO_PASSENGERS]))
        accepts = accepts.at[idx, CARGO_MAIL].set(jnp.where(is_empty_town, jnp.round(pop / 65.0), accepts[idx, CARGO_MAIL]))
        accepts = accepts.at[idx, CARGO_GOODS].set(jnp.where(is_empty_town, jnp.round(pop / 55.0), accepts[idx, CARGO_GOODS]))
        produces = produces.at[idx, CARGO_PASSENGERS].set(jnp.where(is_empty_town, jnp.round(pop / 24.0), produces[idx, CARGO_PASSENGERS]))
        produces = produces.at[idx, CARGO_MAIL].set(jnp.where(is_empty_town, jnp.round(pop / 72.0), produces[idx, CARGO_MAIL]))
        base_production = base_production.at[idx].set(produces[idx])
        storage = storage.at[idx].set(jnp.where(is_empty_town, produces[idx], storage[idx]))

    return {
        "mask": mask,
        "kind": kind,
        "x": x,
        "y": y,
        "produces": produces,
        "accepts": accepts,
        "storage": storage,
        "base_production": base_production,
        "convert_from": convert_from,
        "convert_to": convert_to,
        "population": population,
    }


def _terrain_mask(width: jnp.ndarray, height: jnp.ndarray) -> jnp.ndarray:
    xs = jnp.arange(MAX_WIDTH)[None, :]
    ys = jnp.arange(MAX_HEIGHT)[:, None]
    return (xs < width) & (ys < height)


def _generate_terrain(
    key: jax.Array,
    seed: jnp.ndarray,
    family: int,
    width: jnp.ndarray,
    height: jnp.ndarray,
    terrain_mask: jnp.ndarray,
) -> jnp.ndarray:
    k1, k2, k3, k4 = jax.random.split(key, 4)
    xs = jnp.arange(MAX_WIDTH, dtype=jnp.float32)[None, :]
    ys = jnp.arange(MAX_HEIGHT, dtype=jnp.float32)[:, None]
    n1 = jax.random.uniform(k1, (MAX_HEIGHT, MAX_WIDTH), dtype=jnp.float32)
    n2 = jax.random.uniform(k2, (MAX_HEIGHT, MAX_WIDTH), dtype=jnp.float32)
    ridge = jnp.sin((xs + n1 * 3.0) / 5.0) + jnp.cos((ys - n2 * 2.0) / 6.0)
    rough = (ridge > 1.0) | (jax.random.uniform(k3, (MAX_HEIGHT, MAX_WIDTH)) < 0.07)
    terrain = jnp.where(rough, TERRAIN_ROUGH, TERRAIN_GRASS).astype(jnp.int32)
    random_water = jax.random.uniform(k4, (MAX_HEIGHT, MAX_WIDTH)) < 0.015
    terrain = jnp.where(random_water, TERRAIN_WATER, terrain)
    if family == FAMILY_TERRAIN_GAP:
        river_x = 16 + (seed % jnp.maximum(1, width - 31))
        xgrid = jnp.arange(MAX_WIDTH)[None, :]
        ygrid = jnp.arange(MAX_HEIGHT)[:, None]
        river = (jnp.abs(xgrid - river_x) <= 1) & (ygrid > 3) & (ygrid < height - 4)
        terrain = jnp.where(river, TERRAIN_WATER, terrain)
    return jnp.where(terrain_mask, terrain, TERRAIN_GRASS).astype(jnp.int32)


def _mark_settlement_terrain(terrain: jnp.ndarray, nodes: dict[str, jnp.ndarray]) -> jnp.ndarray:
    xs = jnp.arange(MAX_WIDTH)[None, :]
    ys = jnp.arange(MAX_HEIGHT)[:, None]
    marked = terrain
    for idx in range(MAX_NODES):
        pop = nodes["population"][idx]
        radius = jnp.where(nodes["kind"][idx] == NODE_TOWN, jnp.where(pop > 1800.0, 3, 2), 1)
        dist = jnp.abs(xs - nodes["x"][idx]) + jnp.abs(ys - nodes["y"][idx])
        settlement = nodes["mask"][idx] & (dist <= radius + 1) & (marked != TERRAIN_WATER)
        marked = jnp.where(settlement, TERRAIN_TOWN, marked)
    return marked.astype(jnp.int32)


def _route_templates(terrain: jnp.ndarray, nodes: dict[str, jnp.ndarray]) -> dict[str, jnp.ndarray]:
    path = jnp.zeros((MAX_NODES, MAX_NODES, MODE_COUNT, MAX_PATH, 2), dtype=jnp.int32)
    length = jnp.zeros((MAX_NODES, MAX_NODES, MODE_COUNT), dtype=jnp.int32)
    weighted = jnp.zeros((MAX_NODES, MAX_NODES, MODE_COUNT), dtype=jnp.float32)
    water = jnp.zeros((MAX_NODES, MAX_NODES, MODE_COUNT), dtype=jnp.float32)
    rough = jnp.zeros((MAX_NODES, MAX_NODES, MODE_COUNT), dtype=jnp.float32)
    town = jnp.zeros((MAX_NODES, MAX_NODES, MODE_COUNT), dtype=jnp.float32)
    for src in range(MAX_NODES):
        for dst in range(MAX_NODES):
            route_path, route_length = direct_template_path(nodes["x"][src], nodes["y"][src], nodes["x"][dst], nodes["y"][dst])
            terrain_along = gather_path_terrain(terrain, route_path)
            valid = path_valid_mask(route_length) & (jnp.arange(MAX_PATH) > 0)
            for mode in range(MODE_COUNT):
                terrain_weight = jnp.ones((MAX_PATH,), dtype=jnp.float32)
                terrain_weight += jnp.where(terrain_along == TERRAIN_ROUGH, 0.55 if mode == 1 else 0.35, 0.0)
                terrain_weight += jnp.where(terrain_along == TERRAIN_WATER, 3.6 if mode == 1 else 11.0, 0.0)
                terrain_weight += jnp.where(terrain_along == TERRAIN_TOWN, 1.6 if mode == 1 else 0.35, 0.0)
                path = path.at[src, dst, mode].set(route_path)
                length = length.at[src, dst, mode].set(route_length)
                weighted = weighted.at[src, dst, mode].set(jnp.sum(jnp.where(valid, terrain_weight, 0.0)))
                water = water.at[src, dst, mode].set(jnp.sum(jnp.where(valid & (terrain_along == TERRAIN_WATER), 1.0, 0.0)))
                rough = rough.at[src, dst, mode].set(jnp.sum(jnp.where(valid & (terrain_along == TERRAIN_ROUGH), 1.0, 0.0)))
                town = town.at[src, dst, mode].set(jnp.sum(jnp.where(valid & (terrain_along == TERRAIN_TOWN), 1.0, 0.0)))
    return {
        "path": path,
        "length": length,
        "weighted_distance": weighted,
        "water_tiles": water,
        "rough_tiles": rough,
        "town_tiles": town,
    }


def _objective(family: int, nodes: dict[str, jnp.ndarray]) -> tuple[jnp.ndarray, ...]:
    if family == FAMILY_CHAIN:
        objective_cargo = jnp.argmax(nodes["accepts"][2] > 0).astype(jnp.int32)
        return (
            objective_cargo,
            jnp.array(1350.0, dtype=jnp.float32),
            jnp.array(68_000.0, dtype=jnp.float32),
            jnp.array(2.0, dtype=jnp.float32),
            jnp.array(0.78, dtype=jnp.float32),
        )
    objective_cargo = jnp.argmax(nodes["produces"][0] > 0).astype(jnp.int32)
    delivered_target = jnp.array(1800.0 if family == FAMILY_MIXED_NETWORK else 850.0, dtype=jnp.float32)
    profit_target = jnp.array(120_000.0 if family == FAMILY_MIXED_NETWORK else 42_000.0 if family == FAMILY_LOW_CASH else 68_000.0, dtype=jnp.float32)
    route_target = jnp.array(2.0 if family == FAMILY_MIXED_NETWORK else 1.0, dtype=jnp.float32)
    max_debt_ratio = jnp.array(0.68 if family == FAMILY_LOW_CASH else 0.78, dtype=jnp.float32)
    # Mixed network deliberately rewards all cargo through a sentinel -1 objective cargo.
    objective_cargo = jnp.where(family == FAMILY_MIXED_NETWORK, -1, objective_cargo)
    return objective_cargo, delivered_target, profit_target, route_target, max_debt_ratio
