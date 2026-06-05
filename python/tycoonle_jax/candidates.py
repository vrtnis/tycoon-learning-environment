from __future__ import annotations

import jax
import jax.numpy as jnp

from tycoonle_jax.constants import (
    ACTION_ADD_VEHICLE,
    ACTION_BUILD_ROUTE,
    ACTION_INVALID,
    ACTION_REPAY_LOAN,
    ACTION_TAKE_LOAN,
    ACTION_WAIT,
    CANDIDATE_FEATURES,
    CARGO_COUNT,
    CARGO_VALUES,
    FAMILY_NAMES,
    MAX_CANDIDATES,
    MAX_HEIGHT,
    MAX_NODES,
    MAX_PATH,
    MAX_ROUTES,
    MAX_WIDTH,
    METRIC_COUNT,
    MODE_CAPACITY,
    MODE_COUNT,
    MODE_MAINTENANCE,
    MODE_RAIL,
    MODE_ROAD,
    MODE_SPEED,
    MODE_STATION_COST,
    MODE_TRACK_COST,
    MODE_VEHICLE_COST,
    NODE_FEATURES,
    NODE_PROCESSOR,
    NODE_TOWN,
    ROUTE_FEATURES,
    SCORE_BREAKDOWN_COUNT,
    TERRAIN_COUNT,
    TERRAIN_GRASS,
    TERRAIN_ROUGH,
    TERRAIN_TOWN,
    TERRAIN_WATER,
)
from tycoonle_jax.types import CandidateTable, Observation, State
from tycoonle_jax.utils import build_occupancy, path_valid_mask, round_to

BUILD_CANDIDATE_COUNT = MAX_NODES * MAX_NODES * CARGO_COUNT * MODE_COUNT
ALL_CANDIDATE_COUNT = BUILD_CANDIDATE_COUNT + MAX_ROUTES + 4


def refresh_derived(state: State) -> State:
    metrics, score_breakdown = compute_metrics(state)
    state = state._replace(metrics=metrics, score_breakdown=score_breakdown)
    candidate, action_mask = compute_candidates(state)
    return state._replace(candidate=candidate, action_mask=action_mask)


def compute_metrics(state: State) -> tuple[jnp.ndarray, jnp.ndarray]:
    route_mask_f = state.route_mask.astype(jnp.float32)
    delivered = jnp.sum(jnp.where(state.route_mask, state.route_delivered, 0.0))
    profit = jnp.sum(jnp.where(state.route_mask, state.route_profit, 0.0))
    vehicles = jnp.sum(jnp.where(state.route_mask, state.route_vehicles, 0.0))
    route_count = jnp.sum(route_mask_f)
    utilization = jnp.where(route_count > 0, jnp.sum(state.route_utilization * route_mask_f) / route_count, 0.0)
    in_transit = jnp.sum(jnp.where(state.route_mask[:, None], state.route_transit_amount, 0.0))
    avg_reliability = jnp.where(route_count > 0, jnp.sum(state.route_reliability * route_mask_f) / route_count, 1.0)
    congestion = jnp.where(route_count > 0, jnp.sum(state.route_congestion * route_mask_f) / route_count, 0.0)
    town_growth = jnp.sum(jnp.where(state.node_kind == NODE_TOWN, jnp.maximum(0.0, state.node_population - 700.0), 0.0))
    production_nodes = state.node_mask.astype(jnp.float32)
    production_index = jnp.sum(state.node_production_index * production_nodes) / jnp.maximum(1.0, jnp.sum(production_nodes))
    late_shipments = jnp.sum(jnp.where(state.route_mask & (state.route_last_delay > 0), 1.0, 0.0))
    objective_delivered = jnp.where(
        state.objective_cargo < 0,
        delivered,
        jnp.sum(jnp.where(state.route_mask & (state.route_cargo == state.objective_cargo), state.route_delivered, 0.0)),
    )
    debt_ratio = state.loan / jnp.maximum(1.0, state.max_loan)
    first_delivery = state.first_delivery_month
    cargo_score = jnp.minimum(34.0, (34.0 * objective_delivered) / jnp.maximum(1.0, state.delivered_target))
    profit_score = jnp.minimum(25.0, (25.0 * jnp.maximum(0.0, profit)) / jnp.maximum(1.0, state.profit_target))
    route_score = jnp.minimum(12.0, (12.0 * route_count) / jnp.maximum(1.0, state.route_target))
    debt_score = jnp.maximum(0.0, 14.0 * (1.0 - debt_ratio / state.max_debt_ratio))
    delivery_score = jnp.where(first_delivery < 0, 0.0, jnp.maximum(0.0, 8.0 * (1.0 - first_delivery / jnp.maximum(1.0, state.max_months))))
    utilization_score = jnp.minimum(7.0, utilization * 7.0)
    reliability_score = jnp.minimum(6.0, avg_reliability * 6.0)
    congestion_penalty = jnp.minimum(6.0, congestion * 0.7)
    invalid_penalty = jnp.minimum(25.0, state.invalid_actions.astype(jnp.float32) * 4.0)
    score = jnp.clip(
        cargo_score
        + profit_score
        + route_score
        + debt_score
        + delivery_score
        + utilization_score
        + reliability_score
        - congestion_penalty
        - invalid_penalty,
        0.0,
        100.0,
    )
    metrics = jnp.array(
        [
            round_to(score, 3),
            round_to(delivered, 3),
            round_to(profit, 3),
            round_to(delivered * 0.08 + jnp.maximum(0.0, profit) * 0.0007 + route_count * 5.0, 3),
            route_count,
            vehicles,
            state.invalid_actions.astype(jnp.float32),
            first_delivery.astype(jnp.float32),
            round_to(debt_ratio, 3),
            round_to(utilization, 3),
            round_to(in_transit, 3),
            round_to(avg_reliability, 3),
            round_to(congestion, 3),
            round_to(town_growth, 3),
            round_to(production_index, 3),
            late_shipments,
        ],
        dtype=jnp.float32,
    )
    breakdown = jnp.array(
        [
            round_to(cargo_score, 3),
            round_to(profit_score, 3),
            round_to(route_score, 3),
            round_to(debt_score, 3),
            round_to(delivery_score, 3),
            round_to(utilization_score, 3),
            round_to(reliability_score, 3),
            round_to(-congestion_penalty, 3),
            round_to(-invalid_penalty, 3),
        ],
        dtype=jnp.float32,
    )
    return metrics, breakdown


def compute_candidates(state: State) -> tuple[CandidateTable, jnp.ndarray]:
    build = _build_route_candidates(state)
    add = _add_vehicle_candidates(state)
    extra = _finance_and_wait_candidates(state, build, add)
    all_candidate = _concat_candidates(build, add, extra)

    sort_key = (
        all_candidate.feasible.astype(jnp.float32) * 1_000_000.0
        + all_candidate.directly_executable.astype(jnp.float32) * 100_000.0
        + all_candidate.rank_score
        - (all_candidate.kind == ACTION_INVALID).astype(jnp.float32) * 1_000_000.0
    )
    order = jnp.argsort(-sort_key)[:MAX_CANDIDATES]
    candidate = _take_candidates(all_candidate, order)
    return candidate, candidate.directly_executable & (candidate.kind != ACTION_INVALID)


def make_observation(state: State) -> Observation:
    max_loan = jnp.maximum(1.0, state.max_loan)
    terrain_counts = jnp.array([jnp.sum((state.terrain == idx) & state.terrain_mask) for idx in range(TERRAIN_COUNT)], dtype=jnp.float32)
    total_tiles = jnp.maximum(1.0, jnp.sum(state.terrain_mask.astype(jnp.float32)))
    node_features = _node_features(state)
    route_features = _route_features(state)
    candidate_features = _candidate_features(state)
    return Observation(
        company=jnp.array(
            [
                state.cash / max_loan,
                state.loan / max_loan,
                (state.max_loan - state.loan) / max_loan,
                (state.cash - state.loan) / max_loan,
            ],
            dtype=jnp.float32,
        ),
        time=jnp.array([state.step / jnp.maximum(1, state.max_steps), state.month / jnp.maximum(1, state.max_months)], dtype=jnp.float32),
        objective=jnp.array(
            [
                state.family.astype(jnp.float32) / jnp.maximum(1.0, len(FAMILY_NAMES) - 1),
                jnp.where(state.objective_cargo < 0, -1.0, state.objective_cargo.astype(jnp.float32) / jnp.maximum(1.0, CARGO_COUNT - 1)),
                state.delivered_target / 2000.0,
                state.profit_target / 150_000.0,
            ],
            dtype=jnp.float32,
        ),
        metrics=state.metrics,
        score_breakdown=state.score_breakdown,
        terrain=state.terrain,
        terrain_summary=terrain_counts / total_tiles,
        node_features=node_features,
        route_features=route_features,
        candidate_features=candidate_features,
        action_mask=state.action_mask,
        node_mask=state.node_mask,
        route_mask=state.route_mask,
    )


def _build_route_candidates(state: State) -> CandidateTable:
    idx = jnp.arange(BUILD_CANDIDATE_COUNT, dtype=jnp.int32)
    mode = idx % MODE_COUNT
    cargo = (idx // MODE_COUNT) % CARGO_COUNT
    destination = (idx // (MODE_COUNT * CARGO_COUNT)) % MAX_NODES
    source = (idx // (MODE_COUNT * CARGO_COUNT * MAX_NODES)) % MAX_NODES
    vehicles = jnp.where(mode == MODE_RAIL, 1, 2).astype(jnp.int32)
    node_ok = state.node_mask[source] & state.node_mask[destination] & (source != destination)
    source_has = (state.node_produces[source, cargo] > 0.0) | (state.node_storage[source, cargo] > 0.0)
    destination_accepts = state.node_accepts[destination, cargo] > 0.0
    duplicate = jnp.any(
        state.route_mask[None, :]
        & (state.route_source[None, :] == source[:, None])
        & (state.route_destination[None, :] == destination[:, None])
        & (state.route_cargo[None, :] == cargo[:, None]),
        axis=1,
    )
    path = state.template_path[source, destination, mode]
    path_length = state.template_path_length[source, destination, mode]
    water_tiles = state.template_water_tiles[source, destination, mode]
    rough_tiles = state.template_rough_tiles[source, destination, mode]
    town_tiles = state.template_town_tiles[source, destination, mode]
    road_grid, rail_grid = build_occupancy(state.route_mask, state.route_path, state.route_path_length, state.route_mode)
    px = path[:, :, 0]
    py = path[:, :, 1]
    path_valid = (jnp.arange(MAX_PATH)[None, :] < path_length[:, None]) & (jnp.arange(MAX_PATH)[None, :] > 0)
    road_count = road_grid[py, px]
    rail_count = rail_grid[py, px]
    same_count = jnp.where(mode[:, None] == MODE_ROAD, road_count, rail_count)
    other_count = jnp.where(mode[:, None] == MODE_ROAD, rail_count, road_count)
    shared_tiles = jnp.sum(jnp.where(path_valid & (same_count > 0), 1.0, 0.0), axis=1)
    crossing_tiles = jnp.sum(jnp.where(path_valid & (other_count > 0), 1.0, 0.0), axis=1)
    occupancy_weight = jnp.sum(
        jnp.where(path_valid, jnp.where(same_count > 0, 0.35 + same_count * 0.16, 0.0) + jnp.where(other_count > 0, 2.4 + other_count * 0.35, 0.0), 0.0),
        axis=1,
    )
    weighted_distance = state.template_weighted_distance[source, destination, mode] + occupancy_weight
    distance = jnp.maximum(1.0, path_length.astype(jnp.float32) - 1.0)
    terrain_cost = weighted_distance / distance
    congestion = shared_tiles * 0.18 + crossing_tiles * 0.55 + town_tiles * jnp.where(mode == MODE_RAIL, 0.12, 0.04)
    bridge_cost = water_tiles * jnp.where(mode == MODE_RAIL, 8500.0, 18500.0)
    crossing_cost = crossing_tiles * 6500.0
    shared_credit = shared_tiles * MODE_TRACK_COST[mode] * 0.45
    build_cost = MODE_STATION_COST[mode] * 2.0 + MODE_TRACK_COST[mode] * weighted_distance + bridge_cost + crossing_cost - shared_credit
    vehicle_cost = MODE_VEHICLE_COST[mode] * vehicles.astype(jnp.float32)
    total_cost = round_to(build_cost + vehicle_cost, 0)
    capacity = MODE_CAPACITY[mode] * MODE_SPEED[mode] * vehicles.astype(jnp.float32)
    source_production = jnp.maximum(state.node_produces[source, cargo], state.node_storage[source, cargo])
    accepted = state.node_accepts[destination, cargo]
    drag = jnp.maximum(0.28, 1.0 - distance / 155.0 - congestion * 0.025)
    monthly_delivered = jnp.minimum(jnp.minimum(source_production, accepted), capacity * drag)
    revenue = monthly_delivered * CARGO_VALUES[cargo] * (1.0 + distance / 125.0)
    operating_cost = MODE_MAINTENANCE[mode] * vehicles.astype(jnp.float32) + distance * 7.5 * terrain_cost + congestion * 85.0
    monthly_profit = revenue - operating_cost
    roi = monthly_profit / jnp.maximum(1.0, total_cost)
    build_feasible = (path_length > 1) & ~((mode == MODE_ROAD) & (water_tiles > 2.0))
    route_capacity_left = jnp.sum(state.route_mask.astype(jnp.int32)) < MAX_ROUTES
    can_finance = state.cash + jnp.maximum(0.0, state.max_loan - state.loan) >= total_cost
    objective_boost = jnp.where((state.objective_cargo < 0) | (state.objective_cargo == cargo), 1.8, 0.35)
    missing_upstream = (
        (state.node_kind[source] == NODE_PROCESSOR)
        & (state.node_convert_from[source] >= 0)
        & ~jnp.any(state.route_mask[None, :] & (state.route_destination[None, :] == source[:, None]), axis=1)
    )
    feasible = node_ok & source_has & destination_accepts & ~duplicate & build_feasible & can_finance & route_capacity_left
    directly = feasible & (state.cash >= total_cost)
    rank = round_to(roi * 1000.0 + objective_boost - terrain_cost * 0.1 - congestion * 0.6 - missing_upstream.astype(jnp.float32) * 35.0, 5)
    diagnostics = jnp.stack(
        [
            feasible & ~directly,
            build_feasible & ~can_finance,
            ~build_feasible,
            water_tiles > 0.0,
            town_tiles > 2.0,
            crossing_tiles > 0.0,
            shared_tiles > 0.0,
            missing_upstream,
        ],
        axis=1,
    )
    return CandidateTable(
        kind=jnp.full((BUILD_CANDIDATE_COUNT,), ACTION_BUILD_ROUTE, dtype=jnp.int32),
        source=source,
        destination=destination,
        cargo=cargo,
        mode=mode,
        route=jnp.zeros((BUILD_CANDIDATE_COUNT,), dtype=jnp.int32),
        vehicles=vehicles,
        months=jnp.zeros((BUILD_CANDIDATE_COUNT,), dtype=jnp.int32),
        amount=jnp.zeros((BUILD_CANDIDATE_COUNT,), dtype=jnp.float32),
        total_cost=total_cost,
        monthly_profit=round_to(monthly_profit, 2),
        monthly_delivered=round_to(monthly_delivered, 2),
        rank_score=rank,
        requires_loan=jnp.maximum(0.0, total_cost - state.cash),
        feasible=feasible,
        directly_executable=directly,
        path=path,
        path_length=path_length,
        terrain_cost=round_to(terrain_cost, 3),
        congestion=round_to(congestion, 3),
        diagnostics=diagnostics,
    )


def _add_vehicle_candidates(state: State) -> CandidateTable:
    route = jnp.arange(MAX_ROUTES, dtype=jnp.int32)
    cost = state.route_vehicle_cost
    feasible = state.route_mask & (state.cash + jnp.maximum(0.0, state.max_loan - state.loan) >= cost)
    directly = feasible & (state.cash >= cost)
    age = jnp.maximum(1.0, state.route_age_months.astype(jnp.float32))
    rank = round_to((state.route_profit / age + 800.0) / jnp.maximum(1.0, cost), 5)
    diagnostics = jnp.zeros((MAX_ROUTES, 8), dtype=jnp.bool_).at[:, 0].set(feasible & ~directly)
    return CandidateTable(
        kind=jnp.full((MAX_ROUTES,), ACTION_ADD_VEHICLE, dtype=jnp.int32),
        source=state.route_source,
        destination=state.route_destination,
        cargo=state.route_cargo,
        mode=state.route_mode,
        route=route,
        vehicles=jnp.ones((MAX_ROUTES,), dtype=jnp.int32),
        months=jnp.zeros((MAX_ROUTES,), dtype=jnp.int32),
        amount=jnp.zeros((MAX_ROUTES,), dtype=jnp.float32),
        total_cost=cost,
        monthly_profit=state.route_profit,
        monthly_delivered=state.route_delivered,
        rank_score=rank,
        requires_loan=jnp.maximum(0.0, cost - state.cash),
        feasible=feasible,
        directly_executable=directly,
        path=state.route_path,
        path_length=state.route_path_length,
        terrain_cost=state.route_terrain_cost,
        congestion=state.route_congestion,
        diagnostics=diagnostics,
    )


def _finance_and_wait_candidates(state: State, build: CandidateTable, add: CandidateTable) -> CandidateTable:
    all_requires = jnp.concatenate([build.requires_loan, add.requires_loan], axis=0)
    all_unlock_mask = jnp.concatenate([build.feasible & ~build.directly_executable, add.feasible & ~add.directly_executable], axis=0)
    available_loan = jnp.maximum(0.0, state.max_loan - state.loan)
    required = jnp.max(jnp.where(all_unlock_mask, all_requires, 0.0))
    loan_amount = jnp.minimum(available_loan, required + 12_000.0)
    take_loan_ok = (loan_amount > 0.0) & jnp.any(all_unlock_mask)
    repay_amount = jnp.minimum(state.loan, jnp.maximum(0.0, state.cash - 70_000.0))
    repay_ok = (state.cash > 90_000.0) & (state.loan > 0.0) & (repay_amount > 0.0)
    has_routes = jnp.any(state.route_mask)
    kinds = jnp.array([ACTION_TAKE_LOAN, ACTION_REPAY_LOAN, ACTION_WAIT, ACTION_WAIT], dtype=jnp.int32)
    months = jnp.array([0, 0, 1, 3], dtype=jnp.int32)
    amount = jnp.array([round_to(loan_amount, 0), round_to(repay_amount, 0), 0.0, 0.0], dtype=jnp.float32)
    feasible = jnp.array([take_loan_ok, repay_ok, True, has_routes], dtype=jnp.bool_)
    directly = feasible
    rank = jnp.array([0.75, 0.2, jnp.where(has_routes, 0.4 + _average_route_roi(state), -0.1), jnp.where(has_routes, 0.45 + _average_route_roi(state), -10.0)], dtype=jnp.float32)
    zeros_i = jnp.zeros((4,), dtype=jnp.int32)
    zeros_f = jnp.zeros((4,), dtype=jnp.float32)
    return CandidateTable(
        kind=kinds,
        source=zeros_i,
        destination=zeros_i,
        cargo=zeros_i,
        mode=zeros_i,
        route=zeros_i,
        vehicles=zeros_i,
        months=months,
        amount=amount,
        total_cost=zeros_f,
        monthly_profit=zeros_f,
        monthly_delivered=zeros_f,
        rank_score=rank,
        requires_loan=zeros_f,
        feasible=feasible,
        directly_executable=directly,
        path=jnp.zeros((4, MAX_PATH, 2), dtype=jnp.int32),
        path_length=zeros_i,
        terrain_cost=zeros_f,
        congestion=zeros_f,
        diagnostics=jnp.zeros((4, 8), dtype=jnp.bool_),
    )


def _average_route_roi(state: State) -> jnp.ndarray:
    route_count = jnp.sum(state.route_mask.astype(jnp.float32))
    roi = state.route_profit / jnp.maximum(1.0, state.route_build_cost)
    return jnp.where(route_count > 0.0, jnp.sum(jnp.where(state.route_mask, roi, 0.0)) / route_count, 0.0)


def _concat_candidates(*tables: CandidateTable) -> CandidateTable:
    return CandidateTable(*(jnp.concatenate([getattr(table, field) for table in tables], axis=0) for field in CandidateTable._fields))


def _take_candidates(table: CandidateTable, order: jnp.ndarray) -> CandidateTable:
    return CandidateTable(*(jnp.take(getattr(table, field), order, axis=0) for field in CandidateTable._fields))


def _candidate_features(state: State) -> jnp.ndarray:
    c = state.candidate
    max_loan = jnp.maximum(1.0, state.max_loan)
    features = jnp.stack(
        [
            c.feasible.astype(jnp.float32),
            c.directly_executable.astype(jnp.float32),
            c.requires_loan / max_loan,
            jnp.clip(c.rank_score / 10.0, -10.0, 10.0),
            c.kind.astype(jnp.float32) / 5.0,
            jnp.where(c.kind == ACTION_BUILD_ROUTE, c.cargo.astype(jnp.float32) / jnp.maximum(1.0, CARGO_COUNT - 1), -1.0),
            c.mode.astype(jnp.float32) / 1.0,
            c.total_cost / max_loan,
            jnp.clip(c.monthly_profit / 25_000.0, -10.0, 10.0),
            jnp.clip(c.monthly_delivered / 250.0, -10.0, 10.0),
            c.terrain_cost / 12.0,
            c.congestion / 8.0,
        ],
        axis=1,
    )
    return features.astype(jnp.float32)


def _node_features(state: State) -> jnp.ndarray:
    total_storage = jnp.sum(state.node_storage, axis=1)
    total_accepts = jnp.sum(state.node_accepts, axis=1)
    total_produces = jnp.sum(state.node_produces, axis=1)
    features = jnp.stack(
        [
            state.node_mask.astype(jnp.float32),
            state.node_kind.astype(jnp.float32) / 4.0,
            state.node_x.astype(jnp.float32) / MAX_WIDTH,
            state.node_y.astype(jnp.float32) / MAX_HEIGHT,
            total_produces / 200.0,
            total_accepts / 200.0,
            total_storage / 500.0,
            state.node_population / 3000.0,
            state.node_production_index,
            state.node_rating,
            state.node_service_months.astype(jnp.float32) / 64.0,
            (state.node_last_served_month.astype(jnp.float32) + 999.0) / 1100.0,
            state.node_produces[:, 0] / 200.0,
            state.node_accepts[:, 0] / 200.0,
            jnp.maximum(state.node_convert_from, -1).astype(jnp.float32) / 10.0,
            jnp.maximum(state.node_convert_to, -1).astype(jnp.float32) / 10.0,
        ],
        axis=1,
    )
    return features.astype(jnp.float32)


def _route_features(state: State) -> jnp.ndarray:
    features = jnp.stack(
        [
            state.route_mask.astype(jnp.float32),
            state.route_source.astype(jnp.float32) / MAX_NODES,
            state.route_destination.astype(jnp.float32) / MAX_NODES,
            state.route_cargo.astype(jnp.float32) / CARGO_COUNT,
            state.route_mode.astype(jnp.float32),
            state.route_distance / 100.0,
            state.route_terrain_cost / 12.0,
            state.route_build_cost / 600_000.0,
            state.route_vehicle_cost / 100_000.0,
            state.route_vehicles / 12.0,
            state.route_delivered / 2000.0,
            jnp.clip(state.route_profit / 150_000.0, -10.0, 10.0),
            state.route_utilization,
            state.route_age_months.astype(jnp.float32) / 64.0,
            state.route_travel_time_months.astype(jnp.float32) / 16.0,
            state.route_reliability,
            state.route_congestion / 8.0,
            jnp.sum(state.route_transit_amount, axis=1) / 1000.0,
        ],
        axis=1,
    )
    return features.astype(jnp.float32)
