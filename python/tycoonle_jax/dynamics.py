from __future__ import annotations

import jax
import jax.numpy as jnp

from tycoonle_jax.candidates import refresh_derived
from tycoonle_jax.constants import (
    ACTION_ADD_VEHICLE,
    ACTION_BUILD_ROUTE,
    ACTION_REPAY_LOAN,
    ACTION_TAKE_LOAN,
    ACTION_WAIT,
    CARGO_FOOD,
    CARGO_GOODS,
    CARGO_MAIL,
    CARGO_PASSENGERS,
    CARGO_VALUES,
    MAX_CANDIDATES,
    MAX_NODES,
    MAX_PATH,
    MAX_ROUTES,
    MAX_TRANSIT,
    MODE_CAPACITY,
    MODE_MAINTENANCE,
    MODE_SPEED,
    MODE_VEHICLE_COST,
    NODE_PROCESSOR,
    NODE_TOWN,
    TERRAIN_TOWN,
)
from tycoonle_jax.types import State
from tycoonle_jax.utils import first_true, hash01, one_hot, path_valid_mask, round_to


def step_state(state: State, action: jnp.ndarray) -> tuple[State, dict[str, jnp.ndarray]]:
    before_score = state.metrics[0]
    before_cargo = state.metrics[1]
    before_profit = state.metrics[2]
    before_cash = state.cash
    before_loan = state.loan
    action = action.astype(jnp.int32)
    in_range = (action >= 0) & (action < MAX_CANDIDATES)
    safe_action = jnp.clip(action, 0, MAX_CANDIDATES - 1)
    selected_kind = state.candidate.kind[safe_action]
    valid = in_range & state.action_mask[safe_action] & ~state.done
    effective_kind = jnp.where(valid, selected_kind, 0)

    def invalid_branch(s: State) -> State:
        return s._replace(invalid_actions=s.invalid_actions + 1)

    def build_branch(s: State) -> State:
        return _build_route(s, safe_action)

    def add_vehicle_branch(s: State) -> State:
        return _add_vehicle(s, safe_action)

    def wait_branch(s: State) -> State:
        return _wait_months(s, s.candidate.months[safe_action])

    def take_loan_branch(s: State) -> State:
        return _take_loan(s, s.candidate.amount[safe_action])

    def repay_loan_branch(s: State) -> State:
        return _repay_loan(s, s.candidate.amount[safe_action])

    branches = (invalid_branch, build_branch, add_vehicle_branch, wait_branch, take_loan_branch, repay_loan_branch)
    next_state = jax.lax.switch(effective_kind, branches, state)
    next_state = next_state._replace(step=next_state.step + 1)
    next_state = refresh_derived(next_state)
    terminated = next_state.metrics[0] >= 92.0
    truncated = (~terminated) & ((next_state.step >= next_state.max_steps) | (next_state.month >= next_state.max_months))
    next_state = next_state._replace(done=terminated | truncated)
    reward = round_to(next_state.metrics[0] - before_score, 4)
    extras = {
        "action_index": safe_action,
        "selected_kind": effective_kind,
        "valid_action": valid,
        "terminated": terminated,
        "truncated": truncated,
        "reward": reward,
        "score_delta": reward,
        "cargo_delta": round_to(next_state.metrics[1] - before_cargo, 3),
        "profit_delta": round_to(next_state.metrics[2] - before_profit, 3),
        "cash_delta": round_to(next_state.cash - before_cash, 3),
        "loan_delta": round_to(next_state.loan - before_loan, 3),
        "invalid_action": (~valid).astype(jnp.float32),
        "selected_source": state.candidate.source[safe_action],
        "selected_destination": state.candidate.destination[safe_action],
        "selected_cargo": state.candidate.cargo[safe_action],
        "selected_mode": state.candidate.mode[safe_action],
        "selected_route": state.candidate.route[safe_action],
        "selected_months": state.candidate.months[safe_action],
        "selected_amount": state.candidate.amount[safe_action],
        "selected_path": state.candidate.path[safe_action],
        "selected_path_length": state.candidate.path_length[safe_action],
        "selected_diagnostics": state.candidate.diagnostics[safe_action],
        "action_mask": next_state.action_mask,
        "metrics": next_state.metrics,
        "score_breakdown": next_state.score_breakdown,
    }
    return next_state, extras


def _build_route(state: State, action: jnp.ndarray) -> State:
    c = state.candidate
    route_idx = first_true(~state.route_mask)
    mode = c.mode[action]
    vehicles = c.vehicles[action].astype(jnp.float32)
    vehicle_cost = MODE_VEHICLE_COST[mode]
    path_length = c.path_length[action]
    distance = jnp.maximum(1.0, path_length.astype(jnp.float32) - 1.0)
    return state._replace(
        cash=state.cash - c.total_cost[action],
        route_mask=state.route_mask.at[route_idx].set(True),
        route_source=state.route_source.at[route_idx].set(c.source[action]),
        route_destination=state.route_destination.at[route_idx].set(c.destination[action]),
        route_cargo=state.route_cargo.at[route_idx].set(c.cargo[action]),
        route_mode=state.route_mode.at[route_idx].set(mode),
        route_path=state.route_path.at[route_idx].set(c.path[action]),
        route_path_length=state.route_path_length.at[route_idx].set(path_length),
        route_distance=state.route_distance.at[route_idx].set(distance),
        route_terrain_cost=state.route_terrain_cost.at[route_idx].set(c.terrain_cost[action]),
        route_build_cost=state.route_build_cost.at[route_idx].set(jnp.maximum(0.0, c.total_cost[action] - vehicle_cost * vehicles)),
        route_vehicle_cost=state.route_vehicle_cost.at[route_idx].set(vehicle_cost),
        route_vehicles=state.route_vehicles.at[route_idx].set(vehicles),
        route_travel_time_months=state.route_travel_time_months.at[route_idx].set(
            jnp.maximum(1, jnp.ceil(distance / jnp.where(mode == 1, 12.0, 9.0)).astype(jnp.int32))
        ),
        route_reliability=state.route_reliability.at[route_idx].set(1.0),
        route_station_rating=state.route_station_rating.at[route_idx].set(0.72),
        route_congestion=state.route_congestion.at[route_idx].set(c.congestion[action]),
    )


def _add_vehicle(state: State, action: jnp.ndarray) -> State:
    route_idx = state.candidate.route[action]
    cost = state.route_vehicle_cost[route_idx]
    return state._replace(
        cash=state.cash - cost,
        route_vehicles=state.route_vehicles.at[route_idx].add(1.0),
    )


def _take_loan(state: State, amount: jnp.ndarray) -> State:
    resolved = jnp.minimum(jnp.maximum(0.0, amount), jnp.maximum(0.0, state.max_loan - state.loan))
    return state._replace(cash=state.cash + resolved, loan=state.loan + resolved)


def _repay_loan(state: State, amount: jnp.ndarray) -> State:
    resolved = jnp.minimum(jnp.minimum(jnp.maximum(0.0, amount), state.loan), state.cash)
    return state._replace(cash=state.cash - resolved, loan=state.loan - resolved)


def _wait_months(state: State, months: jnp.ndarray) -> State:
    bounded = jnp.clip(months, 1, 6)
    return jax.lax.fori_loop(0, bounded, lambda _, s: _wait_one_month(s), state)


def _wait_one_month(state: State) -> State:
    def advance(s: State) -> State:
        month = s.month + 1
        s = s._replace(month=month)
        s = _replenish_nodes(s)
        s = _convert_processors(s)
        s = s._replace(cash=s.cash - (s.loan * s.interest_rate) / 12.0)
        return _operate_routes(s)

    return jax.lax.cond(state.month >= state.max_months, lambda s: s, advance, state)


def _replenish_nodes(state: State) -> State:
    cargos = jnp.arange(CARGO_VALUES.shape[0], dtype=jnp.int32)
    variance = 0.86 + hash01(state.seed, (state.node_x + state.node_y)[:, None], cargos[None, :], cargos[None, :] + 71) * 0.32
    processor_output = (state.node_kind[:, None] == NODE_PROCESSOR) & (state.node_convert_to[:, None] == cargos[None, :])
    active = state.node_mask[:, None] & (state.node_base_production > 0.0) & ~processor_output
    produced = state.node_base_production * state.node_production_index[:, None] * variance
    storage = jnp.where(active, jnp.minimum(state.node_storage + produced, state.node_base_production * 5.5), state.node_storage)
    idle = state.node_mask & (state.node_last_served_month < state.month - 8)
    production_index = jnp.where(idle, jnp.maximum(0.65, state.node_production_index * 0.992), state.node_production_index)
    return state._replace(node_storage=storage, node_production_index=round_to(production_index, 3))


def _convert_processors(state: State) -> State:
    node_idx = jnp.arange(MAX_NODES, dtype=jnp.int32)
    active = state.node_mask & (state.node_convert_from >= 0) & (state.node_convert_to >= 0)
    input_cargo = jnp.clip(state.node_convert_from, 0, CARGO_VALUES.shape[0] - 1)
    output_cargo = jnp.clip(state.node_convert_to, 0, CARGO_VALUES.shape[0] - 1)
    available = state.node_storage[node_idx, input_cargo]
    convert = jnp.where(active, jnp.minimum(available, 80.0 * state.node_production_index), 0.0)
    storage = state.node_storage.at[node_idx, input_cargo].add(-convert)
    storage = storage.at[node_idx, output_cargo].add(convert * 0.62)
    return state._replace(node_storage=storage)


def _operate_routes(state: State) -> State:
    active = state.route_mask
    route_idx = jnp.arange(MAX_ROUTES, dtype=jnp.int32)
    rem_dec = jnp.where(state.route_transit_remaining > 0, state.route_transit_remaining - 1, 0)
    arriving = active[:, None] & (state.route_transit_amount > 0.0) & (rem_dec <= 0)
    arrived_amount = jnp.sum(jnp.where(arriving, state.route_transit_amount, 0.0), axis=1)
    transit_amount = jnp.where(arriving, 0.0, state.route_transit_amount)
    transit_remaining = jnp.where(arriving, 0, rem_dec)
    source = state.route_source
    dest = state.route_destination
    cargo = state.route_cargo
    mode = state.route_mode
    available = state.node_storage[source, cargo]
    accepted = state.node_accepts[dest, cargo]
    capacity = state.route_vehicles * MODE_CAPACITY[mode] * MODE_SPEED[mode]
    town_tiles = _route_town_tiles(state)
    congestion = jnp.maximum(0.0, state.route_congestion + town_tiles * 0.012 + jnp.maximum(0.0, state.route_vehicles - 2.0) * 0.08)
    random_delay = hash01(state.seed, state.month, route_idx + 1, 601)
    breakdown_risk = jnp.minimum(0.22, 0.025 + congestion * 0.018 + jnp.maximum(0.0, 0.74 - state.route_reliability) * 0.09)
    delayed = active & (random_delay < breakdown_risk)
    delay_months = jnp.where(delayed, 1 + jnp.floor(hash01(state.seed, state.month, state.route_path_length, 602) * 2.0).astype(jnp.int32), 0)
    drag = jnp.maximum(0.25, 1.0 - state.route_distance / 155.0 - congestion * 0.025 - delay_months.astype(jnp.float32) * 0.08)
    station_rating = jnp.clip(state.route_station_rating + jnp.where(delayed, -0.06, 0.012), 0.35, 1.0)
    loaded = jnp.where(active, jnp.minimum(jnp.minimum(available, accepted * 1.6), capacity * drag * station_rating), 0.0)
    delivered = jnp.where(active, jnp.minimum(accepted, arrived_amount), 0.0)
    revenue = delivered * CARGO_VALUES[cargo] * (1.0 + state.route_distance / 120.0)
    operating_cost = jnp.where(
        active,
        state.route_vehicles * MODE_MAINTENANCE[mode] + state.route_distance * 7.5 * state.route_terrain_cost + congestion * 92.0 + delay_months.astype(jnp.float32) * 440.0,
        0.0,
    )
    profit = jnp.where(active, revenue - operating_cost, 0.0)

    free = transit_amount <= 0.0
    first_free = jnp.argmax(free.astype(jnp.int32), axis=1)
    free_exists = jnp.any(free, axis=1)
    append = active & (loaded > 0.0) & free_exists
    hot = jax.nn.one_hot(first_free, MAX_TRANSIT, dtype=jnp.float32) * append[:, None].astype(jnp.float32)
    transit_amount = transit_amount + hot * round_to(loaded, 3)[:, None]
    transit_remaining = transit_remaining + hot.astype(jnp.int32) * (state.route_travel_time_months + delay_months)[:, None]

    storage = state.node_storage.at[source, cargo].add(-loaded)
    storage = storage.at[dest, cargo].add(delivered)
    storage = jnp.maximum(0.0, storage)

    served = active & (delivered > 0.0)
    node_served = jnp.zeros((MAX_NODES,), dtype=jnp.bool_).at[source].set(served).at[dest].set(served)
    service_add = jnp.zeros((MAX_NODES,), dtype=jnp.int32).at[source].add(served.astype(jnp.int32)).at[dest].add(served.astype(jnp.int32))
    source_boost = jnp.zeros((MAX_NODES,), dtype=jnp.float32).at[source].add(delivered / 8500.0)
    dest_boost = jnp.zeros((MAX_NODES,), dtype=jnp.float32).at[dest].add(jnp.where(state.node_kind[dest] == NODE_TOWN, 0.0, delivered / 10_000.0))
    growth_factor = jnp.where(cargo == CARGO_PASSENGERS, 0.11, jnp.where((cargo == CARGO_GOODS) | (cargo == CARGO_FOOD), 0.07, 0.025))
    population_add = jnp.zeros((MAX_NODES,), dtype=jnp.float32).at[dest].add(jnp.where(state.node_kind[dest] == NODE_TOWN, delivered * growth_factor, 0.0))
    population = state.node_population + population_add
    accepts = state.node_accepts
    produces = state.node_produces
    town = state.node_kind == NODE_TOWN
    accepts = accepts.at[:, CARGO_PASSENGERS].set(jnp.where(town, jnp.round(population / 22.0), accepts[:, CARGO_PASSENGERS]))
    accepts = accepts.at[:, CARGO_MAIL].set(jnp.where(town, jnp.round(population / 65.0), accepts[:, CARGO_MAIL]))
    accepts = accepts.at[:, CARGO_GOODS].set(jnp.where(town, jnp.round(population / 55.0), accepts[:, CARGO_GOODS]))
    produces = produces.at[:, CARGO_PASSENGERS].set(jnp.where(town, jnp.round(population / 24.0), produces[:, CARGO_PASSENGERS]))
    produces = produces.at[:, CARGO_MAIL].set(jnp.where(town, jnp.round(population / 72.0), produces[:, CARGO_MAIL]))
    production_index = jnp.clip(state.node_production_index + source_boost + dest_boost, 0.0, 1.65)
    rating_add = jnp.zeros((MAX_NODES,), dtype=jnp.float32).at[dest].add(delivered / 5000.0)
    rating = jnp.clip(state.node_rating + rating_add, 0.0, 1.0)
    first_delivery = jnp.where((state.first_delivery_month < 0) & jnp.any(delivered > 0.0), state.month, state.first_delivery_month)

    return state._replace(
        cash=state.cash + jnp.sum(profit),
        node_storage=storage,
        node_accepts=accepts,
        node_produces=produces,
        node_population=population,
        node_production_index=round_to(production_index, 3),
        node_service_months=state.node_service_months + service_add,
        node_last_served_month=jnp.where(node_served, state.month, state.node_last_served_month),
        node_rating=round_to(rating, 3),
        first_delivery_month=first_delivery,
        route_transit_amount=transit_amount,
        route_transit_remaining=transit_remaining,
        route_delivered=state.route_delivered + delivered,
        route_revenue=state.route_revenue + revenue,
        route_operating_cost=state.route_operating_cost + operating_cost,
        route_profit=state.route_profit + profit,
        route_age_months=state.route_age_months + active.astype(jnp.int32),
        route_utilization=jnp.where(active, round_to(loaded / jnp.maximum(1.0, capacity), 3), state.route_utilization),
        route_last_delivered=round_to(delivered, 3),
        route_last_delay=delay_months,
        route_reliability=round_to(jnp.where(active, jnp.clip(state.route_reliability + jnp.where(delayed, -0.055, 0.008), 0.45, 1.0), state.route_reliability), 3),
        route_station_rating=round_to(station_rating, 3),
        route_congestion=round_to(congestion, 3),
    )


def _route_town_tiles(state: State) -> jnp.ndarray:
    px = state.route_path[:, :, 0]
    py = state.route_path[:, :, 1]
    terrain = state.terrain[py, px]
    valid = jnp.arange(MAX_PATH)[None, :] < state.route_path_length[:, None]
    return jnp.sum(jnp.where(valid & (terrain == TERRAIN_TOWN), 1.0, 0.0), axis=1)
