# MODELING & SIMULATION

from random import random

import numpy as np

# New: N-body version (vectorized over "others")
def propagate_velocity_multi(time_step, position, velocity, others):
    """
    others: list of [position_dict, mass_float]
    """
    r_self = np.array([position['x'], position['y'], position['z']], dtype=float)
    v_self = np.array([velocity['x'], velocity['y'], velocity['z']], dtype=float)

    if not others:
        # No other bodies; velocity unchanged
        return {'x': v_self[0], 'y': v_self[1], 'z': v_self[2]}

    # Build arrays for positions and masses
    r_others = np.array([[p['x'], p['y'], p['z']] for (p, m) in others], dtype=float)
    m_others = np.array([m for (p, m) in others], dtype=float)

    # Vectorized gravity sum: dv/dt = -sum_i m_i (r_self - r_i) / ||r_self - r_i||^3
    dr = r_self[None, :] - r_others  # shape (N,3)
    dist = np.linalg.norm(dr, axis=1)  # (N,)
    eps = 1e-9  # avoid division by zero
    inv_r3 = 1.0 / np.maximum(dist, eps)**3  # (N,)
    dvdt = -np.sum((m_others * inv_r3)[:, None] * dr, axis=0)  # (3,)

    v_self = v_self + dvdt * float(time_step)
    return {'x': float(v_self[0]), 'y': float(v_self[1]), 'z': float(v_self[2])}

def propagate_position(time_step, position, velocity):
    """Propagate the position of the agent from `time` to `time + timeStep`."""
    # Apply velocity to position
    r_self = np.array([position['x'], position['y'], position['z']])
    v_self = np.array([velocity['x'], velocity['y'], velocity['z']])

    r_self = r_self + v_self * time_step

    return {'x': r_self[0], 'y': r_self[1], 'z': r_self[2]}

# -------- New: helpers to generate many bodies dynamically --------

def random_initial_state(n: int, space_radius: float = 100.0, speed_sigma: float = 0.05,
                         mass_range=(0.1, 1.0), seed=None):
    """
    Generate N randomized bodies with positions ~ N(0, space_radius) and velocities ~ N(0, speed_sigma).
    """
    if seed is not None:
        np.random.seed(seed)
    init = {}
    masses = np.random.uniform(mass_range[0], mass_range[1], size=n)
    positions = np.random.normal(loc=0.0, scale=space_radius, size=(n, 3))
    velocities = np.random.normal(loc=0.0, scale=speed_sigma, size=(n, 3))

    for i in range(1, n + 1):
        init[f'Body{i}'] = {
            'timeStep': 0.01,
            'time': 0.0,
            'position': {'x': float(positions[i-1, 0]), 'y': float(positions[i-1, 1]), 'z': float(positions[i-1, 2])},
            'velocity': {'x': float(velocities[i-1, 0]), 'y': float(velocities[i-1, 1]), 'z': float(velocities[i-1, 2])},
            'mass': float(masses[i-1]),
        }
    return init

def build_n_body_agents(n: int):
    """
    Programmatically build an agent graph for N bodies where each velocity update
    sums the gravitational effect from all other bodies via a nested tuple.
    NOTE: Requires n >= 2.
    """
    if n < 2:
        raise ValueError("build_n_body_agents requires n >= 2")
    cfg = {}
    for i in range(1, n + 1):
        others = []
        for j in range(1, n + 1):
            if j == i:
                continue
            others.append(f"(agent!(Body{j}).position, agent!(Body{j}).mass)")
        # Inner tuple of others with a trailing comma (grammar requires CommaPlus)
        others_block = ",\n                ".join(others) + ","

        consumed_velocity_block = f'''(
                prev!(timeStep),
                prev!(position),
                prev!(velocity),
                (
                    {others_block}
                ),
            )'''

        cfg[f'Body{i}'] = [
            {
                'consumed': consumed_velocity_block,
                'produced': '''velocity''',
                'function': propagate_velocity_multi,
            },
            {
                'consumed': '''(
                    prev!(timeStep),
                    prev!(position),
                    velocity,
                )''',
                'produced': '''position''',
                'function': propagate_position,
            },
            {
                'consumed': '''(
                    prev!(mass),
                )''',
                'produced': '''mass''',
                'function': propagate_mass,
            },
            {
                'consumed': '''(
                    prev!(time),
                    timeStep
                )''',
                'produced': '''time''',
                'function': time_manager,
            },
            {
                'consumed': '''(
                    velocity,
                )''',
                'produced': '''timeStep''',
                'function': timestep_manager,
            }
        ]
    return cfg

def propagate_velocity(time_step, position, velocity, other_position, m_other):
    """Propagate the velocity of the agent from `time` to `time + timeStep`."""
    # Use law of gravitation to update velocity
    r_self = np.array([position['x'], position['y'], position['z']])
    v_self = np.array([velocity['x'], velocity['y'], velocity['z']])
    r_other = np.array([other_position['x'], other_position['y'], other_position['z']])

    r = r_self - r_other
    dvdt = -m_other * r / np.linalg.norm(r)**3
    v_self = v_self + dvdt * time_step

    return {'x': v_self[0], 'y': v_self[1], 'z': v_self[2]}

def propagate_position(time_step, position, velocity):
    """Propagate the position of the agent from `time` to `time + timeStep`."""
    # Apply velocity to position
    r_self = np.array([position['x'], position['y'], position['z']])
    v_self = np.array([velocity['x'], velocity['y'], velocity['z']])

    r_self = r_self + v_self * time_step

    return {'x': r_self[0], 'y': r_self[1], 'z': r_self[2]}

def propagate_mass(mass):
    return mass

def identity(arg):
    return arg

def timestep_manager(velocity):
    """Compute the length of the next simulation timeStep for the agent"""
    return 100

def time_manager(time, timeStep):
    """Compute the time for the next simulation step for the agent"""
    return time + timeStep

'''
NOTE: Declare what agents should exist, what functions should be run to update their state, 
    and bind the consumed arguments and produced results to each other.

Query syntax:
- `<variableName>` will do a dictionary lookup of `variableName` in the current state of the agent
   the query is running for.
- prev!(<query>)` will get the value of `query` from the previous step of simulation.
- `agent!(<agentId>)` will get the most recent state produced by `agentId`.
- `<query>.<name>` will evaluate `query` and then look up `name` in the resulting dictionary.
'''

agents = {
    'Body1': [
        {
            'consumed': '''(
                prev!(timeStep),
                prev!(position),
                prev!(velocity),
                agent!(Body2).position,
                agent!(Body2).mass,
            )''',
            'produced': '''velocity''',
            'function': propagate_velocity,
        },
        {
            'consumed': '''(
                prev!(timeStep),
                prev!(position),
                velocity,
            )''',
            'produced': '''position''',
            'function': propagate_position,
        },
        {
            'consumed': '''(
                prev!(mass),
            )''',
            'produced': '''mass''',
            'function': propagate_mass,
        },
        {
            'consumed': '''(
                prev!(time),
                timeStep
            )''',
            'produced': '''time''',
            'function': time_manager,
        },
        {
            'consumed': '''(
                velocity,
            )''',
            'produced': '''timeStep''',
            'function': timestep_manager,
        }
    ],
    'Body2': [
        {
            'consumed': '''(
                prev!(timeStep),
                prev!(position),
                prev!(velocity),
                agent!(Body1).position,
                agent!(Body1).mass,
            )''',
            'produced': '''velocity''',
            'function': propagate_velocity,
        },
        {
            'consumed': '''(
                prev!(timeStep),
                prev!(position),
                velocity,
            )''',
            'produced': '''position''',
            'function': propagate_position,
        },
        {
            'consumed': '''(
                prev!(mass),
            )''',
            'produced': '''mass''',
            'function': propagate_mass,
        },
        {
            'consumed': '''(
                prev!(time),
                timeStep
            )''',
            'produced': '''time''',
            'function': time_manager,
        },
        {
            'consumed': '''(
                velocity,
            )''',
            'produced': '''timeStep''',
            'function': timestep_manager,
        }
    ]
}

# NOTE: initial values are set here. we intentionally separate the data from the functions operating on it.
data = {
    'Body1': {
        'timeStep': 0.01,
        'time': 0.0,
        'position': {'x': -0.73, 'y': 0, 'z': 0},
        'velocity': {'x': 0, 'y': -0.0015, 'z': 0},
        'mass': 1
    },
    'Body2': {
        'timeStep': 0.01,
        'time': 0.0,
        'position': {'x': 60.34, 'y': 0, 'z': 0},
        'velocity': {'x': 0, 'y': 0.13 , 'z': 0},
        'mass': 0.123
    }
}