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

# -------- Improved: helpers to generate many bodies with orbitalized starts --------
def random_initial_state(
    n: int,
    space_radius: float = 20.0,
    speed_sigma: float = 0.02,
    mass_range=(0.5, 1.5),
    seed=None,
    heavy_center: bool = True,
    center_mass_factor: float = 50.0,
    center_position=(0.0, 0.0, 0.0),
    min_radius: float = 1.0,
    profile: str = "sphere",        # "disc" or "sphere"
    equal_masses: bool = False,      # set True to make all bodies similar in mass
    velocity_noise: float = 0.05,   # noise as a fraction of circular speed
    orbitalize: bool = True,        # set tangential velocities for near-circular orbits
    disc_thickness: float = 0.2,    # stddev of z for disc profile
    zero_total_momentum: bool = True,
):
    """
    Generate N bodies with options for more interesting orbits.
    - If heavy_center=True: Body1 is a massive central body at center_position.
      Others are placed in a disc/sphere and given near-circular velocities.
    - If heavy_center=False: bodies are placed around the origin and velocities
      swirl around the center of mass.
    - equal_masses=True makes bodies closer in weight (often yields cleaner dynamics).
    """
    if seed is not None:
        np.random.seed(int(seed))

    # Masses
    if equal_masses:
        m_val = float((mass_range[0] + mass_range[1]) * 0.5)
        masses = np.full(n, m_val, dtype=float)
    else:
        masses = np.random.uniform(mass_range[0], mass_range[1], size=n).astype(float)

    # Positions
    positions = np.zeros((n, 3), dtype=float)
    center = np.array(center_position, dtype=float)

    def sample_positions(k):
        if profile == "disc":
            # Uniform in area: r from sqrt, theta uniform
            r = np.sqrt(np.random.uniform(low=max(min_radius**2, 1e-6), high=space_radius**2, size=k))
            theta = np.random.uniform(0, 2 * np.pi, size=k)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            z = np.random.normal(loc=0.0, scale=disc_thickness, size=k)
            return np.stack([x, y, z], axis=1)
        else:  # "sphere"
            # Uniform in volume
            u = np.random.uniform(low=max(min_radius**3, 1e-6), high=space_radius**3, size=k)
            r = np.cbrt(u)
            # Random directions on sphere
            v = np.random.normal(size=(k, 3))
            v /= np.linalg.norm(v, axis=1)[:, None]
            return r[:, None] * v

    # Initialize velocities
    velocities = np.zeros((n, 3), dtype=float)

    if heavy_center and n >= 1:
        # Central massive body
        center_mass = max(mass_range) * float(center_mass_factor)
        masses[0] = center_mass
        positions[0] = center
        velocities[0] = np.array([0.0, 0.0, 0.0], dtype=float)

        # Other bodies: positions
        if n > 1:
            positions[1:] = center + sample_positions(n - 1)

            if orbitalize:
                # Give others near-circular tangential velocities around the center
                for i in range(1, n):
                    r_vec = positions[i] - center
                    r = np.linalg.norm(r_vec)
                    if r < 1e-9:
                        continue
                    r_hat = r_vec / r
                    # Tangent direction: cross with a random vector
                    a = np.random.normal(size=3)
                    t = np.cross(r_hat, a)
                    t_norm = np.linalg.norm(t)
                    if t_norm < 1e-12:
                        # Fallback orthogonal
                        a = np.array([1.0, 0.0, 0.0])
                        t = np.cross(r_hat, a)
                        t_norm = np.linalg.norm(t)
                    t_hat = t / t_norm
                    v_circ = np.sqrt(center_mass / r)  # G=1
                    dir_sign = 1.0 if np.random.rand() < 0.5 else -1.0
                    v_vec = dir_sign * v_circ * t_hat
                    # Add small noise
                    v_vec += (velocity_noise * v_circ) * np.random.normal(size=3)
                    velocities[i] = v_vec
            else:
                # Fallback: small random velocities
                velocities[1:] = np.random.normal(loc=0.0, scale=speed_sigma, size=(n - 1, 3))

        # Keep central mass anchored (optional small kick could be added if desired)
        if zero_total_momentum and n > 1:
            # Only remove momentum from the non-central bodies so center stays fixed
            p_total = np.sum((masses[1:, None] * velocities[1:]), axis=0)
            m_rest = np.sum(masses[1:])
            if m_rest > 0:
                v_shift = p_total / m_rest
                velocities[1:] -= v_shift

    else:
        # No heavy center: distribute bodies around origin and swirl around center-of-mass
        positions[:] = sample_positions(n)
        com = np.average(positions, axis=0, weights=masses)
        # Orbitalize around COM
        if orbitalize:
            M_tot = float(np.sum(masses))
            for i in range(n):
                r_vec = positions[i] - com
                r = np.linalg.norm(r_vec)
                if r < 1e-9:
                    continue
                r_hat = r_vec / r
                a = np.random.normal(size=3)
                t = np.cross(r_hat, a)
                t_norm = np.linalg.norm(t)
                if t_norm < 1e-12:
                    a = np.array([1.0, 0.0, 0.0])
                    t = np.cross(r_hat, a)
                    t_norm = np.linalg.norm(t)
                t_hat = t / t_norm
                # Use total mass for a rough Keplerian-like swirl about COM
                v_circ = np.sqrt(M_tot / r)
                dir_sign = 1.0 if np.random.rand() < 0.5 else -1.0
                v_vec = dir_sign * v_circ * t_hat
                v_vec += (velocity_noise * v_circ) * np.random.normal(size=3)
                velocities[i] = v_vec
        else:
            velocities[:] = np.random.normal(loc=0.0, scale=speed_sigma, size=(n, 3))

        if zero_total_momentum:
            p_total = np.sum((masses[:, None] * velocities), axis=0)
            M_tot = np.sum(masses)
            velocities -= p_total / M_tot

    # Build init dict
    init = {}
    for i in range(1, n + 1):
        init[f"Body{i}"] = {
            "timeStep": 0.01,
            "time": 0.0,
            "position": {
                "x": float(positions[i - 1, 0]),
                "y": float(positions[i - 1, 1]),
                "z": float(positions[i - 1, 2]),
            },
            "velocity": {
                "x": float(velocities[i - 1, 0]),
                "y": float(velocities[i - 1, 1]),
                "z": float(velocities[i - 1, 2]),
            },
            "mass": float(masses[i - 1]),
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