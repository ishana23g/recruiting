# SIMULATOR

from functools import reduce
from operator import __or__
import subprocess
import json

# from modsim import agents
import modsim
from store import QRangeStore
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_query(query):
    # NOTE: The query parser is invoked via a subprocess call to the Rust binary
    popen = subprocess.Popen('../queries/target/release/sedaro-nano-queries', stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    (stdout, stderr) = popen.communicate(query)
    if popen.returncode:
        raise Exception(f"Parsing query failed: {stderr}")
    return json.loads(stdout)


# Place this function at the top-level of your simulator.py file (outside the class)
def run_agent_task(agentId, universe, sim_graph_agent, init_agent_keys):
    """
    A stateless worker function that simulates one agent for one step.
    It receives all necessary data and has no access to the Simulator instance.
    This is the key to achieving low-overhead parallelization.
    """
    newState = {}

    # --- Re-implementation of the find/put logic ---
    def _find(current_agent_id, query, universe_state, new_state_dict, prev=False):
        match query["kind"]:
            case "Base":
                if prev: return universe_state[current_agent_id][query["content"]]
                agentState = new_state_dict.get(current_agent_id)
                return agentState.get(query["content"]) if agentState else None
            case "Prev": return _find(current_agent_id, query["content"], universe_state, new_state_dict, prev=True)
            case "Root": return new_state_dict if not prev else universe_state[current_agent_id]
            case "Agent": return universe_state[query["content"]]
            case "Access":
                base = _find(current_agent_id, query["content"]["base"], universe_state, new_state_dict, prev)
                return base.get(query["content"]["field"]) if base else None
            case "Tuple":
                res = [_find(current_agent_id, q, universe_state, new_state_dict, prev) for q in query["content"]]
                return res if all(x is not None for x in res) else None
            case _: return None

    def _put(current_agent_id, query, universe_state, new_state_dict, data):
        match query["kind"]:
            case "Base":
                if current_agent_id not in new_state_dict:
                    new_state_dict[current_agent_id] = {}
                new_state_dict[current_agent_id][query["content"]] = data
            case "Access":
                baseQuery = query["content"]["base"]
                base = _find(current_agent_id, baseQuery, universe_state, new_state_dict)
                if base is None:
                    base = {}
                    _put(current_agent_id, baseQuery, universe_state, new_state_dict, base)
                base[query["content"]["field"]] = data
            case _:
                # Simplified for clarity, add other cases if needed
                raise Exception(f"Production for query kind {query['kind']} is not supported in worker.")

    # --- Re-implementation of the step/run_sm logic ---
    sms = [(agentId, sm) for sm in sim_graph_agent]
    while sms:
        next_sms = []
        progress_made = False
        for sm_agent_id, sm in sms:
            inputs = []
            # This logic is from run_sm
            for q in sm["consumed"]:
                found = _find(sm_agent_id, q, universe, newState)
                if found is None:
                    break
                inputs.append(found)
            
            # If all inputs were found, run the function
            if len(inputs) == len(sm["consumed"]):
                res = sm["func"](*inputs)
                _put(sm_agent_id, sm["produced"], universe, newState, res)
                progress_made = True
            else:
                next_sms.append((sm_agent_id, sm))
        
        if not progress_made and next_sms:
             raise Exception(f"No progress made for agent {agentId}")
        sms = next_sms

    return newState

class Simulator:
    """
    A Simulator is used to simulate the propagation of agents in the universe.
    This class is *not* pure. It mutates the data store in place and maintains internal state.

    It is given an initial state of the following form:
    ```
    {
        'agentId': {
            'time': <time of instantiation>,
            'timeStep': <time step for propagation>,
            **otherSimulatedProperties,
        },
        **otherAgents,
    }
    ```

    Args:
        store (QRangeStore): The data store in which to save the simulation results.
        init (dict): The initial state of the universe.
    """

    def __init__(self, store: QRangeStore, init: dict, agents_config: dict | None = None):
        # NOTE: Creating a Simulator object does all the simulation "building"
        self.store = store
        store[-999999999, 0] = init
        self.init = init
        self.times = {agentId: state["time"] for agentId, state in init.items()}
        self.sim_graph = {}
        agents_source = agents_config if agents_config is not None else modsim.agents
        for (agentId, sms) in agents_source.items():
            agent = []
            for sm in sms:
                consumed = parse_query(sm["consumed"])["content"]
                produced = parse_query(sm["produced"])
                func = sm["function"]
                agent.append({"func": func, "consumed": consumed, "produced": produced})
            self.sim_graph[agentId] = agent

    def read(self, t):
        try:
            data = self.store[t]
        except IndexError:
            data = []
        return reduce(__or__, data, {}) # combine all data into one dictionary

    def step(self, agentId, universe):
        """Run an Agent for a single step."""
        state = dict()
        sms = []
        for sm in self.sim_graph[agentId]:
            sms.append((agentId, sm))
        while sms:
            next_sms = []
            for (agentId, sm) in sms:
                if self.run_sm(agentId, sm, universe, state) is None:
                    next_sms.append((agentId, sm))
            if len(sms) == len(next_sms):
                raise Exception(f"No progress made while evaluating statemanagers for agent {agentId}. Remaining statemanagers: {[sm["func"].__name__ for (agentId, sm) in sms]}")
            sms = next_sms
        return state

    def run_sm(self, agentId, sm, universe, newState):
        """Run a State Manager for a single step."""
        inputs = []
        for q in sm["consumed"]:
            found = self.find(agentId, q, universe, newState)
            if found is None:
                return None
            inputs.append(found)
        res = sm["func"](*inputs)
        self.put(agentId, sm["produced"], universe, newState, res)
        return res

    def find(self, agentId, query, universe, newState: dict, prev=False):
        """Find consumed data to pass to a State Manager."""
        # NOTE: queries are interpreted at runtime here
        match query["kind"]:
            case "Base":
                if prev:
                    return universe[agentId][query["content"]]
                agentState = newState.get(agentId)
                if agentState is None:
                    return None
                return agentState.get(query["content"])
            case "Prev":
                return self.find(agentId, query["content"], universe, newState, prev=True)
            case "Root":
                if prev:
                    return universe[agentId]
                return newState
            case "Agent":
                # agent always gets the previous state
                return universe[query["content"]]
            case "Access":
                base = self.find(agentId, query["content"]["base"], universe, newState, prev)
                if base is None:
                    return None
                return base.get(query["content"]["field"])
            case "Tuple":
                res = []
                for q in query["content"]:
                    found = self.find(agentId, q, universe, newState, prev)
                    if found is None:
                        return None
                    res.append(found)
                return res
            case _:
                return None

    def put(self, agentId, query, universe, newState: dict, data):
        """Put produced data into the universe."""
        match query["kind"]:
            case "Base":
                agentState = newState.get(agentId)
                if agentState is None:
                    agentState = {}
                    newState[agentId] = agentState
                agentState[query["content"]] = data
            case "Prev":
                raise Exception(f"Cannot produce prev query {query}")
            case "Root":
                pass
            case "Agent":
                res = universe[query["content"]]
                if res is None:
                    res = {}
                    universe[query["content"]] = res
                return res
            case "Access":
                baseQuery = query["content"]["base"]
                base = self.find(agentId, baseQuery, universe, newState)
                if base is None:
                    base = {}
                    self.put(agentId, baseQuery, universe, newState, base)
                base[query["content"]["field"]] = data
            case "Tuple":
                raise Exception(f"Tuple production not yet implemented")

    # def simulate(self, iterations: int = 500):
    #     """Simulate the universe for a given number of iterations."""
    #     for _ in range(iterations):
    #         for agentId in self.init:
    #             t = self.times[agentId]
    #             universe = self.read(t - 0.001)

    #             if set(universe) == set(self.init):
    #                 newState = self.step(agentId, universe)
    #                 self.store[t, newState[agentId]["time"]] = newState
    #                 self.times[agentId] = newState[agentId]["time"]
                    
    #             # uni_state = set(universe) == set(self.init)
    #             # newState = self.step(agentId, universe) if uni_state else None
    #             # self.store[t, newState[agentId]["time"]] = newState if uni_state else None
    #             # self.times[agentId] = newState[agentId]["time"] if uni_state else t 

     # --- NEW PARALLEL SIMULATOR ---
    def simulate(self, iterations: int = 500):
        """Simulate the universe in parallel using a stateless worker function."""
        
        # Pre-cache keys for the worker
        init_agent_keys = set(self.init.keys())

        with ProcessPoolExecutor() as executor:
            for _ in range(iterations):
                
                # Step 1: Prepare and submit tasks
                # Map each future to the agentId and time `t` for later updates
                futures = {}
                for agentId in self.init:
                    t = self.times[agentId]
                    universe = self.read(t - 0.001)

                    if set(universe) == init_agent_keys:
                        # Submit the task to the worker with ONLY the data it needs
                        future = executor.submit(
                            run_agent_task,
                            agentId,
                            universe,
                            self.sim_graph[agentId],
                            init_agent_keys
                        )
                        futures[future] = (agentId, t)
                
                # Step 2: Collect results and update state sequentially
                for future in as_completed(futures):
                    newState = future.result()
                    if newState:
                        agentId, t = futures[future]
                        new_time = newState[agentId]["time"]
                        
                        # Update the shared store in the main process
                        self.store[t, new_time] = newState
                        self.times[agentId] = new_time
