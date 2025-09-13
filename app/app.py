# HTTP SERVER

import json

from flask import Flask, request
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from simulator import Simulator
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from store import QRangeStore
import logging
from datetime import datetime
from modsim import build_n_body_agents, random_initial_state
from flask import jsonify

class Base(DeclarativeBase):
    pass


############################## Application Configuration ##############################

app = Flask(__name__)
CORS(app, origins=["http://localhost:3030"])

db = SQLAlchemy(model_class=Base)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///database.db"
db.init_app(app)

logging.basicConfig(level=logging.INFO)

############################## Database Models ##############################


class Simulation(db.Model):
    id: Mapped[int] = mapped_column(primary_key=True)
    data: Mapped[str]


with app.app_context():
    db.create_all()


############################## API Endpoints ##############################

@app.get("/debug/routes")
def debug_routes():
    return jsonify({rule.rule: sorted(list(rule.methods)) for rule in app.url_map.iter_rules()})

@app.get("/")
def health():
    return "<p>Sedaro Nano API - running!</p>"

def _parse_bool(val) -> bool:
    if val is None:
        return False
    return str(val).lower() in ("1", "true", "yes", "on")

@app.get("/simulation")
def get_data():
    # Only return the most recent simulation
    simulation: Simulation = Simulation.query.order_by(Simulation.id.desc()).first()
    try:
        return jsonify(json.loads(simulation.data)) if simulation else jsonify([])
    except Exception:
        return jsonify(simulation.data if simulation else [])

# New: run a random N-body sim, then return results
@app.get("/simulation/random")
def simulate_random():
    try:
        n = int(request.args.get("n", 50))
        iterations = int(request.args.get("iterations", 500))
        seed_param = request.args.get("seed")
        seed = int(seed_param) if seed_param not in (None, "", "None") else None
        workers_raw = request.args.get("workers")
        num_workers = None if workers_raw in (None, "", "auto", "Auto", "None") else int(workers_raw)
        heavy_center = _parse_bool(request.args.get("heavy_center"))

        init = random_initial_state(
            n=n,
            seed=seed,
            heavy_center=heavy_center
        )
        agents_cfg = build_n_body_agents(n)

        t = datetime.now()
        store = QRangeStore()
        simulator = Simulator(store=store, init=init, agents_config=agents_cfg)
        logging.info(f"[GET /simulation/random] Build: {datetime.now() - t}")

        t = datetime.now()
        if num_workers == 1:
            logging.info("[random] single-threaded")
            simulator.simulate(iterations=iterations)
        else:
            logging.info("[random] parallel")
            simulator.simulate_parallel(iterations=iterations, num_workers=num_workers)
        logging.info(f"[GET /simulation/random] Sim: {datetime.now() - t}")

        simulation = Simulation(data=json.dumps(store.store))
        db.session.add(simulation)
        db.session.commit()

        return jsonify(store.store)
    except Exception as e:
        logging.exception("Failed random simulation via GET /simulation/random")
        return jsonify({"error": str(e)}), 400

@app.post("/simulation")
def simulate():
    # Define time and timeStep for each agent
    init: dict = request.json
    for key in init.keys():
        init[key]["time"] = 0
        init[key]["timeStep"] = 0.01

    # Create store and simulator
    t = datetime.now()
    store = QRangeStore()
    simulator = Simulator(store=store, init=init)
    logging.info(f"Time to Build: {datetime.now() - t}")

    # Run simulation
    t = datetime.now()
    workers_raw = request.args.get("workers")
    num_workers = None if workers_raw in (None, "", "auto", "Auto", "None") else int(workers_raw)
    iterations_raw = request.args.get("iterations")
    iterations = int(iterations_raw) if iterations_raw not in (None, "", "None") else 500

    if num_workers == 1:
        logging.info("[regular] single-threaded")
        simulator.simulate(iterations=iterations)
    else:
        logging.info("[regular] parallel")
        simulator.simulate_parallel(iterations=iterations, num_workers=num_workers)
    logging.info(f"Time to Simulate: {datetime.now() - t}")

    # Save data to database
    simulation = Simulation(data=json.dumps(store.store))
    db.session.add(simulation)
    db.session.commit()

    return jsonify(store.store)