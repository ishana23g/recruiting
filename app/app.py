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


@app.get("/simulation")
def get_data():
    # Optional: kick off a random N-body sim when query params are provided
    if request.args.get("n"):
        try:
            n = int(request.args.get("n", 50))
            iterations = int(request.args.get("iterations", 500))
            space_radius = float(request.args.get("space_radius", 100.0))
            speed_sigma = float(request.args.get("speed_sigma", 0.05))
            mass_min = float(request.args.get("mass_min", 0.1))
            mass_max = float(request.args.get("mass_max", 1.0))
            seed_param = request.args.get("seed")
            seed = int(seed_param) if seed_param not in (None, "", "None") else None

            init = random_initial_state(
                n=n,
                space_radius=space_radius,
                speed_sigma=speed_sigma,
                mass_range=(mass_min, mass_max),
                seed=seed,
            )
            agents_cfg = build_n_body_agents(n)

            t = datetime.now()
            store = QRangeStore()
            simulator = Simulator(store=store, init=init, agents_config=agents_cfg)
            logging.info(f"[random via GET /simulation] Build: {datetime.now() - t}")

            t = datetime.now()
            simulator.simulate(iterations=iterations)
            logging.info(f"[random via GET /simulation] Sim: {datetime.now() - t}")

            simulation = Simulation(data=json.dumps(store.store))
            db.session.add(simulation)
            db.session.commit()

            return store.store
        except Exception as e:
            logging.exception("Failed random simulation via GET /simulation")
            return {"error": str(e)}, 400

    # Get most recent simulation from database
    simulation: Simulation = Simulation.query.order_by(Simulation.id.desc()).first()
    return simulation.data if simulation else []


@app.post("/simulation")
def simulate():
    # Get data from request in this form
    # init = {
    #     "Body1": {"x": 0, "y": 0.1, "vx": 0.1, "vy": 0},
    #     "Body2": {"x": 0, "y": 1, "vx": 1, "vy": 0},
    # }

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
    simulator.simulate()
    logging.info(f"Time to Simulate: {datetime.now() - t}")

    # Save data to database
    simulation = Simulation(data=json.dumps(store.store))
    db.session.add(simulation)
    db.session.commit()

    return store.store


@app.route("/simulation/random", methods=["GET", "POST"])
def simulate_random():
    logging.info("simulate_random: received %s with args=%s json=%s", request.method, dict(request.args), request.json)
    try:
        payload = request.json or request.args or {}
        n = int(payload.get("n", 50))
        iterations = int(payload.get("iterations", 500))
        space_radius = float(payload.get("space_radius", 100.0))
        speed_sigma = float(payload.get("speed_sigma", 0.05))
        mass_min = float(payload.get("mass_min", 0.1))
        mass_max = float(payload.get("mass_max", 1.0))
        seed_raw = payload.get("seed")
        seed = int(seed_raw) if seed_raw not in (None, "", "None") else None

        init = random_initial_state(
            n=n,
            space_radius=space_radius,
            speed_sigma=speed_sigma,
            mass_range=(mass_min, mass_max),
            seed=seed,
        )
        agents_cfg = build_n_body_agents(n)

        t = datetime.now()
        store = QRangeStore()
        simulator = Simulator(store=store, init=init, agents_config=agents_cfg)
        logging.info(f"[random] Time to Build: {datetime.now() - t}")

        t = datetime.now()
        simulator.simulate(iterations=iterations)
        logging.info(f"[random] Time to Simulate: {datetime.now() - t}")

        simulation = Simulation(data=json.dumps(store.store))
        db.session.add(simulation)
        db.session.commit()

        return store.store
    
    except Exception as e:
        logging.exception("simulate_random failed")
        return jsonify({"error": str(e)}), 400