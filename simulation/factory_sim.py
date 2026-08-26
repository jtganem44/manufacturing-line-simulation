"""
Manufacturing Line Simulation — Factory Automation Analysis
============================================================
Simulates a 5-station metal parts production line using SimPy.
Generates production, downtime, and quality event logs with
realistic variability, shift effects, and data-quality issues.

Scenario: A mid-size factory producing precision metal brackets.
Stations run 3 shifts/day over 90 days. The goal is to identify
bottlenecks, waste, and automation opportunities.

Usage:
    python simulation/factory_sim.py [--days 90] [--seed 42] [--output data/]
"""

import simpy
import numpy as np
import pandas as pd
import random
import argparse
import os
from datetime import datetime, timedelta


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION — Modify these for automation scenario comparison
# ═══════════════════════════════════════════════════════════════

RANDOM_SEED = 42
SIM_DAYS = 90
SIM_START = datetime(2024, 1, 1, 6, 0, 0)  # Monday 6 AM

# Parts arrive to the line on average every ARRIVAL_INTERVAL minutes.
# The line's constraint is Inspection at 3.5 min/part. Setting arrival
# at 3.3 min (~106 % of constraint capacity) models the post-Meridian
# demand surge that exceeds what the line can sustain.
ARRIVAL_INTERVAL_MEAN = 3.3

# WIP buffer capacities between adjacent stations (number of parts).
# Models physical floor space / pallet positions between stations.
# When a downstream buffer is full, the upstream machine blocks
# (cannot release its finished part), throttling the line naturally.
BUFFER_CAPACITIES = [30, 25, 20, 15]  # S1→S2, S2→S3, S3→S4, S4→S5

# Raw-material staging area before S1.  Limits how many incoming parts
# can queue at the head of the line.  When full, new arrivals wait
# (modelling deferred material deliveries / full staging racks).
RAW_MATERIAL_STAGING = 40

# Shift schedule and performance multipliers.
# Manual stations are slower on swing/night shifts (fatigue, staffing).
SHIFT_SCHEDULE = {
    1: {"start_hour": 6,  "label": "Day"},      # 6 AM – 2 PM
    2: {"start_hour": 14, "label": "Swing"},     # 2 PM – 10 PM
    3: {"start_hour": 22, "label": "Night"},     # 10 PM – 6 AM
}
SHIFT_CYCLE_TIME_MULTIPLIER = {1: 1.00, 2: 1.08, 3: 1.15}
SHIFT_SCRAP_MULTIPLIER = {1: 1.0, 2: 1.1, 3: 1.25}

# Operator pool per shift (assigned round-robin to manual stations).
OPERATORS_PER_SHIFT = {1: ["OP-101", "OP-102", "OP-103", "OP-104"],
                       2: ["OP-201", "OP-202", "OP-203"],
                       3: ["OP-301", "OP-302"]}

# ── Station definitions ───────────────────────────────────────
# cycle_time: (mean_minutes, std_minutes)
# mtbf_hours:  mean time between failures (exponential)
# mttr:        (mean_minutes, std_minutes) for repair duration
# scrap_rate:  base probability a part is scrapped at this station
# num_machines: parallel capacity
# is_manual:   True → affected by shift multipliers
#
# >> TO MODEL AUTOMATION: reduce std, lower scrap_rate,
#    increase mtbf, set is_manual=False, etc.

STATIONS = [
    {
        "name": "S1_CNC_Machining",
        "cycle_time": (4.5, 0.8),
        "num_machines": 2,
        "mtbf_hours": 60,
        "mttr": (45, 15),
        "scrap_rate": 0.03,
        "is_manual": False,
        "defect_types": ["dimensional_error", "surface_finish", "tool_wear_mark"],
        "breakdown_causes": ["tool_breakage", "spindle_failure",
                             "coolant_system", "servo_error"],
    },
    {
        "name": "S2_Welding",
        "cycle_time": (3.0, 0.5),
        "num_machines": 1,
        "mtbf_hours": 80,
        "mttr": (30, 10),
        "scrap_rate": 0.02,
        "is_manual": False,
        "defect_types": ["weak_weld", "porosity", "spatter_damage"],
        "breakdown_causes": ["electrode_wear", "gas_flow_issue",
                             "power_supply_fault", "wire_feed_jam"],
    },
    {
        "name": "S3_Assembly",
        "cycle_time": (6.0, 1.5),
        "num_machines": 3,
        "mtbf_hours": 200,
        "mttr": (15, 5),
        "scrap_rate": 0.04,
        "is_manual": True,
        "defect_types": ["misalignment", "missing_fastener",
                         "incorrect_torque", "cosmetic_scratch"],
        "breakdown_causes": ["pneumatic_tool_failure", "fixture_jam",
                             "sensor_misread"],
    },
    {
        "name": "S4_Inspection",
        "cycle_time": (3.5, 0.6),
        "num_machines": 1,
        "mtbf_hours": 150,
        "mttr": (20, 8),
        "scrap_rate": 0.0,   # Inspection detects defects; doesn't create them
        "is_manual": True,
        "defect_types": [],
        "breakdown_causes": ["calibration_drift", "camera_failure",
                             "software_crash"],
    },
    {
        "name": "S5_Packaging",
        "cycle_time": (2.0, 0.4),
        "num_machines": 1,
        "mtbf_hours": 120,
        "mttr": (25, 10),
        "scrap_rate": 0.01,
        "is_manual": True,
        "defect_types": ["label_error", "packaging_damage"],
        "breakdown_causes": ["conveyor_jam", "label_printer_fault",
                             "seal_bar_failure"],
    },
]

# ── Data-quality noise (simulates real MES/SCADA messiness) ───
MISSING_DATA_RATE = 0.02      # 2 % of readings randomly missing
DUPLICATE_RECORD_RATE = 0.005 # 0.5 % duplicate rows
TIMESTAMP_JITTER_SEC = 30     # ±30 s random jitter on logged times


# ═══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def sim_minutes_to_datetime(sim_minutes: float) -> datetime:
    """Convert simulation clock (minutes from 0) to a wall-clock datetime."""
    return SIM_START + timedelta(minutes=sim_minutes)


def get_shift(sim_minutes: float) -> int:
    """Return the current shift number (1, 2, or 3)."""
    hour = sim_minutes_to_datetime(sim_minutes).hour
    if 6 <= hour < 14:
        return 1
    elif 14 <= hour < 22:
        return 2
    else:
        return 3


def get_operator(shift: int) -> str:
    """Pick a random operator from the current shift's pool."""
    return random.choice(OPERATORS_PER_SHIFT[shift])


# ═══════════════════════════════════════════════════════════════
# STATION MODEL
# ═══════════════════════════════════════════════════════════════

class Station:
    """A production station with machines, breakdowns, and quality variation."""

    def __init__(self, env, config, downtime_log):
        self.env = env
        self.name = config["name"]
        self.config = config
        self.machine = simpy.Resource(env, capacity=config["num_machines"])
        self.broken = False
        self.downtime_log = downtime_log

        # Start the independent breakdown process.
        # NOTE: When broken=True, ALL machines at this station are down.
        # This models shared-resource failures (power, conveyor, PLC).
        # For per-machine failures, you'd track each machine separately.
        self.env.process(self._failure_process())

    def _failure_process(self):
        """Randomly generate breakdowns using exponential inter-arrival."""
        while True:
            # Time until next failure (exponential distribution)
            ttf = random.expovariate(1.0 / (self.config["mtbf_hours"] * 60))
            yield self.env.timeout(ttf)

            # — Station goes down —
            self.broken = True
            cause = random.choice(self.config["breakdown_causes"])
            start = self.env.now

            # Repair duration (normal, floored at 5 min)
            repair = max(5.0, random.gauss(*self.config["mttr"]))
            yield self.env.timeout(repair)

            # — Station comes back up —
            self.broken = False

            self.downtime_log.append({
                "station": self.name,
                "start_min": start,
                "end_min": self.env.now,
                "duration_min": round(self.env.now - start, 2),
                "cause": cause,
            })

    def get_cycle_time(self, shift: int) -> float:
        """Sample a cycle time, adjusted for shift if manual station."""
        ct = max(0.5, random.gauss(*self.config["cycle_time"]))
        if self.config["is_manual"]:
            ct *= SHIFT_CYCLE_TIME_MULTIPLIER[shift]
        return ct

    def check_quality(self, shift: int) -> str | None:
        """Return a defect type string, or None if part is good."""
        scrap_rate = self.config["scrap_rate"]
        if self.config["is_manual"]:
            scrap_rate *= SHIFT_SCRAP_MULTIPLIER[shift]
        if random.random() < scrap_rate and self.config["defect_types"]:
            return random.choice(self.config["defect_types"])
        return None


# ═══════════════════════════════════════════════════════════════
# SIMULATION PROCESSES
# ═══════════════════════════════════════════════════════════════

def part_flow(env, part_id, stations, buffers, raw_staging,
              production_log, quality_log):
    """Move one part sequentially through every station.

    Buffer logic (blocking model):
      After processing at station i, the part tries to enter the
      downstream buffer (buffers[i]).  If that buffer is full the part
      stays on the machine — blocking it from starting the next part —
      until a slot opens.  When the part finishes at station i+1 it
      returns the slot to buffers[i], freeing space for upstream.

      Station 0 (CNC) also frees a raw-material staging slot when its
      part moves on, allowing a new arrival to enter the line.

    Buffers are SimPy Containers initialised to their capacity.
      get(1) = consume a slot  (part enters the inter-station WIP area)
      put(1) = release a slot  (part leaves that WIP area)
    """
    status = "good"

    for i, station in enumerate(stations):
        # Request a machine at this station
        with station.machine.request() as req:
            queue_enter = env.now
            yield req
            queue_wait = env.now - queue_enter

            # If station is broken, wait for repair (poll every minute)
            downtime_wait_start = env.now
            while station.broken:
                yield env.timeout(1)
            downtime_wait = env.now - downtime_wait_start

            # Process the part
            shift = get_shift(env.now)
            operator = get_operator(shift) if station.config["is_manual"] else None
            cycle_time = station.get_cycle_time(shift)
            proc_start = env.now
            yield env.timeout(cycle_time)
            proc_end = env.now

            # Quality check
            defect = station.check_quality(shift)
            if defect:
                status = "scrapped"
                quality_log.append({
                    "part_id": f"P-{part_id:06d}",
                    "station": station.name,
                    "defect_type": defect,
                    "detection_time_min": env.now,
                    "shift": shift,
                    "operator": operator,
                })

            # ── Blocking: wait for downstream buffer space ────────
            blocking_time = 0.0
            if i < len(buffers) and status == "good":
                block_start = env.now
                yield buffers[i].get(1)
                blocking_time = env.now - block_start

        # ── Machine released (with-block exited) ─────────────────

        # Free upstream buffer slot (part has moved on from that area)
        if i > 0:
            yield buffers[i - 1].put(1)
        elif i == 0:
            # Part leaves the raw-material staging area
            yield raw_staging.put(1)

        # Log production event
        production_log.append({
            "part_id": f"P-{part_id:06d}",
            "station": station.name,
            "shift": shift,
            "operator": operator,
            "queue_wait_min": round(queue_wait, 2),
            "downtime_wait_min": round(downtime_wait, 2),
            "cycle_time_min": round(cycle_time, 2),
            "blocking_time_min": round(blocking_time, 2),
            "start_min": round(proc_start, 2),
            "end_min": round(proc_end, 2),
            "status": status,
        })

        # If scrapped, free upstream buffer and exit
        if status == "scrapped":
            return

    # Part completed all stations
    return


def part_generator(env, stations, buffers, raw_staging,
                   production_log, quality_log):
    """Generate new parts arriving at the line."""
    part_id = 0
    while True:
        part_id += 1

        # Wait for a raw-material staging slot before entering the line
        yield raw_staging.get(1)

        env.process(
            part_flow(env, part_id, stations, buffers, raw_staging,
                      production_log, quality_log)
        )

        # Exponential inter-arrival (models variable raw-material feed)
        interval = random.expovariate(1.0 / ARRIVAL_INTERVAL_MEAN)
        yield env.timeout(interval)


# ═══════════════════════════════════════════════════════════════
# DATA EXPORT & NOISE INJECTION
# ═══════════════════════════════════════════════════════════════

def inject_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Add realistic data-quality issues to a DataFrame."""
    df = df.copy()
    n = len(df)
    rng = np.random.default_rng(RANDOM_SEED + 99)

    # 1. Random missing values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        mask = rng.random(n) < MISSING_DATA_RATE
        df.loc[mask, col] = np.nan

    # 2. Duplicate some rows
    n_dupes = int(n * DUPLICATE_RECORD_RATE)
    if n_dupes > 0:
        dupe_idx = rng.choice(n, size=n_dupes, replace=False)
        dupes = df.iloc[dupe_idx].copy()
        df = pd.concat([df, dupes], ignore_index=True)

    # 3. Timestamp jitter (add to time columns)
    time_cols = [c for c in df.columns if c.endswith("_min")]
    for col in time_cols:
        valid = df[col].notna()
        jitter = rng.uniform(
            -TIMESTAMP_JITTER_SEC / 60, TIMESTAMP_JITTER_SEC / 60, size=valid.sum()
        )
        df.loc[valid, col] = df.loc[valid, col] + jitter

    # 4. Shuffle to destroy perfect ordering (like a real data dump)
    df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    return df


def add_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert absolute sim-minute columns to human-readable datetimes.

    Only converts columns that represent points in time (start, end,
    detection), NOT duration columns (queue_wait, downtime_wait, cycle_time).
    """
    df = df.copy()
    # Only convert absolute-time columns, not durations
    absolute_time_cols = [c for c in df.columns
                          if c.endswith("_min")
                          and any(k in c for k in ("start", "end", "detection"))]
    for col in absolute_time_cols:
        dt_col = col.replace("_min", "_timestamp")
        df[dt_col] = df[col].apply(
            lambda x: sim_minutes_to_datetime(x).strftime("%Y-%m-%d %H:%M:%S")
            if pd.notna(x) else None
        )
    return df


def export_data(production_log, downtime_log, quality_log, output_dir):
    """Convert logs to DataFrames, inject noise, and save CSVs."""
    os.makedirs(output_dir, exist_ok=True)

    # Production log
    df_prod = pd.DataFrame(production_log)
    df_prod = add_datetime_columns(df_prod)
    df_prod = inject_noise(df_prod)
    df_prod.to_csv(os.path.join(output_dir, "production_log.csv"), index=False)

    # Downtime events
    df_down = pd.DataFrame(downtime_log)
    df_down = add_datetime_columns(df_down)
    df_down = inject_noise(df_down)
    df_down.to_csv(os.path.join(output_dir, "downtime_events.csv"), index=False)

    # Quality events
    df_qual = pd.DataFrame(quality_log)
    df_qual = add_datetime_columns(df_qual)
    df_qual = inject_noise(df_qual)
    df_qual.to_csv(os.path.join(output_dir, "quality_events.csv"), index=False)

    return df_prod, df_down, df_qual


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def run_simulation(sim_days=SIM_DAYS, seed=RANDOM_SEED, output_dir="data"):
    """Run the full factory simulation and export data."""
    random.seed(seed)
    np.random.seed(seed)

    sim_duration = sim_days * 24 * 60  # total minutes
    if sim_duration <= 0:
        raise ValueError(f"sim_days must be positive (got {sim_days})")

    # Data collectors
    production_log = []
    downtime_log = []
    quality_log = []

    # Build the SimPy environment
    env = simpy.Environment()

    # Create stations
    stations = [Station(env, cfg, downtime_log) for cfg in STATIONS]

    # Create inter-station WIP buffers (Container initialised to capacity =
    # all slots free).  get(1) consumes a slot; put(1) releases one.
    buffers = [
        simpy.Container(env, capacity=cap, init=cap)
        for cap in BUFFER_CAPACITIES
    ]

    # Raw-material staging area before the first station
    raw_staging = simpy.Container(
        env, capacity=RAW_MATERIAL_STAGING, init=RAW_MATERIAL_STAGING
    )

    # Start generating parts
    env.process(
        part_generator(env, stations, buffers, raw_staging,
                       production_log, quality_log)
    )

    # Run
    print(f"Running simulation: {sim_days} days, seed={seed}")
    print(f"  Stations: {[s.name for s in stations]}")
    env.run(until=sim_duration)

    # ── Summary stats (computed from CLEAN logs, before noise) ──
    part_ids_all = {r["part_id"] for r in production_log}
    part_ids_completed = {r["part_id"] for r in production_log
                          if r["station"] == "S5_Packaging" and r["status"] == "good"}
    part_ids_scrapped = {r["part_id"] for r in production_log
                         if r["status"] == "scrapped"}
    total_downtime_hrs = (sum(r["duration_min"] for r in downtime_log) / 60
                          if downtime_log else 0)

    total_parts = len(part_ids_all)
    completed = len(part_ids_completed)
    scrapped = len(part_ids_scrapped)
    yield_pct = (completed / total_parts * 100) if total_parts > 0 else 0.0

    print(f"\n{'='*50}")
    print(f"  SIMULATION SUMMARY")
    print(f"{'='*50}")
    print(f"  Simulation period : {sim_days} days")
    print(f"  Parts entered line: {total_parts}")
    print(f"  Parts completed   : {completed}")
    print(f"  Parts scrapped    : {scrapped}")
    print(f"  Yield             : {yield_pct:.1f}%")
    print(f"  Total downtime    : {total_downtime_hrs:.1f} hours")
    print(f"  Downtime events   : {len(downtime_log)}")
    print(f"  Quality events    : {len(quality_log)}")
    print(f"{'='*50}")

    # ── Export data (noise injected into CSVs only) ───────────
    print(f"\nExporting data to '{output_dir}/'...")
    df_prod, df_down, df_qual = export_data(
        production_log, downtime_log, quality_log, output_dir
    )

    print(f"\n  Files written:")
    print(f"    {output_dir}/production_log.csv   ({len(df_prod):,} rows)")
    print(f"    {output_dir}/downtime_events.csv  ({len(df_down):,} rows)")
    print(f"    {output_dir}/quality_events.csv   ({len(df_qual):,} rows)")

    return df_prod, df_down, df_qual


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Factory Line Simulation")
    parser.add_argument("--days", type=int, default=SIM_DAYS, help="Number of days to simulate")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument("--output", type=str, default="data", help="Output directory for CSV files")
    args = parser.parse_args()

    run_simulation(sim_days=args.days, seed=args.seed, output_dir=args.output)