"""
This file aims to simulate a realistic factory environment using SimPy. The scenario is a 5-station metal parts production line. It generates
production, downtime, and quality event logs with realistic variability, shift-effects, and data-quality issues. 

Scenario: A mid-size factory producing precision metal brackets.
Stations run 3 shifts/day over 90 days. The goal is to identify
bottlenecks, waste, and automation opportunities.
"""

import simpy
import numpy as np
import pandas as pd
import random
import argparse
import os
from datetime import datetime, timedelta

# SCENARIO CONFIGURATION
random.seed(42) 
SIM_DAYS = 90
SIM_START = datetime(2024, 1, 1, 6, 0) 

SHIFT_SCHEDULE = {1 : {"start_hour": 6, "label": "Day"},        # 6am to 2pm
                  2 : {"start_hour": 14, "label": "Swing"},     # 2pm to 10pm
                  3 : {"start_hour": 22, "label": "Night"}}     # 10pm to 6am

SHIFT_CYCLE_TIME_MULTIPLIER = {1: 1.0, 2: 1.08, 3: 1.15} 
SHIFT_SCRAP_MULTIPLIER = {1: 1.0, 2: 1.1, 3: 1.25}

# Operator pool per shift (assigned round-robin to manual stations).
OPERATORS_PER_SHIFT = {1: ["OP-101", "OP-102", "OP-103", "OP-104"],
                       2: ["OP-201", "OP-202", "OP-203"],
                       3: ["OP-301", "OP-302"]}

# ----- Station Definitions -----
# cycle_time:   (mean_minutes, std_minutes)
# mtbf_hours:   mean time between failures (exponential)
# mttr          (mean_minutes, std_minutes) for repair duration
# scrap_rate:   base scrap rate (percentage)
# num_machines: parallel capacity
# is_manual:    whether the station is manual (True) or automated (False)

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
            "breakdown_causes": ["tool_breakage", "spindle_failure", "coolant_system", "servo_error"]
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
            "breakdown_causes": ["electrode_wear", "gas_flow_issue", "power_supply_fault", "wire_feed_jam"],
        },
        {
            "name": "S3_Assembly",
            "cycle_time": (6.0, 1.5),
            "num_machines": 3,
            "mtbf_hours": 200,
            "mttr": (15, 5),
            "scrap_rate": 0.04,
            "is_manual": True,
            "defect_types": ["misalignment", "missing_fastener", "incorrect_torque", "cosmetic_scratch"],
            "breakdown_causes": ["pneumatic_tool_failure", "fixture_jam", "sensor_misread"],
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
            "breakdown_causes": ["calibration_drift", "camera_failure", "software_crash"],
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
            "breakdown_causes": ["conveyor_jam", "label_printer_fault", "seal_bar_failure"],
        },
]

# Data-Quality Issues
MISSING_DATA_RATE = 0.02      # 2 % of readings randomly missing
DUPLICATE_RECORD_RATE = 0.005 # 0.5 % duplicate rows
TIMESTAMP_JITTER_SEC = 30     # ±30 s random jitter on logged times


# ----- Helper Functions -----

def sim_minutes_to_datetime(sim_minutes: float) -> datetime:
    """Convert simulation minutes to a datetime object based on SIM_START."""
    return SIM_START + timedelta(minutes=sim_minutes)

def get_shift(sim_minutes: float) -> int:
    """Determine the current shift based on simulation minutes."""
    hour = sim_minutes_to_datetime(sim_minutes).hour
    if 6 <= hour < 14:
        return 1
    elif 14 <= hour < 22:
        return 2
    else:
        return 3

def get_operator(shift: int) -> str:
    """Select an operator for the given shift using round-robin assignment."""
    return random.choice(OPERATORS_PER_SHIFT[shift])


# ----- Station Model -----

class Station:
    """ A production station with machines, breakdowns, and quality variation """

    def __init__(self, env, config, downtime_log):
        self.env = env 
        self.name = config["name"]
        self.config = config
        self.machine = simpy.Resource(env, capacity=config["num_machines"])
        self.broken = False
        self.downtime_log = downtime_log

        # Start the independent breakdown process.
        # NOTE: When broken = True, ALL machines at this station are down.
        # This models shared-resource failures (power, conveyor, PLC).
        # For per-machine failures, you'd track each machine separately.
        self.env.process(self._failure_process())


    def _failure_process(self):
        """ Randomly generate breakdowns using exponential inter-arrival """
        while True:
            # Time until next failure (exponential distribution)
            ttf = random.expovariate(1.0 / self.config["mtbf_hours"]) * 60  # convert hours to minutes
            yield self.env.timeout(ttf)

            # Station goes down
            self.broken = True
            breakdown_cause = random.choice(self.config["breakdown_causes"])
            start = self.env.now

            # Repair duration (normal distribution)
            repair = max(5, random.gauss(*self.config["mttr"]))  # minimum 5 minutes
            yield self.env.timeout(repair)

            # Station repaired
            self.broken = False

            self.downtime_log.append({
                "station": self.name,
                "start_time": start,
                "end_time": self.env.now,
                "duration": round(self.env.now - start, 2),
                "cause": breakdown_cause
            })

    def get_cycle_time(self, shift: int) -> float:
        """Sample cycle time for this station, adjusted for shift effects."""
        ct = max(0.5, random.gauss(*self.config["cycle_time"]))  # minimum 0.5 minutes
        if self.config["is_manual"]:
            ct *= SHIFT_CYCLE_TIME_MULTIPLIER[shift]
        return ct

    def check_quality(self, shift: int) -> str | None:
        """ Return a defect type string, or None if part is good. """
        scrap_rate = self.config["scrap_rate"]
        if self.config["is_manual"]:
            scrap_rate *= SHIFT_SCRAP_MULTIPLIER[shift]
        if random.random() < scrap_rate and self.config["defect_types"]: 
            return random.choice(self.config["defect_types"])
        return None


# ----- Simulation Process -----
