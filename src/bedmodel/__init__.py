"""Inpatient bed demand: Monte Carlo simulation of bed occupancy by ward type, on synthetic data."""

from .metrics import beds_for, coverage, summary
from .model import (BASELINE, ORIGINAL, RECOMMENDED, BedModel, Method, Scenario, data_interval, discharge_targets,
                    fit, simulate, total, true_occupancy)
from .synthetic import WARD_GROUPS, WARD_TYPES, Extract, Hospital, make_hospital, observe_year

__all__ = ["beds_for", "coverage", "summary", "BASELINE", "ORIGINAL", "RECOMMENDED", "BedModel", "Method", "Scenario",
           "data_interval", "discharge_targets", "fit", "simulate", "total", "true_occupancy", "WARD_GROUPS",
           "WARD_TYPES", "Extract", "Hospital", "make_hospital", "observe_year"]
