"""
Synthetic industrial sensor + maintenance data generator.

Simulates snapshot records for a fleet of machines. Failure risk is driven
by a physically-motivated (but synthetic) combination of vibration,
temperature, time-since-maintenance, equipment age, prior failure history,
oil quality and load — with noise added so the relationship isn't trivially
linear and a model actually has something to learn.
"""

import numpy as np
import pandas as pd

MACHINE_TYPES = ["Pump", "Compressor", "Motor", "Conveyor", "Turbine"]

FEATURE_COLUMNS = [
    "machine_type",
    "age_years",
    "operational_hours",
    "temperature_avg",
    "temperature_std",
    "vibration_avg",
    "vibration_std",
    "pressure_avg",
    "rotation_speed_avg",
    "humidity_avg",
    "load_pct",
    "oil_quality_index",
    "days_since_last_maintenance",
    "num_previous_failures",
]

TARGET_COLUMN = "failure_within_14_days"


def generate_dataset(n_machines: int = 400, seed: int = 42) -> pd.DataFrame:
    """Generate one snapshot row per machine with an engineered failure label."""
    rng = np.random.default_rng(seed)

    machine_type = rng.choice(MACHINE_TYPES, size=n_machines, p=[0.28, 0.22, 0.22, 0.18, 0.10])
    age_years = np.clip(rng.gamma(shape=2.2, scale=2.3, size=n_machines), 0.1, 25)
    operational_hours = age_years * rng.uniform(1500, 4200, size=n_machines)

    # Baseline sensor behavior with type-specific offsets
    type_temp_offset = pd.Series(machine_type).map(
        {"Pump": 2, "Compressor": 8, "Motor": 4, "Conveyor": -3, "Turbine": 10}
    ).values
    type_vibe_offset = pd.Series(machine_type).map(
        {"Pump": 0.3, "Compressor": 0.9, "Motor": 0.1, "Conveyor": 0.6, "Turbine": 1.2}
    ).values

    days_since_last_maintenance = np.clip(rng.exponential(scale=45, size=n_machines), 1, 400)
    num_previous_failures = rng.poisson(lam=np.clip(age_years / 6, 0.05, None), size=n_machines)
    oil_quality_index = np.clip(
        100 - days_since_last_maintenance * rng.uniform(0.05, 0.25, size=n_machines)
        - num_previous_failures * rng.uniform(1, 4, size=n_machines)
        + rng.normal(0, 5, size=n_machines),
        5, 100,
    )
    load_pct = np.clip(rng.normal(65, 18, size=n_machines), 5, 130)

    # Degradation-linked sensor drift: aging, high load, low oil quality push
    # temperature & vibration up.
    degradation = (
        0.9 * (age_years / 25)
        + 0.7 * (days_since_last_maintenance / 400)
        + 0.6 * ((100 - oil_quality_index) / 100)
        + 0.5 * np.clip((load_pct - 70) / 60, 0, None)
    )

    temperature_avg = np.clip(
        55 + type_temp_offset + degradation * 22 + rng.normal(0, 4, size=n_machines), 30, 140
    )
    temperature_std = np.clip(1.5 + degradation * 3 + rng.normal(0, 0.6, size=n_machines), 0.2, 12)
    vibration_avg = np.clip(
        1.0 + type_vibe_offset + degradation * 3.2 + rng.normal(0, 0.3, size=n_machines), 0.1, 9
    )
    vibration_std = np.clip(0.1 + degradation * 0.8 + rng.normal(0, 0.1, size=n_machines), 0.02, 2.5)
    pressure_avg = np.clip(rng.normal(100, 14, size=n_machines) - degradation * 8, 20, 180)
    rotation_speed_avg = np.clip(rng.normal(1500, 220, size=n_machines) - degradation * 90, 300, 3000)
    humidity_avg = np.clip(rng.normal(48, 14, size=n_machines), 5, 95)

    # Latent failure risk score combining everything (logistic form)
    z = (
        -6.2
        + 0.09 * (temperature_avg - 55)
        + 0.55 * (vibration_avg - 1.0)
        + 0.9 * vibration_std
        + 0.012 * days_since_last_maintenance
        + 0.55 * num_previous_failures
        + 0.10 * age_years
        - 0.035 * (oil_quality_index - 50)
        + 0.02 * np.clip(load_pct - 70, 0, None)
        + rng.normal(0, 0.9, size=n_machines)  # irreducible noise
    )
    prob_failure = 1 / (1 + np.exp(-z))
    failure_within_14_days = rng.binomial(1, prob_failure)

    df = pd.DataFrame({
        "machine_id": [f"M-{i:04d}" for i in range(1, n_machines + 1)],
        "machine_type": machine_type,
        "age_years": age_years.round(2),
        "operational_hours": operational_hours.round(0),
        "temperature_avg": temperature_avg.round(2),
        "temperature_std": temperature_std.round(2),
        "vibration_avg": vibration_avg.round(3),
        "vibration_std": vibration_std.round(3),
        "pressure_avg": pressure_avg.round(2),
        "rotation_speed_avg": rotation_speed_avg.round(1),
        "humidity_avg": humidity_avg.round(1),
        "load_pct": load_pct.round(1),
        "oil_quality_index": oil_quality_index.round(1),
        "days_since_last_maintenance": days_since_last_maintenance.round(0).astype(int),
        "num_previous_failures": num_previous_failures,
        TARGET_COLUMN: failure_within_14_days,
    })
    return df


if __name__ == "__main__":
    data = generate_dataset()
    data.to_csv("sample_sensor_data.csv", index=False)
    print(data.head())
    print("\nFailure rate:", data[TARGET_COLUMN].mean().round(3))
