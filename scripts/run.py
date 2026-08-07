"""Main entry point for HINet experiments.

Usage:
    # Run with default config
    python scripts/run.py

    # Override config values
    python scripts/run.py model.type=NetEst data.dataset=BC

    # Use a custom config file
    python scripts/run.py --config config/my_experiment.yaml

    # Run a W&B sweep (configure the sweep section in the YAML config first)
    python scripts/run.py --sweep
"""
import os
import sys
import argparse
import random
import time
import itertools

DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.dirname(DIR)
os.chdir(DIR)
sys.path.append(DIR)

import copy
import numpy as np
import torch
import wandb

import src.data.data_generator as data_generator
from src.config import load_config, to_flat_dict, build_setting_string
from src.trainers import create_trainer
from utils.utils import loadData

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


def _apply_dotted_overrides(base_config, dotted_overrides):
    """Return a deep-copied nested config with dotted-key overrides applied."""
    updated = copy.deepcopy(base_config)
    for key, value in dotted_overrides.items():
        keys = key.split(".")
        d = updated
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value
    return updated


def _extract_sweep_values(param_spec):
    """Extract a list of candidate values from a W&B parameter spec."""
    if isinstance(param_spec, dict):
        if "values" in param_spec:
            return list(param_spec["values"])
        if "value" in param_spec:
            return [param_spec["value"]]
    raise ValueError(
        "Unsupported sweep parameter spec. Expected {'values': [...]} or "
        "{'value': ...}."
    )


def _build_local_sweep_runs(sweep_section):
    """Build local run overrides from the sweep section.

    For grid sweeps, this returns the full cartesian product.
    For unsupported methods, this falls back to one run using the first value
    per parameter.
    """
    method = sweep_section.get("method", "grid")
    parameters = sweep_section.get("parameters", {})
    if not parameters:
        return [{}]

    keys = list(parameters.keys())
    values_per_key = [_extract_sweep_values(parameters[k]) for k in keys]

    if method == "grid":
        return [
            {k: v for k, v in zip(keys, combo)}
            for combo in itertools.product(*values_per_key)
        ]

    # Minimal fallback for non-grid methods during W&B outages.
    return [{k: values[0] for k, values in zip(keys, values_per_key)}]


def run_local_sweep(flat_config, nested_config, sweep_section, reason):
    """Run sweep combinations locally in offline mode if W&B sweep API fails."""
    print(
        f"W&B sweep unavailable ({type(reason).__name__}: {reason}). "
        "Falling back to local offline sweep execution.",
        flush=True,
    )

    runs = _build_local_sweep_runs(sweep_section)
    prev_mode = os.environ.get("WANDB_MODE")
    os.environ["WANDB_MODE"] = "offline"

    try:
        for idx, dotted_overrides in enumerate(runs, start=1):
            print(
                f"[Local sweep] Run {idx}/{len(runs)} overrides={dotted_overrides}",
                flush=True,
            )
            run_nested = _apply_dotted_overrides(nested_config, dotted_overrides)
            run_flat = to_flat_dict(run_nested)
            run_flat["wandb_project"] = flat_config["wandb_project"]
            run_experiment(run_flat, run_nested)
    finally:
        if prev_mode is None:
            os.environ.pop("WANDB_MODE", None)
        else:
            os.environ["WANDB_MODE"] = prev_mode


def run_experiment(flat_config, nested_config):
    """Run a single experiment with the given flat config dict."""
    wandb_start = time.time()
    print("Initializing W&B...", flush=True)
    try:
        run = wandb.init(
            config=flat_config,
            project=flat_config.get("wandb_project", "HINet"),
            job_type="sweep",
        )
    except Exception as exc:
        print(
            f"W&B init failed ({type(exc).__name__}: {exc}). "
            "Retrying in offline mode.",
            flush=True,
        )
        run = wandb.init(
            config=flat_config,
            project=flat_config.get("wandb_project", "HINet"),
            job_type="sweep",
            mode="offline",
        )

    run_mode = getattr(getattr(run, "settings", None), "mode", "unknown")
    print(
        f"W&B initialized in {time.time() - wandb_start:.2f}s "
        f"(run_id={wandb.run.id}, mode={run_mode})",
        flush=True,
    )
    try:
        # W&B sweep overrides use dotted keys (e.g. "model.type", "data.homophily").
        # Apply every dotted key back into a copy of the nested config, then re-flatten
        # so that to_flat_dict's renaming logic (e.g. model.type -> model_type) is
        # applied correctly for any sweep parameter without a hardcoded mapping.
        updated_nested = copy.deepcopy(nested_config)
        for key, value in dict(wandb.config).items():
            if "." in key:
                keys = key.split(".")
                d = updated_nested
                for k in keys[:-1]:
                    d = d.setdefault(k, {})
                d[keys[-1]] = value
        effective_flat = to_flat_dict(updated_nested)
        effective_flat["wandb_project"] = flat_config["wandb_project"]
        wandb.config.update(effective_flat, allow_val_change=True)
        config = wandb.config

        setting = build_setting_string(config)
        data_file = "data/simulated/" + setting + ".pkl"
        torch.manual_seed(config["seed"])
        np.random.seed(config["seed"])
        random.seed(config["seed"])

        wandb.log({"startup_marker": 1}, step=0)
        if wandb.run is not None:
            wandb.run.summary["setting"] = setting
            wandb.run.summary["data_file"] = data_file

        print(f"Dataset setting: {setting}", flush=True)
        print(f"Simulating data -> {data_file}", flush=True)
        sim_start = time.time()
        data_generator.simulate_data(config, setting=setting)
        sim_elapsed = time.time() - sim_start
        print(
            f"Data simulation done in {sim_elapsed:.2f}s "
            f"(file_exists={os.path.exists(data_file)})",
            flush=True,
        )
        wandb.log({"data_simulation_seconds": sim_elapsed}, step=0)

        print("Loading data...", flush=True)
        load_start = time.time()
        train_data, val_data, test_data = loadData(setting)
        print(f"Data loaded in {time.time() - load_start:.2f}s", flush=True)

        print(f"Creating trainer for model: {config['model_type']}", flush=True)
        trainer = create_trainer(
            config=dict(config),
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            device=True,
        )

        trainer.train_test_best_model(
            epochs_range=config["epochs_range"],
            hidden_range=config["hidden_range"],
            alpha_range=config["alpha_range"],
            lr_range=config["lr_range"],
            num_seeds=config["num_seeds"],
        )
    finally:
        wandb.finish()


def build_sweep_config(flat_config):
    """Build W&B sweep config from the flat config's sweep section."""
    return {
        "name": "sweep",
        "method": "grid",
        "parameters": flat_config.get("sweep_parameters", {}),
    }


def main():
    parser = argparse.ArgumentParser(description="HINet Experiment Runner")
    parser.add_argument("--config", default="config/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--sweep", action="store_true",
                        help="Run as W&B sweep")
    parser.add_argument("overrides", nargs="*",
                        help="Config overrides as dotted.key=value")
    args = parser.parse_args()

    nested_config = load_config(args.config, args.overrides)
    flat_config = to_flat_dict(nested_config)

    # Add wandb project name
    flat_config["wandb_project"] = nested_config.get("experiment", {}).get(
        "wandb_project", "HINet"
    )

    if args.sweep:
        sweep_section = nested_config.get("sweep", {})
        sweep_config = {
            "name": "sweep",
            "method": sweep_section.get("method", "grid"),
            "parameters": sweep_section.get("parameters", {}),
        }
        try:
            sweep_id = wandb.sweep(sweep_config, project=flat_config["wandb_project"])
            wandb.agent(sweep_id, function=lambda: run_experiment(flat_config, nested_config))
        except Exception as exc:
            run_local_sweep(flat_config, nested_config, sweep_section, reason=exc)
    else:
        run_experiment(flat_config, nested_config)


if __name__ == "__main__":
    main()
