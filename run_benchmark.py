#!/usr/bin/env python3
"""
HARBench Full Benchmark Runner

Evaluates all 5 metrics:
1. Average Performance - 18 datasets with multi-sensor
2. Domain Robustness - Daily/Exercise/Industry averages
3. Position Robustness - Single-sensor evaluation across 8 position categories
4. Few-shot Performance - 1%, 2%, 5%, 10%, 20%, 50% data
5. Zero-shot Performance - LODO on DSADS/MHEALTH/PAMAP2

Supports 13 models:
  ResNet-based: resnet, mtl, harnet, simclr, moco, timechannel, timemask, cpc
  Transformer-based: selfpab, limubert, imumae
  Foundation Models: patchtst, moment

Usage:
    python run_benchmark.py --model mtl
    python run_benchmark.py --model mtl --num_gpus 4
    python run_benchmark.py --model mtl --num_gpus 2 --parallel 2
"""

import argparse
import json
import os
import subprocess
import sys
import multiprocessing
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Force spawn method for CUDA compatibility
multiprocessing.set_start_method('spawn', force=True)


# =============================================================================
# Model Configuration
# =============================================================================

# 13 supported models
MODELS = [
    # ResNet-based (SSL-Wearables)
    "resnet", "mtl", "harnet", "simclr", "moco",
    "timechannel", "timemask", "cpc",
    # Transformer-based
    "selfpab", "limubert", "imumae",
    # Foundation Models
    "patchtst", "moment",
]


# =============================================================================
# Dataset Configuration (from paper Table 2)
# =============================================================================

# 17 labeled datasets with ALL available sensors (from processed_strict format)
# Using all sensors for Average Performance evaluation
DATASETS = {
    # Daily (9 datasets)
    "daily": {
        "forthtrace": {"sensors": ["LeftWrist", "RightWrist", "Torso", "RightThigh", "LeftAnkle"], "domain": "daily"},
        "harth": {"sensors": ["LowerBack", "RightThigh"], "domain": "daily"},
        "imwsha": {"sensors": ["Wrist", "Chest", "Thigh"], "domain": "daily"},
        "paal": {"sensors": ["Phone"], "domain": "daily"},
        "pamap2": {"sensors": ["hand", "chest", "ankle"], "domain": "daily"},
        "realworld": {"sensors": ["Chest", "Forearm", "Head", "Shin", "Thigh", "UpperArm", "Waist"], "domain": "daily"},
        "selfback": {"sensors": ["Wrist", "Thigh"], "domain": "daily"},
        "ucaehar": {"sensors": ["SmartGlasses"], "domain": "daily"},
        "uschad": {"sensors": ["Phone"], "domain": "daily"},
        "ward": {"sensors": ["LeftArm", "RightArm", "Waist", "LeftAnkle", "RightAnkle"], "domain": "daily"},
    },
    # Exercise (4 datasets)
    "exercise": {
        "dsads": {"sensors": ["Torso", "RightArm", "LeftArm", "RightLeg", "LeftLeg"], "domain": "exercise"},
        "mex": {"sensors": ["Wrist", "Thigh"], "domain": "exercise"},
        "mhealth": {"sensors": ["Chest", "LeftAnkle", "RightWrist"], "domain": "exercise"},
        "realdisp": {"sensors": ["RightLowerArm", "RightUpperArm", "Back", "LeftUpperArm", "LeftLowerArm", "RightCalf", "RightThigh", "LeftThigh", "LeftCalf"], "domain": "exercise"},
    },
    # Industry (4 datasets)
    "industry": {
        "lara": {"sensors": ["LeftArm", "LeftLeg", "Neck", "RightArm", "RightLeg"], "domain": "industry"},
        "openpack": {"sensors": ["RightWrist", "LeftWrist", "RightUpperArm", "LeftUpperArm"], "domain": "industry"},
        "exoskeletons": {"sensors": ["Chest", "RightLeg", "LeftLeg", "RightWrist", "LeftWrist"], "domain": "industry"},
        "vtt_coniot": {"sensors": ["Hip", "Back", "UpperArm"], "domain": "industry"},
    },
}

# Position categories for single-sensor evaluation
# Total: 66 dataset-position configurations across 8 categories
# (matches total sensors in 18 datasets)
#
# Sensor name mapping (paper -> processed_strict):
# - HARTH back -> LowerBack, thigh -> RightThigh
# - IMWSHA chest/thigh/wrist -> Chest/Thigh/Wrist
# - LARa Torsor -> Neck
# - PAAL Phone -> Phone
# - SELFBACK watch -> Wrist
# - UCAEHAR Glasses -> SmartGlasses
# - USCHAD smartphone -> Phone
# - WARD FrontCenter -> Waist, LowerLeftForearm -> LeftArm, LowerRightForearm -> RightArm
#
# Corrections from paper:
# - MHEALTH RightLowerArm -> RightWrist (actual sensor is wrist, moved to Wrist category)
# - REALDISP LeftCalf/RightCalf -> Leg (calf is part of leg, not arm)
POSITION_CONFIGS = {
    # Arm: 15 configs
    "Arm": [
        ("dsads", "LeftArm"), ("dsads", "RightArm"),
        ("lara", "LeftArm"), ("lara", "RightArm"),
        ("openpack", "LeftUpperArm"), ("openpack", "RightUpperArm"),
        ("realdisp", "LeftLowerArm"), ("realdisp", "LeftUpperArm"),
        ("realdisp", "RightLowerArm"), ("realdisp", "RightUpperArm"),
        ("realworld", "Forearm"), ("realworld", "UpperArm"),
        ("vtt_coniot", "UpperArm"),
        ("ward", "LeftArm"), ("ward", "RightArm"),
    ],
    # Leg: 17 configs
    "Leg": [
        ("dsads", "LeftLeg"), ("dsads", "RightLeg"),
        ("exoskeletons", "LeftLeg"), ("exoskeletons", "RightLeg"),
        ("forthtrace", "RightThigh"),
        ("harth", "RightThigh"),
        ("imwsha", "Thigh"),
        ("lara", "LeftLeg"), ("lara", "RightLeg"),
        ("mex", "Thigh"),
        ("realdisp", "LeftCalf"), ("realdisp", "LeftThigh"),
        ("realdisp", "RightCalf"), ("realdisp", "RightThigh"),
        ("realworld", "Shin"), ("realworld", "Thigh"),
        ("selfback", "Thigh"),
    ],
    # Front: 11 configs
    "Front": [
        ("dsads", "Torso"),
        ("exoskeletons", "Chest"),
        ("forthtrace", "Torso"),
        ("imwsha", "Chest"),
        ("lara", "Neck"),
        ("mhealth", "Chest"),
        ("pamap2", "chest"),
        ("realworld", "Chest"), ("realworld", "Waist"),
        ("vtt_coniot", "Hip"),
        ("ward", "Waist"),
    ],
    # Ankle: 5 configs
    "Ankle": [
        ("forthtrace", "LeftAnkle"),
        ("mhealth", "LeftAnkle"),
        ("pamap2", "ankle"),
        ("ward", "LeftAnkle"), ("ward", "RightAnkle"),
    ],
    # Wrist: 11 configs
    "Wrist": [
        ("exoskeletons", "LeftWrist"), ("exoskeletons", "RightWrist"),
        ("forthtrace", "LeftWrist"), ("forthtrace", "RightWrist"),
        ("imwsha", "Wrist"),
        ("mex", "Wrist"),
        ("mhealth", "RightWrist"),
        ("openpack", "LeftWrist"), ("openpack", "RightWrist"),
        ("pamap2", "hand"),
        ("selfback", "Wrist"),
    ],
    # Phone: 2 configs
    "Phone": [
        ("paal", "Phone"),
        ("uschad", "Phone"),
    ],
    # Back: 3 configs
    "Back": [
        ("harth", "LowerBack"),
        ("realdisp", "Back"),
        ("vtt_coniot", "Back"),
    ],
    # Head: 2 configs
    "Head": [
        ("realworld", "Head"),
        ("ucaehar", "SmartGlasses"),
    ],
}

# Few-shot ratios
FEWSHOT_RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]


def get_all_datasets():
    """Get all datasets as flat dict."""
    all_ds = {}
    for domain_datasets in DATASETS.values():
        all_ds.update(domain_datasets)
    return all_ds


# =============================================================================
# Job Execution
# =============================================================================

def run_single_job(job):
    """Run a single finetune job. Returns (job_id, result)."""
    job_id = job["id"]
    dataset = job["dataset"]
    sensors = job["sensors"]
    model = job["model"]
    gpu_id = job["gpu_id"]  # GPU index (0, 1, 2, ...)
    output_dir = job["output_dir"]
    weights = job.get("weights")
    data_ratio = job.get("data_ratio", 1.0)
    epochs = job.get("epochs", 100)
    patience = job.get("patience", 10)
    batch_size = job.get("batch_size", 64)
    max_samples_per_epoch = job.get("max_samples_per_epoch")
    cmd = [
        sys.executable, "finetune.py",
        "--model", model,
        "--dataset", dataset,
        "--sensors", *sensors,
        "--device", "cuda:0",  # Always use cuda:0 within isolated GPU context
        "--output_dir", output_dir,
        "--epochs", str(epochs),
        "--patience", str(patience),
        "--batch_size", str(batch_size),
    ]

    if max_samples_per_epoch is not None:
        cmd.extend(["--max_samples_per_epoch", str(max_samples_per_epoch)])
    if weights:
        cmd.extend(["--weights", weights])
    if data_ratio < 1.0:
        cmd.extend(["--data_ratio", str(data_ratio)])

    # Set CUDA_VISIBLE_DEVICES to isolate GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, env=env)

        # Find latest results.json (finetune.py adds "finetune/" subdirectory)
        finetune_dir = os.path.join(output_dir, "finetune")
        if os.path.exists(finetune_dir):
            for run_dir in sorted(os.listdir(finetune_dir), reverse=True):
                results_path = os.path.join(finetune_dir, run_dir, "results.json")
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        return (job_id, json.load(f))
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr[-500:] if e.stderr else "Unknown error"
        return (job_id, {"error": error_msg, "stdout": e.stdout[-500:] if e.stdout else ""})

    return (job_id, None)


def run_jobs_parallel(jobs, num_gpus, parallel_per_gpu):
    """Run jobs in parallel across multiple GPUs with dynamic GPU assignment."""
    from queue import Queue
    from concurrent.futures import ThreadPoolExecutor
    import threading

    total_workers = num_gpus * parallel_per_gpu

    # Create GPU slot queue: each GPU has parallel_per_gpu slots
    gpu_queue = Queue()
    for gpu_id in range(num_gpus):
        for _ in range(parallel_per_gpu):
            gpu_queue.put(gpu_id)

    results = {}
    completed = [0]  # Use list for mutable counter in closure
    total = len(jobs)
    results_lock = threading.Lock()

    print(f"\nRunning {total} jobs with {total_workers} workers ({num_gpus} GPUs x {parallel_per_gpu} parallel)")

    def run_job_with_gpu(job):
        """Get GPU from queue, run job, return GPU to queue."""
        gpu_id = gpu_queue.get()  # Block until GPU available
        try:
            job["gpu_id"] = gpu_id
            return run_single_job(job)
        finally:
            gpu_queue.put(gpu_id)  # Return GPU to queue

    # Use ThreadPoolExecutor (subprocess handles actual parallelism)
    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        futures = {executor.submit(run_job_with_gpu, job): job for job in jobs}

        for future in as_completed(futures):
            job = futures[future]
            job_id, result = future.result()

            with results_lock:
                results[job_id] = result
                completed[0] += 1

                # Progress update
                if result and "error" not in result and "summary" in result:
                    f1 = result["summary"]["mean_f1"]
                    print(f"  [{completed[0]}/{total}] {job_id}: F1={f1:.4f}")
                elif result and "error" in result:
                    print(f"  [{completed[0]}/{total}] {job_id}: Error - {result['error'][:100]}")
                else:
                    print(f"  [{completed[0]}/{total}] {job_id}: Failed")

    return results


# =============================================================================
# Job Generation
# =============================================================================

def generate_average_jobs(args):
    """Generate jobs for Average Performance evaluation."""
    all_datasets = get_all_datasets()
    jobs = []

    # Filter by --datasets if specified
    if args.datasets:
        filter_set = set(d.lower() for d in args.datasets)
        all_datasets = {k: v for k, v in all_datasets.items() if k.lower() in filter_set}

    for dataset, config in all_datasets.items():
        jobs.append({
            "id": f"average/{dataset}",
            "dataset": dataset,
            "sensors": config["sensors"],
            "model": args.model,
            "output_dir": os.path.join(args.output_dir, "average", dataset),
            "weights": args.weights,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "max_samples_per_epoch": args.max_samples_per_epoch,
            "domain": config["domain"],
        })

    return jobs


def generate_position_jobs(args):
    """Generate jobs for Position Robustness evaluation."""
    jobs = []

    # Filter by --datasets if specified
    filter_set = set(d.lower() for d in args.datasets) if args.datasets else None

    for category, configs in POSITION_CONFIGS.items():
        for dataset, sensor in configs:
            if filter_set and dataset.lower() not in filter_set:
                continue
            jobs.append({
                "id": f"position/{category}/{dataset}_{sensor}",
                "dataset": dataset,
                "sensors": [sensor],
                "model": args.model,
                "output_dir": os.path.join(args.output_dir, "position", f"{dataset}_{sensor}"),
                "weights": args.weights,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "max_samples_per_epoch": args.max_samples_per_epoch,
                "category": category,
            })

    return jobs


def generate_fewshot_jobs(args):
    """Generate jobs for Few-shot Performance evaluation."""
    all_datasets = get_all_datasets()
    jobs = []

    # Filter by --datasets if specified
    if args.datasets:
        filter_set = set(d.lower() for d in args.datasets)
        all_datasets = {k: v for k, v in all_datasets.items() if k.lower() in filter_set}

    for ratio in FEWSHOT_RATIOS:
        for dataset, config in all_datasets.items():
            jobs.append({
                "id": f"fewshot/{ratio}/{dataset}",
                "dataset": dataset,
                "sensors": config["sensors"],
                "model": args.model,
                "output_dir": os.path.join(args.output_dir, "fewshot", f"{dataset}_ratio{ratio}"),
                "weights": args.weights,
                "data_ratio": ratio,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "max_samples_per_epoch": args.max_samples_per_epoch,
                "ratio": ratio,
            })

    return jobs


# =============================================================================
# Result Aggregation
# =============================================================================

def aggregate_average_results(jobs, results):
    """Aggregate results for Average Performance."""
    print("\n" + "="*60)
    print("1. Average Performance (18 datasets)")
    print("="*60)

    dataset_results = {}
    for job in jobs:
        job_id = job["id"]
        dataset = job["dataset"]
        domain = job["domain"]
        result = results.get(job_id)

        if result and "summary" in result:
            f1 = result["summary"]["mean_f1"]
            dataset_results[dataset] = {"f1": f1, "domain": domain}

    if dataset_results:
        avg_f1 = np.mean([r["f1"] for r in dataset_results.values()])
        print(f"\n  Average Performance: {avg_f1:.4f} ({len(dataset_results)}/18 datasets)")
        return {"datasets": dataset_results, "average_f1": avg_f1}
    return {}


def aggregate_domain_results(average_results):
    """Compute Domain Robustness from average performance results."""
    dataset_results = average_results.get("datasets", {})
    if not dataset_results:
        return {}

    print("\n" + "="*60)
    print("2. Domain Robustness")
    print("="*60)

    domain_f1s = defaultdict(list)
    for dataset, r in dataset_results.items():
        domain_f1s[r["domain"]].append(r["f1"])

    domain_results = {}
    for domain, f1s in domain_f1s.items():
        mean_f1 = np.mean(f1s)
        domain_results[domain] = {"mean_f1": mean_f1, "n_datasets": len(f1s)}
        print(f"  {domain}: F1={mean_f1:.4f} ({len(f1s)} datasets)")

    robustness = np.mean([r["mean_f1"] for r in domain_results.values()])
    print(f"\n  Domain Robustness: {robustness:.4f}")

    return {"domains": domain_results, "domain_robustness": robustness}


def aggregate_position_results(jobs, results):
    """Aggregate results for Position Robustness."""
    print("\n" + "="*60)
    print("3. Position Robustness (single-sensor)")
    print("="*60)

    category_f1s = defaultdict(list)
    config_results = {}

    for job in jobs:
        job_id = job["id"]
        category = job["category"]
        result = results.get(job_id)

        if result and "summary" in result:
            f1 = result["summary"]["mean_f1"]
            category_f1s[category].append(f1)
            config_key = f"{job['dataset']}/{job['sensors'][0]}"
            config_results[config_key] = f1

    category_results = {}
    for category, f1s in category_f1s.items():
        mean_f1 = np.mean(f1s)
        category_results[category] = {"mean_f1": mean_f1, "n_configs": len(f1s)}
        print(f"  {category}: F1={mean_f1:.4f} ({len(f1s)} configs)")

    if category_results:
        robustness = np.mean([r["mean_f1"] for r in category_results.values()])
        print(f"\n  Position Robustness: {robustness:.4f}")
        return {
            "categories": category_results,
            "configs": config_results,
            "position_robustness": robustness
        }
    return {}


def aggregate_fewshot_results(jobs, results):
    """Aggregate results for Few-shot Performance."""
    print("\n" + "="*60)
    print("4. Few-shot Performance")
    print("="*60)

    ratio_f1s = defaultdict(list)

    for job in jobs:
        job_id = job["id"]
        ratio = job["ratio"]
        result = results.get(job_id)

        if result and "summary" in result:
            ratio_f1s[ratio].append(result["summary"]["mean_f1"])

    summary = {}
    for ratio in FEWSHOT_RATIOS:
        f1s = ratio_f1s.get(ratio, [])
        if f1s:
            mean_f1 = np.mean(f1s)
            summary[f"{ratio*100:.0f}%"] = mean_f1
            print(f"  {ratio*100:.0f}%: F1={mean_f1:.4f} ({len(f1s)} datasets)")

    if summary:
        avg = np.mean(list(summary.values()))
        print(f"\n  Few-shot Average: {avg:.4f}")
        return {"ratios": summary, "fewshot_average": avg}
    return {}


# =============================================================================
# Zero-shot (runs separately - not parallelized)
# =============================================================================

def run_zeroshot(model, device, output_dir, weights=None, epochs=100, max_samples_per_epoch=None):
    """Run zero-shot evaluation."""
    cmd = [
        sys.executable, "finetune.py",
        "--model", model,
        "--zeroshot",
        "--device", device,
        "--output_dir", output_dir,
        "--epochs", str(epochs),
    ]

    if max_samples_per_epoch is not None:
        cmd.extend(["--max_samples_per_epoch", str(max_samples_per_epoch)])
    if weights:
        cmd.extend(["--weights", weights])

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

        zeroshot_dir = os.path.join(output_dir, "zeroshot")
        if os.path.exists(zeroshot_dir):
            for run_dir in sorted(os.listdir(zeroshot_dir), reverse=True):
                results_path = os.path.join(zeroshot_dir, run_dir, "results.json")
                if os.path.exists(results_path):
                    with open(results_path) as f:
                        return json.load(f)
        else:
            print(f"  Warning: zeroshot_dir not found: {zeroshot_dir}")
    except subprocess.CalledProcessError as e:
        print(f"  Error: {e.stderr[:500] if e.stderr else 'Unknown'}")

    return None


def eval_zeroshot_performance(args):
    """Evaluate Zero-shot Performance (LODO)."""
    print("\n" + "="*60)
    print("5. Zero-shot Performance (LODO)")
    print("="*60)

    result = run_zeroshot(args.model, "cuda:0", args.output_dir, weights=args.weights,
                          epochs=args.epochs, max_samples_per_epoch=args.max_samples_per_epoch)

    if result and "dataset_results" in result:
        for dataset, r in result["dataset_results"].items():
            print(f"  {dataset}: F1={r['f1']:.4f}")
        print(f"\n  Zero-shot Average: {result['summary']['mean_f1']:.4f}")
        return result

    print("  Failed")
    return {}


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="HARBench Full Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported models (13 total):

  ResNet-based (SSL-Wearables):
    resnet       - Random init baseline
    mtl          - Multi-Task Learning pretrained
    harnet       - HARNet (OxWearables official)
    simclr       - SimCLR pretrained
    moco         - MoCo pretrained
    timechannel  - Masked Resnet (time+channel)
    timemask     - Masked Resnet (time only)
    cpc          - Contrastive Predictive Coding

  Transformer-based:
    selfpab      - SelfPAB (STFT + Transformer)
    limubert     - LIMU-BERT
    imumae       - IMU-Video-MAE (ECCV 2024)

  Foundation Models:
    patchtst     - PatchTST (requires transformers)
    moment       - MOMENT (requires momentfm)

Examples:
  python run_benchmark.py --model mtl
  python run_benchmark.py --model mtl --num_gpus 4
  python run_benchmark.py --model mtl --num_gpus 2 --parallel 2
"""
    )
    parser.add_argument("--model", type=str, default="resnet", choices=MODELS,
                        help="Model to use (default: resnet)")
    parser.add_argument("--weights", type=str, default=None,
                        help="Override pretrained weights path (optional)")
    parser.add_argument("--num_gpus", type=int, default=4,
                        help="Number of GPUs to use (default: 4)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Number of parallel jobs per GPU (default: 1)")
    parser.add_argument("--output_dir", type=str, default="results/benchmark", help="Output dir")
    parser.add_argument("--epochs", type=int, default=100, help="Max epochs")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--max_samples_per_epoch", type=int, default=3200,
                        help="Max samples per epoch (default: 3200 = batch_size * 25)")
    parser.add_argument("--eval", type=str, default="all",
                        choices=["all", "average", "domain", "position", "fewshot", "zeroshot"],
                        help="Which evaluation to run")
    parser.add_argument("--datasets", type=str, nargs="+", default=None,
                        help="Specific datasets to evaluate (e.g., --datasets paal uschad)")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join(args.output_dir, f"{timestamp}_{args.model}_{args.eval}")
    os.makedirs(args.output_dir, exist_ok=True)

    total_workers = args.num_gpus * args.parallel

    print("="*60)
    print("HARBench Full Benchmark")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Weights: {args.weights or '(model default)'}")
    print(f"GPUs: {args.num_gpus}, Parallel per GPU: {args.parallel} (Total workers: {total_workers})")
    print(f"Output: {args.output_dir}")
    print(f"Evaluation: {args.eval}")

    all_results = {}

    # Average Performance
    if args.eval in ["all", "average", "domain"]:
        jobs = generate_average_jobs(args)
        results = run_jobs_parallel(jobs, args.num_gpus, args.parallel)
        all_results["average"] = aggregate_average_results(jobs, results)

    # Domain Robustness (computed from average)
    if args.eval in ["all", "domain"]:
        if "average" in all_results:
            all_results["domain"] = aggregate_domain_results(all_results["average"])

    # Position Robustness
    if args.eval in ["all", "position"]:
        jobs = generate_position_jobs(args)
        results = run_jobs_parallel(jobs, args.num_gpus, args.parallel)
        all_results["position"] = aggregate_position_results(jobs, results)

    # Few-shot Performance
    if args.eval in ["all", "fewshot"]:
        jobs = generate_fewshot_jobs(args)
        results = run_jobs_parallel(jobs, args.num_gpus, args.parallel)
        all_results["fewshot"] = aggregate_fewshot_results(jobs, results)

    # Zero-shot Performance
    if args.eval in ["all", "zeroshot"]:
        all_results["zeroshot"] = eval_zeroshot_performance(args)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    summary = {}
    if "average" in all_results and all_results["average"]:
        summary["average_performance"] = all_results["average"].get("average_f1", 0)
        print(f"  1. Average Performance: {summary['average_performance']:.4f}")

    if "domain" in all_results and all_results["domain"]:
        summary["domain_robustness"] = all_results["domain"].get("domain_robustness", 0)
        print(f"  2. Domain Robustness:   {summary['domain_robustness']:.4f}")

    if "position" in all_results and all_results["position"]:
        summary["position_robustness"] = all_results["position"].get("position_robustness", 0)
        print(f"  3. Position Robustness: {summary['position_robustness']:.4f}")

    if "fewshot" in all_results and all_results["fewshot"]:
        summary["fewshot_performance"] = all_results["fewshot"].get("fewshot_average", 0)
        print(f"  4. Few-shot Performance: {summary['fewshot_performance']:.4f}")

    if "zeroshot" in all_results and all_results["zeroshot"]:
        summary["zeroshot_performance"] = all_results["zeroshot"].get("summary", {}).get("mean_f1", 0)
        print(f"  5. Zero-shot Performance: {summary['zeroshot_performance']:.4f}")

    # Save
    final_results = {
        "model": args.model,
        "weights": args.weights,
        "timestamp": timestamp,
        "num_gpus": args.num_gpus,
        "parallel_per_gpu": args.parallel,
        "summary": summary,
        "detailed": all_results,
    }

    results_path = os.path.join(args.output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
