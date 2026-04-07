from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate experiment plots for AI Agent Arena.")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("experiment_results.json"),
        help="Structured experiment results JSON produced by the trainer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory for saved plots and comparison summaries.",
    )
    parser.add_argument("--dpi", type=int, default=160, help="Figure DPI.")
    return parser.parse_args()


def load_results(results_path: Path) -> dict[str, Any]:
    return json.loads(results_path.read_text())


def plot_reward_histories(results: dict[str, Any], output_dir: Path, dpi: int) -> Path:
    training_runs = results["training_runs"]
    static_rewards = training_runs["static_agent"]["reward_history"]
    dynamic_rewards = training_runs["dynamic_agent"]["reward_history"]

    plt.figure(figsize=(10, 5))
    plt.plot(static_rewards, label="Static training reward", linewidth=2)
    plt.plot(dynamic_rewards, label="Dynamic training reward", linewidth=2)
    plt.title("Reward vs Episodes")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "reward_vs_episodes.png"
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    return output_path


def plot_chaos_vs_success(results: dict[str, Any], output_dir: Path, dpi: int) -> Path | None:
    chaos_impact = results["experiments"].get("chaos_level_impact", [])
    if not chaos_impact:
        return None

    chaos_levels = [entry["chaos_level"] for entry in chaos_impact]
    success_rates = [entry["eval_summary"]["success_rate"] for entry in chaos_impact]

    plt.figure(figsize=(8, 5))
    plt.plot(chaos_levels, success_rates, marker="o", linewidth=2, color="tab:red")
    plt.title("Chaos Level vs Success Rate")
    plt.xlabel("Chaos level")
    plt.ylabel("Success rate")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.tight_layout()

    output_path = output_dir / "chaos_level_vs_success_rate.png"
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    return output_path


def plot_static_dynamic_comparison(
    results: dict[str, Any],
    output_dir: Path,
    dpi: int,
) -> Path:
    static_dynamic = results["experiments"]["static_vs_dynamic"]
    labels = ["Static Eval", "Dynamic Eval"]
    success_rates = [
        static_dynamic["static_eval"]["success_rate"],
        static_dynamic["dynamic_eval"]["success_rate"],
    ]

    plt.figure(figsize=(7, 5))
    bars = plt.bar(labels, success_rates, color=["tab:blue", "tab:orange"])
    plt.title("Static vs Dynamic Comparison")
    plt.ylabel("Success rate")
    plt.ylim(0.0, 1.0)
    plt.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, success_rates):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")

    plt.tight_layout()
    output_path = output_dir / "static_vs_dynamic_comparison.png"
    plt.savefig(output_path, dpi=dpi)
    plt.close()
    return output_path


def build_comparison_summary(results: dict[str, Any]) -> dict[str, Any]:
    robustness = results["experiments"]["static_vs_dynamic"]["robustness"]
    generalization = results["experiments"]["seen_vs_unseen_layout"]["generalization"]
    chaos_impact = results["experiments"].get("chaos_level_impact", [])

    summary: dict[str, Any] = {
        "project": results["metadata"]["project"],
        "key_insight": "Agents trained in static environments fail under dynamic conditions.",
        "robustness": robustness,
        "generalization": generalization,
    }

    if chaos_impact:
        summary["chaos_sweep"] = [
            {
                "chaos_level": entry["chaos_level"],
                "success_rate": entry["eval_summary"]["success_rate"],
                "average_reward": entry["eval_summary"]["average_reward"],
            }
            for entry in chaos_impact
        ]

    return summary


def save_comparison_summary(summary: dict[str, Any], output_dir: Path) -> Path:
    output_path = output_dir / "comparison_summary.json"
    output_path.write_text(json.dumps(summary, indent=2))
    return output_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = load_results(args.results_path)
    saved_paths = [
        plot_reward_histories(results, args.output_dir, args.dpi),
        plot_static_dynamic_comparison(results, args.output_dir, args.dpi),
    ]

    chaos_plot = plot_chaos_vs_success(results, args.output_dir, args.dpi)
    if chaos_plot is not None:
        saved_paths.append(chaos_plot)

    comparison_summary = build_comparison_summary(results)
    saved_paths.append(save_comparison_summary(comparison_summary, args.output_dir))

    print("Saved analysis artifacts:")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
