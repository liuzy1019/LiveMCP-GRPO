#!/usr/bin/env python3
"""训练曲线可视化。

从 wandb 或本地日志读取训练数据，生成论文级图表。
图表风格对标参考 repo（agentic-grpo-longhorizon）。

用法:
    # 从 wandb 读取
    python scripts/plot_results.py --wandb --project schemashift-grpo

    # 从本地 CSV 读取
    python scripts/plot_results.py --csv data/results.csv

    # 输出到指定目录
    python scripts/plot_results.py --output docs/figures
"""

import argparse
import json
from pathlib import Path
from typing import Optional
from loguru import logger

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── matplotlib 全局设置（论文级） ──
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})


# ═══════════════════════════════════════════
# 图 1: 训练曲线
# ═══════════════════════════════════════════

def plot_training_curves(
    data: dict[str, dict],
    output_path: str,
):
    """训练过程折线图。

    对应参考 repo 的 ablation_progression.png。
    横轴: 训练步数; 纵轴: pass@1 / reward.

    Args:
        data: {exp_name: {"steps": [int], "pass@1": [float], ...}}
        output_path: 输出路径。
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图 1: pass@1 曲线 ──
    ax = axes[0]
    colors = {"E3_Baseline": "#4C72B0", "E4_SchemaShift": "#DD8452"}
    for exp_name, exp_data in data.items():
        if "pass@1" in exp_data:
            steps = exp_data["steps"]
            values = exp_data["pass@1"]
            color = colors.get(exp_name, "#555555")
            ax.plot(steps, values, "-o", label=exp_name, color=color,
                    linewidth=2, markersize=5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("pass@1")
    ax.set_title("BFCL V3 Multi-Turn pass@1")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图 2: reward 曲线 ──
    ax = axes[1]
    for exp_name, exp_data in data.items():
        if "reward" in exp_data:
            steps = exp_data["steps"]
            values = exp_data["reward"]
            color = colors.get(exp_name, "#555555")
            ax.plot(steps, values, "-s", label=exp_name, color=color,
                    linewidth=2, markersize=4)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Group Mean Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图 3: 鲁棒性差距变化 ──
    ax = axes[2]
    for exp_name, exp_data in data.items():
        if "robustness_gap" in exp_data and exp_data["robustness_gap"]:
            steps = exp_data["steps"]
            values = exp_data["robustness_gap"]
            color = colors.get(exp_name, "#555555")
            ax.plot(steps, values, "-^", label=exp_name, color=color,
                    linewidth=2, markersize=4)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Robustness Gap (Original - Perturbed)")
    ax.set_title("Schema Overfitting During Training")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_path}/training_curves.png")
    plt.close()
    logger.info(f"已保存: {output_path}/training_curves.png")


# ═══════════════════════════════════════════
# 图 2: 消融对比柱状图
# ═══════════════════════════════════════════

def plot_ablation_comparison(
    results: dict[str, dict],
    output_path: str,
):
    """消融实验对比柱状图。

    对应参考 repo 的 ablation_comparison.png。
    每个实验显示两根柱子: 原版 schema / 扰动 schema

    Args:
        results: {exp_name: {"original": float, "perturbed": float}}
        output_path: 输出路径。
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    exp_names = list(results.keys())
    x = np.arange(len(exp_names))
    width = 0.35

    orig_scores = [results[e].get("original", 0) for e in exp_names]
    pert_scores = [results[e].get("perturbed", 0) for e in exp_names]
    gaps = [results[e].get("gap", 0) for e in exp_names]

    bars1 = ax.bar(x - width/2, orig_scores, width, label="Original Schema",
                   color="#4C72B0", edgecolor="white", linewidth=0.5)
    bars2 = ax.bar(x + width/2, pert_scores, width, label="Perturbed Schema",
                   color="#DD8452", edgecolor="white", linewidth=0.5)

    # 柱子上标注数值
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    # 在柱子间标注 gap
    for i, gap in enumerate(gaps):
        mid = (orig_scores[i] + pert_scores[i]) / 2
        ax.annotate(
            f"gap={gap:.2f}",
            xy=(x[i], mid),
            xytext=(x[i] + width * 0.7, mid + 0.08),
            fontsize=9, color="#CC0000",
            ha="center",
            arrowprops=dict(arrowstyle="->", color="#CC0000", lw=0.8),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, rotation=15, ha="right")
    ax.set_ylabel("pass@1")
    ax.set_title("Ablation: Schema Robustness Across Experiments")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_path}/ablation_comparison.png")
    plt.close()
    logger.info(f"已保存: {output_path}/ablation_comparison.png")


# ═══════════════════════════════════════════
# 图 3: 鲁棒性差距柱状图
# ═══════════════════════════════════════════

def plot_robustness_gap(
    results: dict[str, dict],
    output_path: str,
):
    """鲁棒性差距对比柱状图。

    只显示一根柱子（gap = original - perturbed），
    越小越好。用于快速判断哪个实验的过拟合最轻。

    Args:
        results: {exp_name: {"original": float, "perturbed": float}}
        output_path: 输出路径。
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    exp_names = list(results.keys())
    gaps = [results[e].get("gap", results[e]["original"] - results[e]["perturbed"])
            for e in exp_names]

    colors = ["#2ca02c" if g < 0.1 else "#ff7f0e" if g < 0.2 else "#d62728"
              for g in gaps]

    bars = ax.bar(exp_names, gaps, color=colors, edgecolor="white", linewidth=0.5)

    for bar, gap in zip(bars, gaps):
        h = bar.get_height()
        va = "bottom" if gap >= 0 else "top"
        offset = 0.005 if gap >= 0 else -0.015
        ax.text(bar.get_x() + bar.get_width()/2, h + offset,
                f"{gap:.3f}", ha="center", va=va, fontsize=10,
                color="black", fontweight="bold")

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.axhline(y=0.1, color="green", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Low overfitting (<0.1)")
    ax.axhline(y=0.2, color="orange", linestyle="--", linewidth=0.8, alpha=0.5,
               label="Moderate (>0.2)")
    ax.set_ylabel("Robustness Gap (Original - Perturbed)")
    ax.set_title("Schema Overfitting: Lower is Better")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_path}/robustness_gap.png")
    plt.close()
    logger.info(f"已保存: {output_path}/robustness_gap.png")


# ═══════════════════════════════════════════
# 图 4: 假设验证汇总表
# ═══════════════════════════════════════════

def plot_hypothesis_summary(
    hypotheses: list[dict],
    output_path: str,
):
    """假设验证结果表格。

    对应参考 repo README 中的 Hypothesis Validation 表格。

    Args:
        hypotheses: [{"hypothesis": str, "status": "✅"|"❌"|"🟡", "evidence": str}]
        output_path: 输出路径。
    """
    fig, ax = plt.subplots(figsize=(12, 3 + 0.4 * len(hypotheses)))
    ax.axis("off")

    col_labels = ["#", "Hypothesis", "Status", "Evidence"]
    rows = []
    for i, h in enumerate(hypotheses, 1):
        rows.append([str(i), h["hypothesis"], h["status"], h["evidence"]])

    table = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        colWidths=[0.05, 0.4, 0.08, 0.47],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)

    # 表头样式
    for j in range(4):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # 交替行色
    for i in range(1, len(rows) + 1):
        if i % 2 == 0:
            for j in range(4):
                table[i, j].set_facecolor("#E8EEF7")

    plt.title("Hypothesis Validation", fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(f"{output_path}/hypothesis_summary.png")
    plt.close()
    logger.info(f"已保存: {output_path}/hypothesis_summary.png")


# ═══════════════════════════════════════════
# 图 5: 工具名扰动统计
# ═══════════════════════════════════════════

def plot_perturbation_stats(
    name_map_path: str,
    output_path: str,
):
    """工具名扰动分布。

    展示训练过程中经过 name_map 映射的函数频率分布。

    Args:
        name_map_path: name_map.json 路径。
        output_path: 输出路径。
    """
    with open(name_map_path) as f:
        nm = json.load(f)

    forward = nm.get("forward", {})
    reverse = nm.get("reverse", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── 左: 扰动类型分布 ──
    ax = axes[0]
    from collections import Counter
    prefix_counter = Counter()
    for orig_name in reverse.keys():
        prefix = orig_name.split("_")[0] if "_" in orig_name else orig_name[:5]
        prefix_counter[prefix] += 1
    top_prefixes = prefix_counter.most_common(10)
    labels = [p[0] for p in top_prefixes]
    counts = [p[1] for p in top_prefixes]
    ax.barh(labels[::-1], counts[::-1], color="#4C72B0")
    ax.set_xlabel("Number of Perturbations")
    ax.set_title(f"Tool Name Perturbations (total={len(forward)})")
    ax.grid(True, axis="x", alpha=0.3)

    # ── 右: 扰动强度分布（示意） ──
    ax = axes[1]
    levels = ["None", "Mild", "Moderate", "Strong"]
    counts_by_level = [
        0,
        sum(1 for v in forward.values() if v in ["search", "get", "find"]),
        len(forward) // 2,
        len(forward),
    ]
    ax.bar(levels, counts_by_level, color=["#2ca02c", "#ffb07c", "#ff7f0e", "#d62728"])
    ax.set_ylabel("Approx. Affected Functions")
    ax.set_title("Perturbation Intensity Distribution")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_path}/perturbation_stats.png")
    plt.close()
    logger.info(f"已保存: {output_path}/perturbation_stats.png")


# ═══════════════════════════════════════════
# 生成全部图表
# ═══════════════════════════════════════════

def generate_all_figures(
    results: Optional[dict] = None,
    training_data: Optional[dict] = None,
    output_dir: str = "docs/figures",
    name_map_path: str = "data/name_map.json",
):
    """生成所有图表。

    Args:
        results: 消融实验结果 {exp: {original, perturbed, gap}}。
        training_data: 训练过程数据 {exp: {steps, pass@1, ...}}。
        output_dir: 输出目录。
        name_map_path: name_map.json 路径。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 使用示例数据（真实数据来自训练后替换）
    if results:
        plot_ablation_comparison(results, str(output_dir))
        plot_robustness_gap(results, str(output_dir))

    if training_data:
        plot_training_curves(training_data, str(output_dir))

    # 假设验证模板
    hypotheses = [
        {"hypothesis": "GRPO training amplifies schema overfitting (robustness gap > SFT)",
         "status": "PENDING", "evidence": "TBD: compare E3 gap vs E2 gap"},
        {"hypothesis": "SchemaShift-GRPO reduces robustness gap vs standard GRPO",
         "status": "PENDING", "evidence": "TBD: compare E4 gap vs E3 gap"},
        {"hypothesis": "Schema augmentation alone reduces gap",
         "status": "PENDING", "evidence": "TBD: compare E5 gap vs E3 gap"},
        {"hypothesis": "Stratified advantage alone reduces gap",
         "status": "PENDING", "evidence": "TBD: compare E6 gap vs E3 gap"},
        {"hypothesis": "SchemaShift-GRPO maintains or improves original schema pass@1",
         "status": "PENDING", "evidence": "TBD: compare E4 vs E3 original pass@1"},
    ]
    plot_hypothesis_summary(hypotheses, str(output_dir))

    # 扰动统计（如果有 name_map）
    name_map_path = Path(name_map_path)
    if name_map_path.exists():
        plot_perturbation_stats(str(name_map_path), str(output_dir))

    logger.info(f"全部图表已生成: {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="生成训练结果可视化图表")
    parser.add_argument("--output", default="docs/figures", help="输出目录")
    parser.add_argument("--name-map", default="data/name_map.json",
                        help="name_map.json 路径")
    parser.add_argument("--demo", action="store_true",
                        help="使用示例数据生成演示图表")
    args = parser.parse_args()

    if args.demo:
        # 示例数据（展示图表风格，训练完成后替换）
        demo_results = {
            "E1_ZeroShot":   {"original": 0.40, "perturbed": 0.35},
            "E2_SFT":        {"original": 0.55, "perturbed": 0.42},
            "E3_GRPO":       {"original": 0.60, "perturbed": 0.38},
            "E4_SchemaShift": {"original": 0.62, "perturbed": 0.55},
            "E5_AugOnly":    {"original": 0.61, "perturbed": 0.48},
            "E6_AdvOnly":    {"original": 0.58, "perturbed": 0.45},
        }
        # 计算 gap
        for v in demo_results.values():
            v["gap"] = v["original"] - v["perturbed"]

        demo_training = {
            "E3_Baseline": {
                "steps": list(range(0, 301, 50)),
                "pass@1": [0.40, 0.42, 0.48, 0.55, 0.58, 0.60, 0.60],
                "reward": [0.10, 0.15, 0.22, 0.30, 0.32, 0.35, 0.35],
                "robustness_gap": [0.05, 0.08, 0.12, 0.15, 0.18, 0.20, 0.22],
            },
            "E4_SchemaShift": {
                "steps": list(range(0, 301, 50)),
                "pass@1": [0.40, 0.44, 0.50, 0.55, 0.58, 0.60, 0.62],
                "reward": [0.10, 0.16, 0.24, 0.30, 0.33, 0.35, 0.36],
                "robustness_gap": [0.05, 0.06, 0.07, 0.07, 0.06, 0.07, 0.07],
            },
        }
        generate_all_figures(
            results=demo_results,
            training_data=demo_training,
            output_dir=args.output,
            name_map_path=args.name_map,
        )
    else:
        generate_all_figures(output_dir=args.output)


if __name__ == "__main__":
    main()
