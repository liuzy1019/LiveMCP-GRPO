"""
SchemaShift-GRPO 自定义 advantage estimator。

注册为 verl 的 "schemashift_grpo" estimator。
优先从 non_tensor_batch（由 _get_gen_batch 保留，register_estimator.py 转发）读取
perturbation_level、scenario_type 和 group_id，fallback 到从 uid 解析。

uid 格式: {task_id}___{level}___{scenario_type}（新格式）
         {task_id}___{level}（旧格式，兼容）
"""

from collections import defaultdict
import os

import numpy as np
import torch
from loguru import logger

try:
    from verl.trainer.ppo.core_algos import register_adv_est
except ImportError as e:
    raise ImportError(
        "verl 未安装或版本不兼容，schemashift_grpo estimator 无法注册。"
        f"原始错误: {e}"
    )


# ── LATA helpers ────────────────────────────────────────────────────

_LATA_WARNED = False


def _resolve_lata_mode(config) -> str:
    """从 config 或环境变量解析 LATA 模式"""
    # 环境变量优先（调试方便）
    if os.environ.get("OVAL_LATA", ""):
        return os.environ["OVAL_LATA"]
    # config 次之
    if config:
        return str(config.get("lata_mode", config.get("lata", "none")))
    return "none"


def _apply_lata(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    mode: str,
    config,
) -> torch.Tensor:
    """Apply LATA allocation to trajectory-level advantages.

    Fall back to standard allocation on import failure.
    """
    global _LATA_WARNED
    try:
        from src.oval_mcp.training.lata import LATAAllocator, LATAConfig
        allocator = LATAAllocator(LATAConfig(mode=mode))
        result = allocator.allocate_from_mask(advantages, response_mask)
        if not _LATA_WARNED:
            logger.info(
                f"[LATA] mode={mode} | mean_len={result.mean_length:.1f} "
                f"| scale_range=[{min(result.per_token_scale):.3f}, {max(result.per_token_scale):.3f}]"
            )
            _LATA_WARNED = True
        return result.token_advantages
    except Exception as e:
        if not _LATA_WARNED:
            logger.warning(f"[LATA] 分配失败，回退到 standard allocation: {e}")
            _LATA_WARNED = True
        return advantages.unsqueeze(-1) * response_mask.float()


def _parse_uid(uid: str) -> tuple[str, str, str]:
    """从 uid 解析 (task_id, perturbation_level, scenario_type)。

    支持新格式 {task_id}___{level}___{scenario_type} 和
    旧格式 {task_id}___{level}（scenario_type 默认 "single_step"）。
    """
    parts = uid.split("___")
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        return parts[0], parts[1], "single_step"
    return uid, "none", "single_step"


def _diagnose_batch(index, non_tensor_batch, task_ids, levels, scenario_types, scores):
    """记录诊断信息，帮助排查数据流问题。"""
    n = len(scores)
    nunique_task = len(set(task_ids))
    nunique_level = len(set(levels))
    nunique_scenario = len(set(scenario_types))
    nb_fields = set(non_tensor_batch.keys()) if non_tensor_batch else set()
    logger.info(
        f"[schemashift_grpo] batch={n}, tasks={nunique_task}, "
        f"levels={nunique_level}, scenarios={nunique_scenario}, "
        f"nb_fields={nb_fields}, "
        f"score_range=[{scores.min().item():.4f}, {scores.max().item():.4f}]"
    )

    # 运行时 group 完整性检查：每个 group 应有 9 条（3:3:3）
    # 如果大量 group 不完整，说明 shuffle 或 batch_size 配置错误
    from collections import Counter
    group_sizes = Counter(task_ids)
    incomplete = {g: c for g, c in group_sizes.items() if c != 9}
    if incomplete:
        n_incomplete = len(incomplete)
        logger.warning(
            f"[schemashift_grpo] {n_incomplete}/{nunique_task} groups 记录数 != 9。"
            f"StratAdv 可能退化（请确认 data.shuffle=False 且 train_batch_size 为 9 的倍数）。"
            f"不完整 group 示例 (前3): "
            + ", ".join(f"{g}:{c}" for g, c in list(incomplete.items())[:3])
        )
    else:
        # 数据驱动的分层多样性诊断（不硬编码标签集）
        for gid in set(task_ids):
            g_levels = [levels[i] for i, tid in enumerate(task_ids) if tid == gid]
            unique_levels = set(g_levels)
            if len(unique_levels) < 2:
                logger.warning(
                    f"[schemashift_grpo] group {gid} 所有样本 "
                    f"perturbation_level 相同 ({unique_levels})，"
                    f"分层 advantage 退化为 GRPO baseline"
                )


@register_adv_est("schemashift_grpo")
def compute_schemashift_grpo_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config=None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A = strat_z + beta * global_z。

    与 verl 集成：
    - verl 的 ray_trainer 通过 adv_kwargs 传入 token_level_rewards, response_mask, index(uid), config
    - non_tensor_batch 通过 register_estimator.py 的 monkey-patch 注入
    - 含 perturbation_level（分层）、scenario_type（场景类型）和 group_id（分组=task_id）
    - fallback: 从 uid 解析

    MCP-RL Full 扩展：
    - stratum = (perturbation_level, scenario_type)
    - fallback: stratum 内 < 3 样本时逐级回退（scenario → global）
    """
    # P1-3: 支持 config.beta 和 config.schemashift.beta 两种路径
    if config:
        beta_val = config.get("beta", None)
        if beta_val is None:
            # fallback: nested path
            ss_cfg = config.get("schemashift", {})
            if isinstance(ss_cfg, dict):
                beta_val = ss_cfg.get("beta", 0.25)
            else:
                beta_val = 0.25
        beta = float(beta_val)
    else:
        beta = 0.25
    min_stratum_size = int(config.get("min_stratum_size", 3) if config else 3)

    scores = token_level_rewards.sum(dim=-1)
    bsz = scores.shape[0]

    # ── 从 non_tensor_batch（优先）或 uid（fallback）解析分组和分层信息 ──
    non_tensor_batch = kwargs.get("non_tensor_batch")
    task_ids = []
    levels = []
    scenario_types = []
    source = "uid"  # 标记数据来源，用于诊断

    if non_tensor_batch is not None:
        nb_groups = non_tensor_batch.get("group_id")
        nb_levels = non_tensor_batch.get("perturbation_level")
        nb_scenarios = non_tensor_batch.get("scenario_type")

        # fallback: 从 extra_info dict 中展开（旧版 parquet 兼容）
        if (nb_groups is None or nb_levels is None) and "extra_info" in non_tensor_batch:
            extra_infos = non_tensor_batch["extra_info"]
            if isinstance(extra_infos, np.ndarray) and extra_infos.ndim > 0:
                extras = extra_infos.tolist()
            elif isinstance(extra_infos, (list, tuple)):
                extras = list(extra_infos)
            else:
                extras = [extra_infos]
            if extras and isinstance(extras[0], dict):
                if nb_groups is None:
                    nb_groups = np.array([e.get("group_id", e.get("episode_id", f"unk_{i}")) for i, e in enumerate(extras)], dtype=object)
                if nb_levels is None:
                    nb_levels = np.array([e.get("perturbation_level", "none") for e in extras], dtype=object)
                if nb_scenarios is None:
                    nb_scenarios = np.array([e.get("scenario_type", "single_step") for e in extras], dtype=object)

        if nb_groups is not None and nb_levels is not None:
            # 安全地将值转为 Python 列表（兼容 np.ndarray 向量/标量、list、单值 str）
            def _to_list(val):
                if isinstance(val, np.ndarray) and val.ndim > 0:
                    return val.tolist()
                elif isinstance(val, (list, tuple)):
                    return list(val)
                else:
                    # 单值（str, int, np.str_ 标量等）
                    return [str(val)]

            task_ids = _to_list(nb_groups)
            levels = _to_list(nb_levels)
            scenario_types = _to_list(nb_scenarios) if nb_scenarios is not None else ["single_step"] * bsz
            # 单值 → 广播到 batch 长度
            if len(task_ids) == 1 and bsz > 1:
                task_ids = task_ids * bsz
            if len(levels) == 1 and bsz > 1:
                levels = levels * bsz
            if len(scenario_types) == 1 and bsz > 1:
                scenario_types = scenario_types * bsz
            source = "non_tensor_batch"

    if not task_ids or not levels:
        # fallback: 从 uid 解析
        logger.warning(
            "non_tensor_batch 中缺少 perturbation_level/group_id，"
            "回退到 uid 解析 — 训练阶段分层信息将丢失，"
            "advantage 退化（建议检查数据流水线）"
        )
        for uid in index:
            tid, level, scenario = _parse_uid(str(uid))
            task_ids.append(tid)
            levels.append(level)
            scenario_types.append(scenario)
        source = "uid (fallback)"

    # P1-8: 校验 metadata 数组长度等于 batch size
    if len(task_ids) != bsz or len(levels) != bsz or len(scenario_types) != bsz:
        raise ValueError(
            f"[schemashift_grpo] metadata 长度不匹配 batch size: "
            f"task_ids={len(task_ids)}, levels={len(levels)}, "
            f"scenario_types={len(scenario_types)}, bsz={bsz}, source={source}"
        )

    with torch.no_grad():
        advantages = torch.zeros(bsz, device=scores.device)

        # 按 task_id 分组
        task2indices = defaultdict(list)
        for i, tid in enumerate(task_ids):
            task2indices[tid].append(i)

        # 检测是否所有 group 都是 size=1（当前数据无 E4 分组结构）
        # 此时 fallback 到 batch-level GRPO（整个 batch 做 z-score）
        max_group_size = max(len(v) for v in task2indices.values())
        if max_group_size == 1:
            # Batch-level GRPO fallback：所有样本在 batch 内比较
            if norm_adv_by_std_in_grpo and bsz >= 2:
                batch_mean = scores.mean()
                batch_std = scores.std(unbiased=False).clamp(min=epsilon)
                advantages = (scores - batch_mean) / batch_std
            elif bsz >= 2:
                advantages = scores - scores.mean()
            # else: bsz=1, advantage=0（无法比较）

            if not hasattr(compute_schemashift_grpo_advantage, '_fallback_warned'):
                logger.warning(
                    f"[schemashift_grpo] 所有 group 均为 size=1 (n_groups={len(task2indices)})，"
                    f"fallback 到 batch-level GRPO。如需 StratAdv，请确保数据包含 E4 分组结构。"
                )
                compute_schemashift_grpo_advantage._fallback_warned = True
        else:
            # 正常 E4 分层逻辑
            for tid, indices in task2indices.items():
                idx_tensor = torch.tensor(indices, device=scores.device)
                group_scores = scores[idx_tensor]
                group_levels = [levels[i] for i in indices]
                group_scenarios = [scenario_types[i] for i in indices]
                n_group = len(group_scores)

                # 按 (perturbation_level, scenario_type) 2 维分层
                stratum2local = defaultdict(list)
                scenario2local = defaultdict(list)
                for local_i, (level, scenario) in enumerate(zip(group_levels, group_scenarios)):
                    stratum2local[(level, scenario)].append(local_i)
                    scenario2local[scenario].append(local_i)

                # Step 1: 层内归一化（带 fallback）
                strat_advs = torch.zeros(n_group, device=scores.device)
                for stratum_key, loc_indices in stratum2local.items():
                    loc_tensor = torch.tensor(loc_indices, device=scores.device)
                    stratum_scores = group_scores[loc_tensor]
                    n_stratum = len(loc_indices)

                    if n_stratum >= min_stratum_size:
                        # 正常 z-score
                        if norm_adv_by_std_in_grpo:
                            s_mean = stratum_scores.mean()
                            s_std = stratum_scores.std(unbiased=False).clamp(min=epsilon)
                            strat_advs[loc_tensor] = (stratum_scores - s_mean) / s_std
                        else:
                            strat_advs[loc_tensor] = stratum_scores - stratum_scores.mean()
                    elif n_stratum == 2:
                        # 只减均值，不除 std
                        strat_advs[loc_tensor] = stratum_scores - stratum_scores.mean()
                    elif n_stratum == 1:
                        # 回退到 scenario-level
                        scenario = stratum_key[1]
                        sc_indices = scenario2local[scenario]
                        if len(sc_indices) >= 2:
                            sc_tensor = torch.tensor(sc_indices, device=scores.device)
                            sc_scores = group_scores[sc_tensor]
                            sc_mean = sc_scores.mean()
                            if len(sc_indices) >= min_stratum_size and norm_adv_by_std_in_grpo:
                                sc_std = sc_scores.std(unbiased=False).clamp(min=epsilon)
                                strat_advs[loc_tensor] = (stratum_scores - sc_mean) / sc_std
                            else:
                                strat_advs[loc_tensor] = stratum_scores - sc_mean
                        else:
                            # 回退到 group-level global
                            g_mean = group_scores.mean()
                            if n_group >= min_stratum_size and norm_adv_by_std_in_grpo:
                                g_std = group_scores.std(unbiased=False).clamp(min=epsilon)
                                strat_advs[loc_tensor] = (stratum_scores - g_mean) / g_std
                            else:
                                strat_advs[loc_tensor] = stratum_scores - g_mean

                # Step 2: 全局 z-score 用作残差
                group_mean = group_scores.mean()
                if norm_adv_by_std_in_grpo and n_group >= 2:
                    group_std = group_scores.std(unbiased=False).clamp(min=epsilon)
                    global_z = (group_scores - group_mean) / group_std
                else:
                    global_z = group_scores - group_mean

                # Step 3: A = strat_z + beta * global_z
                advantages[idx_tensor] = strat_advs + beta * global_z

        # 首次执行时打诊断日志
        if not hasattr(compute_schemashift_grpo_advantage, '_diagnosed'):
            _diagnose_batch(index, non_tensor_batch, task_ids, levels, scenario_types, scores)
            compute_schemashift_grpo_advantage._diagnosed = True

        # ── 饱和组检测与跳过（§9.2-9.3） ──
        # std(J) < min_group_std → advantage = 0，不产生 policy gradient
        # 饱和组 rollout 仍参与 lambda_safe 的 hat_C_batch（在 register_estimator 层已处理）
        min_group_std = float(config.get("min_group_std", 1e-6) if config else 1e-6)
        n_saturated_groups = 0
        n_total_groups = len(task2indices)
        saturated_group_ids: list[str] = []

        if max_group_size >= 2:
            for tid, indices in task2indices.items():
                if len(indices) < 2:
                    continue
                group_j = scores[torch.tensor(indices, device=scores.device)]
                g_mean = group_j.mean()
                g_var = ((group_j - g_mean) ** 2).mean()
                g_std = g_var.sqrt()
                if g_std < min_group_std:
                    advantages[torch.tensor(indices, device=scores.device)] = 0.0
                    n_saturated_groups += 1
                    saturated_group_ids.append(tid)

        if n_saturated_groups > 0 and not hasattr(compute_schemashift_grpo_advantage, '_sat_warned'):
            logger.warning(
                f"[schemashift_grpo] SATURATION: {n_saturated_groups}/{n_total_groups} groups skipped "
                f"(std(J) < {min_group_std:.0e}). "
                f"saturated_ids (前5): {saturated_group_ids[:5]}"
            )
            compute_schemashift_grpo_advantage._sat_warned = True

        # ── LATA: Length-Aware Token Allocation ──
        # 当全部组饱和（advantage 已全零）时跳过 LATA 分配，避免无效计算
        lata_mode = _resolve_lata_mode(config)
        if lata_mode != "none" and n_saturated_groups < n_total_groups:
            advantages = _apply_lata(advantages, response_mask, lata_mode, config)
        else:
            advantages = advantages.unsqueeze(-1) * response_mask

    return advantages, advantages.clone()
