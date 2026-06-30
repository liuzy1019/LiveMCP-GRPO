"""lambda_safe 跨进程共享状态管理器（含 stall protection 和进程安全锁）。

OVAL-MCP §9 要求：
  - lambda_safe 在 batch 边界更新（projected dual ascent）
  - lambda_safe 由 reward function 消费（计算 J 时使用）
  - lambda_safe 更新需要所有 valid rollout 的 C_safety 值

Stall protection (§9.3)：
  - 连续 K_stall 步 hat_C > tau_unsafe_stall 时冻结 lambda_safe
  - 冻结期间 λ 不再增大，但允许减小（hat_C 回到正常范围时自动解冻）
  - streak 和 frozen 状态持久化到文件中，跨进程重启不丢失

进程安全：
  - atomic_update() 在整个 load→update→save 周期持有 fcntl 排他锁
  - 防止 Ray 多 actor 并发更新时的 write-after-read 覆盖
  - 锁文件跟随进程生命周期，进程崩溃自动释放

使用方式:
  # 推荐：原子更新（进程安全）
  state, new_lambda, skipped = LambdaState.atomic_update(c_safety_values)

  # 只读（无锁，os.replace 本身原子安全）
  state = LambdaState.load_or_default()
  lambda_val = state.lambda_safe
"""

from __future__ import annotations

import fcntl
import json
import os
import threading


DEFAULT_STATE_PATH = "/tmp/ovalmcp_lambda_state.json"


def _lock_file(path: str) -> int:
    """获取排他文件锁的文件描述符。"""
    lock_path = path + ".lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _unlock_file(fd: int) -> None:
    """释放文件锁并关闭 fd。"""
    try:
        fcntl.flock(fd, fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        os.close(fd)
    except Exception:
        pass


class LambdaState:
    """File-backed shared state for lambda_safe with stall protection.

    Threading lock (_lock) 保护单进程内 save() 并发。
    跨进程安全由 atomic_update() 的 fcntl 文件锁保证。
    """

    def __init__(
        self,
        lambda_safe: float = 1.0,
        alpha_lambda: float = 0.01,
        epsilon: float = 0.05,
        lambda_safe_max: float = 3.0,
        step: int = 0,
        state_path: str = DEFAULT_STATE_PATH,
        lambda_increase_streak: int = 0,
        stall_frozen: bool = False,
    ):
        self.lambda_safe = lambda_safe
        self.alpha_lambda = alpha_lambda
        self.epsilon = epsilon
        self.lambda_safe_max = lambda_safe_max
        self.step = step
        self._path = state_path
        self._lock = threading.Lock()

        # stall protection (persisted across process restarts)
        self._lambda_increase_streak: int = lambda_increase_streak
        self._stall_frozen: bool = stall_frozen

    # ── 原子更新（跨进程安全） ──────────────────────────────────────

    @classmethod
    def atomic_update(
        cls,
        c_safety_values: list[int],
        path: str = DEFAULT_STATE_PATH,
        k_stall: int = 10,
        tau_unsafe_stall: float = 0.5,
    ) -> tuple["LambdaState", float, float, bool]:
        """跨进程安全的 load→update→save 原子操作。

        持有 fcntl 排他锁，防止并发 write-after-read 覆盖。
        返回 (state, old_lambda, new_lambda, update_skipped)。
        """
        fd = _lock_file(path)
        try:
            state = cls.load_or_default(path=path)
            old_lambda = state.lambda_safe
            new_lambda, skipped = state.update(
                c_safety_values,
                k_stall=k_stall,
                tau_unsafe_stall=tau_unsafe_stall,
            )
            state.save()
            return state, old_lambda, new_lambda, skipped
        finally:
            _unlock_file(fd)

    # ── update ──────────────────────────────────────────────────────

    def update(
        self,
        c_safety_values: list[int],
        k_stall: int = 10,
        tau_unsafe_stall: float = 0.5,
    ) -> tuple[float, bool]:
        """Dual ascent update with stall protection.

        Returns (new_lambda_safe, update_skipped).
        update_skipped=True → lambda_safe was frozen and the increase was skipped.

        Stall logic:
          1. hat_C > tau_unsafe_stall AND λ would increase → streak++
          2. streak >= k_stall → freeze (skip future increases)
          3. hat_C <= tau_unsafe_stall → decrement streak, unfreeze when streak falls below k_stall
          4. Frozen mode only blocks increases; decreases are always applied.
        """
        if not c_safety_values:
            return self.lambda_safe, False

        hat_c = sum(c_safety_values) / len(c_safety_values)
        new_lambda = self.lambda_safe + self.alpha_lambda * (hat_c - self.epsilon)
        lambda_would_increase = new_lambda > self.lambda_safe
        unsafe_dominant = hat_c > tau_unsafe_stall

        # ── streak tracking ──
        if unsafe_dominant and lambda_would_increase:
            self._lambda_increase_streak += 1
            if self._lambda_increase_streak >= k_stall:
                self._stall_frozen = True
        elif not unsafe_dominant:
            self._lambda_increase_streak = max(0, self._lambda_increase_streak - 1)
            if self._lambda_increase_streak < k_stall:
                self._stall_frozen = False

        # ── apply update ──
        if self._stall_frozen and lambda_would_increase:
            self.step += 1
            return self.lambda_safe, True

        self.lambda_safe = max(0.0, min(new_lambda, self.lambda_safe_max))
        self.step += 1
        return self.lambda_safe, False

    # ── diagnostics ─────────────────────────────────────────────────

    @property
    def is_stall_frozen(self) -> bool:
        return self._stall_frozen

    @property
    def stall_streak(self) -> int:
        return self._lambda_increase_streak

    @property
    def is_at_max(self) -> bool:
        return self.lambda_safe >= self.lambda_safe_max

    # ── persistence ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "lambda_safe": self.lambda_safe,
            "alpha_lambda": self.alpha_lambda,
            "epsilon": self.epsilon,
            "lambda_safe_max": self.lambda_safe_max,
            "step": self.step,
            "lambda_increase_streak": self._lambda_increase_streak,
            "stall_frozen": self._stall_frozen,
        }

    def save(self) -> None:
        with self._lock:
            tmp = self._path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(self.to_dict(), f)
            os.replace(tmp, self._path)

    @classmethod
    def load_or_default(
        cls,
        path: str = DEFAULT_STATE_PATH,
        **overrides,
    ) -> "LambdaState":
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                return cls(
                    lambda_safe=data.get("lambda_safe", 1.0),
                    alpha_lambda=data.get("alpha_lambda", 0.01),
                    epsilon=data.get("epsilon", 0.05),
                    lambda_safe_max=data.get("lambda_safe_max", 10.0),
                    step=data.get("step", 0),
                    state_path=path,
                    lambda_increase_streak=data.get("lambda_increase_streak", 0),
                    stall_frozen=data.get("stall_frozen", False),
                    **overrides,
                )
            except (json.JSONDecodeError, KeyError):
                pass
        return cls(state_path=path, **overrides)

    @classmethod
    def reset(cls, path: str = DEFAULT_STATE_PATH) -> None:
        """含锁重置：删除状态文件，下次 load_or_default 返回默认值。"""
        fd = _lock_file(path)
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        finally:
            _unlock_file(fd)


__all__ = ["LambdaState", "DEFAULT_STATE_PATH"]
