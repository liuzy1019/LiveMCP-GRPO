"""公共 AST 匹配逻辑。

eval、reward、agent loop 三处共享此模块，避免训练 loop 依赖 eval 主文件。
"""
import ast
import re
from typing import Any


# ── _parse_bfcl_native_args 相关常量 ──
_PARSE_MAX_INPUT_LEN = 8192
_PARSE_MAX_ARGS = 64
_PARSE_MAX_KEY_LEN = 64
_PARSE_MAX_LITERAL_LEN = 4096
_PARSE_NAME_MAX_SCAN = 256
_IDENT_RE = re.compile(r"^[a-zA-Z_]\w{0,63}$")


def _looks_like_number(s: str) -> bool:
    """判断字符串是否像数字字面量。"""
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        return False


def _parse_bfcl_native_args(func_call_str: str) -> tuple[str, dict[str, Any]]:
    """解析 BFCL 原生格式的函数调用字符串，bounded linear parser。

    设计目标：永不卡死、无 O(N²) 路径、超长/畸形输入直接降级返回。
    任何分支失败都返回 ("", {}) 或 (name, {}), 由调用方走 fallback。
    """
    if not func_call_str or len(func_call_str) > _PARSE_MAX_INPUT_LEN:
        return "", {}

    # ---- Step 1: 函数名 ----
    head = func_call_str[:_PARSE_NAME_MAX_SCAN]
    name_match = re.match(r"([a-zA-Z_][\w.]{0,127})\s*\(", head)
    if not name_match:
        return "", {}
    name = name_match.group(1).rstrip(".")
    if not name:
        return "", {}

    # ---- Step 2: 定位匹配的 ')' ----
    args_start = func_call_str.find("(")
    if args_start < 0:
        return name, {}
    paren_depth = 0
    args_end = -1
    for j in range(args_start, len(func_call_str)):
        c = func_call_str[j]
        if c == "(":
            paren_depth += 1
        elif c == ")":
            paren_depth -= 1
            if paren_depth == 0:
                args_end = j
                break
    if args_end < 0:
        return name, {}
    args_part = func_call_str[args_start + 1:args_end]
    if not args_part or not args_part.strip():
        return name, {}

    # ---- Step 3: 切段 ----
    segments: list[tuple[str, int]] = []
    n = len(args_part)
    i = 0
    seg_start = 0
    eq_offset = -1
    quote = ""
    bracket = 0
    brace = 0
    paren = 0
    while i < n:
        if len(segments) >= _PARSE_MAX_ARGS:
            return name, {}
        c = args_part[i]
        if quote:
            if c == "\\" and i + 1 < n:
                i += 2
                continue
            if c == quote:
                quote = ""
            i += 1
            continue
        if c in ("'", '"'):
            quote = c
            i += 1
            continue
        if c == "[":
            bracket += 1
        elif c == "]" and bracket > 0:
            bracket -= 1
        elif c == "{":
            brace += 1
        elif c == "}" and brace > 0:
            brace -= 1
        elif c == "(":
            paren += 1
        elif c == ")" and paren > 0:
            paren -= 1
        elif c == "=" and bracket == 0 and brace == 0 and paren == 0 and eq_offset < 0:
            eq_offset = i - seg_start
        elif c == "," and bracket == 0 and brace == 0 and paren == 0:
            segments.append((args_part[seg_start:i], eq_offset))
            seg_start = i + 1
            eq_offset = -1
        i += 1
    tail = args_part[seg_start:n]
    if tail.strip():
        segments.append((tail, eq_offset))

    # ---- Step 4: 分 kwarg/positional ----
    args: dict[str, Any] = {}
    positional_idx = 0
    for seg_text, eq_off in segments:
        seg = seg_text.strip()
        if not seg:
            continue
        if eq_off >= 0:
            raw_key = seg_text[:eq_off]
            raw_val = seg_text[eq_off + 1:]
            key_stripped = raw_key.strip()
            if (
                len(key_stripped) <= _PARSE_MAX_KEY_LEN
                and _IDENT_RE.match(key_stripped)
            ):
                args[key_stripped] = raw_val.strip()
                continue
        args[f"_pos_{positional_idx}"] = seg
        positional_idx += 1

    # ---- Step 5: 字面量回填 ----
    for k in list(args.keys()):
        v = args[k]
        if not isinstance(v, str):
            continue
        s = v.strip()
        if not s or len(s) > _PARSE_MAX_LITERAL_LEN:
            continue
        if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
            try:
                args[k] = ast.literal_eval(s)
                continue
            except (ValueError, SyntaxError):
                args[k] = s[1:-1]
                continue
        if s[0] in "[{" or s in ("True", "False", "None") or _looks_like_number(s):
            try:
                args[k] = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                pass
    return name, args


def _normalize_value(v) -> str:
    """AST 类型宽松匹配：将值归一化为可比较的字符串。

    BFCL 官方 AST 评估允许以下等价：
    - "1" == 1 (字符串数字 vs 整数)
    - "True" == True (字符串布尔 vs 布尔)
    - "None" == None
    - "[1, 2]" == [1, 2] (字符串列表 vs 列表)
    """
    if v is None:
        return "None"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        if isinstance(v, float) and v == int(v):
            return str(int(v))
        return str(v)
    if isinstance(v, list):
        # 无序匹配：对 list 元素排序后再字符串化
        # 对齐 ComponentReward._values_match 的 multiset 语义
        try:
            sorted_list = sorted(v, key=lambda x: str(x))
            return str(sorted_list)
        except TypeError:
            return str(v)
    s = str(v)
    try:
        num = float(s)
        if num == int(num) and "." not in s:
            return str(int(num))
        return str(num)
    except (ValueError, TypeError):
        pass
    if len(s) >= 2 and s[0] in ('"', "'") and s[-1] == s[0]:
        s = s[1:-1]
    return s


def _is_null_equivalent(val) -> bool:
    """判断值是否为 null 等价。"""
    if val is None:
        return True
    if isinstance(val, str) and val.strip().lower() in ("", "null", "none"):
        return True
    return False


def _to_bool(v) -> bool | None:
    """尝试将值转为布尔（宽松，用于 values_match）。"""
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.strip().lower()
        if low in ("true", "1", "yes"):
            return True
        if low in ("false", "0", "no"):
            return False
    if isinstance(v, (int, float)):
        return bool(v)
    return None


def _to_bool_strict(v) -> bool | None:
    """严格布尔转换（用于 exact match + schema 校验）。

    只接受：
    - 实际 bool 值
    - 字符串 "true" / "false"（大小写不敏感）
    不接受 int(2)、"yes"、"no" 等任意 truthy/falsy 值。
    """
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        low = v.strip().lower()
        if low == "true":
            return True
        if low == "false":
            return False
    return None


def type_compatible(model_val: Any, oracle_val: Any) -> bool:
    """检查 model 值与 oracle 值的类型是否兼容（共享实现）。

    用于 exact match 判定：只有类型兼容 + 值匹配才算 exact success。
    规则：
    - None: model 也必须是 None
    - str: model 也必须是 str
    - bool: model 必须是 bool 或 "true"/"false" 字符串
    - int/float: model 必须是 int/float 或可解析为数字的字符串
    - list: model 必须是 list
    - dict: model 必须是 dict
    """
    if model_val is None:
        return oracle_val is None
    if oracle_val is None:
        return True  # oracle 允许 null

    # 字符串类型
    if isinstance(oracle_val, str):
        return isinstance(model_val, str)
    # 布尔类型（必须在 int/float 之前，因为 bool 是 int 的子类）
    if isinstance(oracle_val, bool):
        return isinstance(model_val, bool) or (
            isinstance(model_val, str) and model_val.strip().lower() in ("true", "false")
        )
    # 数值类型（int/float 互通）
    if isinstance(oracle_val, (int, float)) and not isinstance(oracle_val, bool):
        # P1-1: 排除 bool model_val（bool 是 int 子类，但 schema 语义不同）
        if isinstance(model_val, bool):
            return False
        if isinstance(model_val, (int, float)):
            return True
        # 字符串形式的数字也接受
        if isinstance(model_val, str):
            try:
                float(model_val)
                return True
            except (ValueError, TypeError):
                return False
        return False
    # 列表类型
    if isinstance(oracle_val, list):
        return isinstance(model_val, list)
    # 字典类型
    if isinstance(oracle_val, dict):
        return isinstance(model_val, dict)
    return True


def recursive_type_compatible(model_val: Any, oracle_val: Any) -> bool:
    """递归检查 model 值与 oracle 值的类型是否兼容。

    与 type_compatible 的区别：对 list/dict 容器递归检查内部元素的类型兼容性。
    用于 schema_valid 组件的递归类型校验。
    """
    # null-equivalence
    if _is_null_equivalent(model_val) and _is_null_equivalent(oracle_val):
        return True
    if _is_null_equivalent(model_val) or _is_null_equivalent(oracle_val):
        # 一侧 null 另一侧非 null：顶层 type_compatible 允许 oracle=None 时 model 任意
        return type_compatible(model_val, oracle_val)

    # 顶层类型检查
    if not type_compatible(model_val, oracle_val):
        return False

    # 列表：递归检查每个元素（使用 multiset 匹配找最佳配对）
    if isinstance(oracle_val, list) and isinstance(model_val, list):
        if len(model_val) != len(oracle_val):
            return False
        remaining = list(model_val)
        for o_elem in oracle_val:
            found = False
            for i, m_elem in enumerate(remaining):
                if recursive_type_compatible(m_elem, o_elem):
                    remaining.pop(i)
                    found = True
                    break
            if not found:
                return False
        return True

    # Dict：递归检查每个 value
    if isinstance(oracle_val, dict) and isinstance(model_val, dict):
        if set(model_val.keys()) != set(oracle_val.keys()):
            return False
        return all(
            recursive_type_compatible(model_val[k], oracle_val[k])
            for k in oracle_val
        )

    return True


def values_match(model_val, oracle_val) -> bool:
    """递归值匹配（共享实现，replay/reward/eval 统一使用）。

    规则：
    - None / null / "" / "null" 视为等价（null-equivalence）
    - 字符串比较：strip() + lower()
    - 数值比较：float(a) == float(b) with tolerance 1e-9
    - 布尔比较：两侧 cast 到 bool（宽松）
    - 列表比较：sorted element-wise（无序 multiset 匹配）
    - Dict/object 比较：递归 key-value match
    """
    # null-equivalence
    if _is_null_equivalent(model_val) and _is_null_equivalent(oracle_val):
        return True
    if _is_null_equivalent(model_val) or _is_null_equivalent(oracle_val):
        return False

    # 布尔比较（必须在数值之前，因为 bool 是 int 子类）
    if isinstance(oracle_val, bool) or isinstance(model_val, bool):
        mb = _to_bool(model_val)
        ob = _to_bool(oracle_val)
        if mb is None or ob is None:
            return False
        return mb == ob

    # 数值比较
    if isinstance(oracle_val, (int, float)) or isinstance(model_val, (int, float)):
        try:
            return abs(float(model_val) - float(oracle_val)) < 1e-9
        except (ValueError, TypeError):
            return False

    # 字符串比较
    if isinstance(oracle_val, str) and isinstance(model_val, str):
        return model_val.strip().lower() == oracle_val.strip().lower()

    # 列表比较（无序，recursive multiset matching）
    if isinstance(oracle_val, list) and isinstance(model_val, list):
        if len(model_val) != len(oracle_val):
            return False
        remaining = list(model_val)
        for o_elem in oracle_val:
            found = False
            for i, m_elem in enumerate(remaining):
                if values_match(m_elem, o_elem):
                    remaining.pop(i)
                    found = True
                    break
            if not found:
                return False
        return True

    # Dict 递归比较
    if isinstance(oracle_val, dict) and isinstance(model_val, dict):
        if set(model_val.keys()) != set(oracle_val.keys()):
            return False
        return all(
            values_match(model_val[k], oracle_val[k])
            for k in oracle_val
        )

    # 字符串化 fallback
    return str(model_val).strip().lower() == str(oracle_val).strip().lower()


def strict_values_match(model_val, oracle_val) -> bool:
    """严格值匹配：递归要求 type_compatible + 值匹配。

    用于 exact match 判定（replay 释放 observation、reward exact_success）。
    与 values_match 的区别：
    - boolean 场景下不接受 int(2)、"yes" 等任意 truthy/falsy 值
    - 要求类型兼容性（schema 层面合法）
    - 对 list/dict 容器递归使用 strict_values_match（而非宽松的 values_match）
    """
    # null-equivalence（两侧都是 null 等价）
    if _is_null_equivalent(model_val) and _is_null_equivalent(oracle_val):
        return True
    if _is_null_equivalent(model_val) or _is_null_equivalent(oracle_val):
        return False

    # 先检查顶层类型兼容性
    if not type_compatible(model_val, oracle_val):
        return False

    # 布尔：使用严格转换
    if isinstance(oracle_val, bool) or isinstance(model_val, bool):
        mb = _to_bool_strict(model_val)
        ob = _to_bool_strict(oracle_val)
        if mb is None or ob is None:
            return False
        return mb == ob

    # 列表：递归 strict matching（无序 multiset）
    if isinstance(oracle_val, list):
        if not isinstance(model_val, list) or len(model_val) != len(oracle_val):
            return False
        remaining = list(model_val)
        for o_elem in oracle_val:
            found = False
            for i, m_elem in enumerate(remaining):
                if strict_values_match(m_elem, o_elem):
                    remaining.pop(i)
                    found = True
                    break
            if not found:
                return False
        return True

    # Dict：递归 strict matching
    if isinstance(oracle_val, dict):
        if not isinstance(model_val, dict) or set(model_val.keys()) != set(oracle_val.keys()):
            return False
        return all(
            strict_values_match(model_val[k], oracle_val[k])
            for k in oracle_val
        )

    # 数值比较（type_compatible 已确保类型合法）
    if isinstance(oracle_val, (int, float)) or isinstance(model_val, (int, float)):
        try:
            return abs(float(model_val) - float(oracle_val)) < 1e-9
        except (ValueError, TypeError):
            return False

    # 字符串比较
    if isinstance(oracle_val, str) and isinstance(model_val, str):
        return model_val.strip().lower() == oracle_val.strip().lower()

    # 字符串化 fallback
    return str(model_val).strip().lower() == str(oracle_val).strip().lower()


def strict_args_match(agent_args: dict, gt_args: dict) -> bool:
    """严格参数匹配：每个参数都要求 type_compatible + values_match。

    用于 exact match 判定（replay 释放 observation、reward exact_success）。
    """
    if set(agent_args.keys()) != set(gt_args.keys()):
        return False
    for k in gt_args:
        if not strict_values_match(agent_args[k], gt_args[k]):
            return False
    return True


def _args_match(agent_args: dict, gt_args: dict) -> bool:
    """AST 宽松匹配：比较两个参数字典。

    使用 values_match 进行递归值比较，支持嵌套 dict/list、
    数值/字符串归一化、大小写不敏感等语义。
    """
    if set(agent_args.keys()) != set(gt_args.keys()):
        return False
    for k in gt_args:
        if not values_match(agent_args[k], gt_args[k]):
            return False
    return True


def map_enum_values(
    func_name: str,
    args: dict,
    enum_map: dict,
) -> dict:
    """对参数值做 enum 映射，同时检查合法性。

    enum_map 支持两种格式：
    1. 精确格式（推荐）: {func_name: {param_name: {perturbed_val: original_val}}}
    2. 扁平格式（兼容旧数据）: {perturbed_val: original_val}

    映射规则：
    - 模型输出 perturbed enum value → 映射回 original（判对）
    - 模型输出 original enum value（在 perturbed schema 下非法）→ 标记为无效（判错）
    - 非 enum 参数值不受影响

    Args:
        func_name: 当前函数名（已 resolve 为 original name）。
        args: 模型输出的参数字典。
        enum_map: enum 映射表。

    Returns:
        映射后的参数字典。
    """
    if not enum_map:
        return args

    # 判断格式：如果任一 top-level value 是 dict，则为精确格式（nested）
    # 注意：不能只看第一个 value，防止混合格式导致 _map_enum_flat 收到不可哈希的 dict
    has_nested = any(isinstance(v, dict) for v in enum_map.values()) if enum_map else False

    if has_nested:
        return _map_enum_nested(func_name, args, enum_map)
    else:
        return _map_enum_flat(args, enum_map)


def _map_enum_nested(func_name: str, args: dict, enum_map: dict) -> dict:
    """精确格式：按 func_name + param_name 限定映射范围。"""
    func_enums = enum_map.get(func_name, {})
    if not func_enums:
        return args

    mapped = {}
    for k, v in args.items():
        param_enums = func_enums.get(k)
        if not param_enums:
            mapped[k] = v
            continue

        v_str = str(v)
        # reverse: {original_val: perturbed_val}
        reverse = {}
        for pert, orig in param_enums.items():
            try:
                hash(orig)
                reverse[orig] = pert
            except TypeError:
                pass

        if v_str in param_enums:
            # 合法 perturbed value → 映射回 original
            mapped[k] = param_enums[v_str]
        elif v_str in reverse:
            # 模型输出了 original value，perturbed schema 下非法
            mapped[k] = f"__INVALID_ENUM_{v_str}__"
        else:
            mapped[k] = v
    return mapped


def _map_enum_flat(args: dict, enum_map: dict) -> dict:
    """扁平格式（兼容旧数据）：全局映射 + 合法性检查。"""
    # P1-fix: 过滤不可哈希的 value（如 dict），这些值无法作为反向索引的 key
    reverse = {}
    for k, v in enum_map.items():
        try:
            hash(v)
            reverse[v] = k
        except TypeError:
            pass

    mapped = {}
    for k, v in args.items():
        v_str = str(v)
        if v_str in enum_map:
            # 合法 perturbed value → 映射回 original
            mapped[k] = enum_map[v_str]
        elif v_str in reverse:
            # original value 在 perturbed schema 下非法
            mapped[k] = f"__INVALID_ENUM_{v_str}__"
        else:
            mapped[k] = v
    return mapped
