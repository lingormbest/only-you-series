"""工程清单、纯造价和软约束。

金额统一为元。
未知桥隧清单输出 null，不输出空列表。
理想坡度用于计算超出量；没有单价依据时不计算坡度货币惩罚。
"""

from __future__ import annotations

import math

def number(value, name, minimum=0.0):
    """拒绝布尔值、非有限数和低于下限的数。"""
    if isinstance(value, bool):
        raise ValueError(f"{name}不能是布尔值")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name}必须是数值") from exc
    if not math.isfinite(result) or result < minimum:
        raise ValueError(f"{name}必须是 >= {minimum} 的有限数")
    return result

def interpolate(x, knots):
    """连续分段线性插值；缺少节点或超出覆盖范围时返回None。"""
    if knots is None:
        return None
    if not isinstance(knots, list) or len(knots) < 2:
        raise ValueError("插值节点至少需要两个[自变量,单价]")

    pairs = []
    for item in knots:
        if not isinstance(item, list) or len(item) != 2:
            raise ValueError("插值节点必须为[自变量,单价]")
        pairs.append((
            number(item[0], "插值自变量"),
            number(item[1], "插值单价"),
        ))

    if any(b[0] <= a[0] for a, b in zip(pairs, pairs[1:])):
        raise ValueError("插值自变量必须严格递增")
    if not pairs[0][0] <= x <= pairs[-1][0]:
        return None

    for (x0, y0), (x1, y1) in zip(pairs, pairs[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pairs[-1][1]

def bridge_price(length, height, surface):
    """桥长方向插值后，再沿桥高方向插值。

    surface:
    [
      {"height_m": ..., "length_price_knots": [[长度,单价], ...]},
      ...
    ]
    """
    if surface is None:
        return None
    if not isinstance(surface, list) or len(surface) < 2:
        raise ValueError("桥梁单价面至少需要两个高度层")

    rows = []
    for layer in surface:
        h = number(layer["height_m"], "桥梁计价高度")
        price = interpolate(length, layer["length_price_knots"])
        if price is None:
            return None
        rows.append([h, price])
    return interpolate(height, rows)

def validate_structures(raw, route_length):
    """校验完整桥隧清单。

    None表示未知；非空列表中的每一项代表一个完整单体工程。
    空列表只有在read_inventory中取得明确状态后才代表确认无桥隧。
    """
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("structures必须是列表或null")

    result = []
    ids = set()
    last_end = 0.0

    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("工程记录必须是对象")
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier or identifier in ids:
            raise ValueError("每项工程必须有唯一非空字符串id")
        ids.add(identifier)

        kind = item.get("type")
        if kind not in ("bridge", "tunnel"):
            raise ValueError("工程类型必须是bridge或tunnel")

        start = number(item["start_m"], "工程起点")
        end = number(item["end_m"], "工程终点")
        if end <= start or start < last_end - 1e-6:
            raise ValueError("工程里程必须有序且互不重叠")
        if end > route_length + 1e-6:
            raise ValueError("工程终点超出线路")

        if result and kind == result[-1]["type"] == "tunnel":
            if abs(start - last_end) <= 1e-6:
                raise ValueError("相接的连续隧道必须合并，不能拆分绕过长度检查")

        height = item.get("height_m")
        if height is not None:
            height = number(height, "桥梁计价高度")

        result.append({
            "id": identifier,
            "type": kind,
            "start_m": start,
            "end_m": end,
            "length_m": end - start,
            "height_m": height,
        })
        last_end = end

    return result

def read_inventory(engineering, route_length):
    """用明确状态区分未知、有工程、已确认无工程。"""
    status = engineering.get("structures_status", "unknown")
    raw = engineering.get("structures")

    if status == "unknown":
        if raw is not None:
            raise ValueError("structures_status=unknown时structures必须为null")
        return None

    if status == "provided":
        if not isinstance(raw, list) or not raw:
            raise ValueError("provided必须对应非空完整桥隧清单")
        return validate_structures(raw, route_length)

    if status == "confirmed_absent":
        if raw != []:
            raise ValueError("confirmed_absent必须显式提供空列表")
        return []

    raise ValueError("structures_status必须是unknown/provided/confirmed_absent")

def evaluate_cost(length, engineering, pricing):
    """计算已知分项；只有所有必需分项齐全才输出纯造价。"""
    length = number(length, "线路长度")
    structures = read_inventory(engineering, length)
    quantities = engineering.get("quantities")
    if not isinstance(quantities, dict):
        raise ValueError("engineering.quantities必须是对象")

    components = {}
    missing = []

    def add(name, quantity, unit_price):
        q = None if quantity is None else number(quantity, f"{name}工程量")
        unit = (
            None if unit_price is None
            else number(unit_price, f"{name}单价")
        )
        if q is None:
            components[name] = None
            missing.append(f"{name}:缺少工程量")
        elif q == 0:
            components[name] = 0.0
        elif unit is None:
            components[name] = None
            missing.append(f"{name}:缺少单价")
        else:
            components[name] = q * unit

    add("track", length, pricing.get("track_yuan_per_m"))
    add("land", quantities.get("land_area_m2"), pricing.get("land_yuan_per_m2"))
    add("fill", quantities.get("fill_volume_m3"), pricing.get("fill_yuan_per_m3"))
    add("cut", quantities.get("cut_volume_m3"), pricing.get("cut_yuan_per_m3"))

    # 缺少清单时明细本身就是未知，不能初始化成[]造成误解。
    details = None if structures is None else []

    for kind in ("bridge", "tunnel"):
        if structures is None:
            components[kind] = None
            missing.append(f"{kind}:缺少完整单体工程清单")
            continue

        values = []
        for item in structures:
            if item["type"] != kind:
                continue

            if kind == "tunnel":
                unit = interpolate(
                    item["length_m"],
                    pricing.get("tunnel_length_price_knots"),
                )
            elif item["height_m"] is None:
                unit = None
            else:
                unit = bridge_price(
                    item["length_m"], item["height_m"],
                    pricing.get("bridge_price_surface"),
                )

            cost = None if unit is None else item["length_m"] * unit
            values.append(cost)
            details.append({
                "id": item["id"],
                "type": kind,
                "length_m": item["length_m"],
                "unit_price_yuan_per_m": unit,
                "cost_yuan": cost,
                "status": "missing_price_or_geometry" if cost is None else "computed",
            })

        if any(value is None for value in values):
            components[kind] = None
            missing.append(f"{kind}:缺少计价高度/连续单价节点，或超出插值范围")
        else:
            components[kind] = sum(values)

    # 桥台、桥墩、洞门是不同对象，数量和单价分别管理。
    for name in ("portal", "abutment", "pier"):
        count = quantities.get(f"{name}_count")
        if count is not None:
            count = number(count, f"{name}数量")
            if not count.is_integer():
                raise ValueError(f"{name}数量必须为整数")
        add(name, count, pricing.get(f"{name}_yuan_each"))

    return {
        "available": not missing,
        "pure_cost_yuan": None if missing else sum(components.values()),
        "known_components_sum_yuan": sum(
            value for value in components.values() if value is not None
        ),
        "components_yuan": components,
        "structures_status": engineering.get("structures_status", "unknown"),
        "structure_inventory": structures,
        "structure_cost_details": details,
        "missing_inputs": missing,
        "warning": "已知费用分项之和不是总造价；当前未复现论文表3",
    }

def evaluate_soft(engineering, config, grade_sections=None):
    """12‰理想坡度只计算超出量；坡度货币惩罚仍关闭。"""
    ideal = number(config["ideal_grade_permille"], "理想坡度")
    grade_metrics = {
        "status": "disabled_pending_price_basis",
        "ideal_grade_permille": ideal,
        "max_abs_grade_permille": None,
        "max_excess_permille": None,
        "excess_integral_permille_m": None,
        "penalty_yuan": 0.0,
        "note": "阈值已确定；货币惩罚按9.22关闭，不表示已经核定能耗成本",
    }

    if grade_sections:
        maximum = 0.0
        integral = 0.0
        for section in grade_sections:
            g = abs(float(section["grade_permille"]))
            if not math.isfinite(g):
                raise ValueError("坡度必须为有限数")
            length = number(section["length_m"], "坡段长度")
            maximum = max(maximum, g)
            integral += max(g - ideal, 0.0) * length
        grade_metrics.update({
            "max_abs_grade_permille": maximum,
            "max_excess_permille": max(maximum - ideal, 0.0),
            "excess_integral_permille_m": integral,
        })

    env = config["environment"]
    if not isinstance(env["enabled"], bool):
        raise ValueError("environment.enabled必须是布尔值")

    missing = []
    penalty = 0.0
    status = "disabled"
    if env["enabled"]:
        raw_length = engineering.get("environment_length_m")
        raw_price = env.get("base_price_yuan_per_m")
        if raw_length is None:
            missing.append("environment_length_m")
        if raw_price is None:
            missing.append("environment.base_price_yuan_per_m")

        threshold = number(env["threshold_m"], "环境阈值")
        alpha = number(env["alpha"], "环境偏好系数")

        if missing:
            penalty = None
            status = "missing_input"
        else:
            excess = max(number(raw_length, "环境长度") - threshold, 0.0)
            penalty = excess * alpha * number(raw_price, "环境基准单价")
            status = "computed"

    return {
        "available": not missing,
        "soft_penalty_yuan": penalty,
        "items": {
            "bridge_height": {"status": "disabled_no_double_counting"},
            "tunnel_length": {"status": "disabled_no_double_counting"},
            "route_extension": {"status": "disabled_no_double_counting"},
            "grade": grade_metrics,
            "environment": {"status": status, "penalty_yuan": penalty},
        },
        "missing_inputs": missing,
    }