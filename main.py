"""真实DEM约束层入口。

9.22优先：
- Simulate仅两级FSS，不访问DEM。
- Expand执行FSS、DEM和可执行的线路结构检查。
- 未提供桥隧清单不代表无桥隧。
- 理想坡度12‰，硬约束纵坡20‰。
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.windows import Window

from economics import (
    evaluate_cost, evaluate_soft, number, read_inventory,
)

@dataclass(frozen=True)
class Point:
    x: float
    y: float
    z: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in (self.x, self.y, self.z)):
            raise ValueError("节点坐标和高程必须为有限数")

def distance(a, b):
    return math.hypot(b.x - a.x, b.y - a.y)

def grade(a, b):
    length = distance(a, b)
    if length <= 1e-8:
        raise ValueError("相邻节点平面位置重合")
    return (b.z - a.z) * 1000.0 / length

def turn_angle(a, b, c):
    """直行为0，掉头为π。"""
    ux, uy = b.x - a.x, b.y - a.y
    vx, vy = c.x - b.x, c.y - b.y
    if math.hypot(ux, uy) <= 1e-8 or math.hypot(vx, vy) <= 1e-8:
        raise ValueError("重复节点无法计算转角")
    return math.atan2(abs(ux * vy - uy * vx), ux * vx + uy * vy)

class DEM:
    """米制投影DEM，按像元缓存；高程单位要求为米。"""

    def __init__(self, path, expected_crs=None):
        self.src = rasterio.open(path)
        self.cache = {}
        self.hits = 0
        self.misses = 0
        try:
            if self.src.crs is None:
                raise ValueError("DEM缺少CRS")
            crs = CRS.from_user_input(self.src.crs)
            if not crs.is_projected or len(crs.axis_info) < 2:
                raise ValueError("DEM必须是投影坐标系")
            if any(
                abs(axis.unit_conversion_factor - 1.0) > 1e-9
                for axis in crs.axis_info[:2]
            ):
                raise ValueError("DEM水平坐标单位必须为米")
            if expected_crs and crs != CRS.from_user_input(expected_crs):
                raise ValueError("DEM坐标系与配置不一致")
        except Exception:
            self.src.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.src.close()

    def cell(self, x, y):
        row, col = self.src.index(x, y)
        return int(row), int(col)

    def contains(self, x, y):
        row, col = self.cell(x, y)
        return 0 <= row < self.src.height and 0 <= col < self.src.width

    def sample(self, x, y):
        key = self.cell(x, y)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]

        self.misses += 1
        row, col = key
        value = None
        if 0 <= row < self.src.height and 0 <= col < self.src.width:
            raw = self.src.read(
                1, window=Window(col, row, 1, 1), masked=True
            )[0, 0]
            if not np.ma.is_masked(raw):
                candidate = (
                    float(raw) * self.src.scales[0] + self.src.offsets[0]
                )
                if math.isfinite(candidate):
                    value = candidate

        self.cache[key] = value
        return value

    def statistics(self):
        minimum, maximum, count = None, None, 0
        for _, window in self.src.block_windows(1):
            values = self.src.read(
                1, window=window, masked=True
            ).compressed().astype(float)
            values = values * self.src.scales[0] + self.src.offsets[0]
            values = values[np.isfinite(values)]
            if values.size:
                lo, hi = float(values.min()), float(values.max())
                minimum = lo if minimum is None else min(minimum, lo)
                maximum = hi if maximum is None else max(maximum, hi)
                count += int(values.size)
        return {
            "min_m": minimum,
            "max_m": maximum,
            "valid_pixel_count": count,
        }

    def metadata(self):
        nodata = self.src.nodata
        return {
            "path": str(self.src.name),
            "crs": str(self.src.crs),
            "width": self.src.width,
            "height": self.src.height,
            "resolution_m": list(self.src.res),
            "nodata": (
                float(nodata)
                if nodata is not None and math.isfinite(float(nodata))
                else None
            ),
            "note": "NoData未声明不等于地理范围内所有区域都有可靠高程",
        }

def coarse_check(start, end, parent, candidate, generated, cfg):
    """两级粗筛：平面不通过时不再计算纵断面；不访问DEM。"""
    generated = number(generated, "已生成长度")
    length = distance(parent, candidate)
    if length <= 1e-8:
        raise ValueError("父节点和候选节点重合")

    residual = (
        cfg["lambda_max"] * distance(start, end)
        - generated - length - distance(candidate, end)
    )
    if residual < -1e-6:
        return {
            "passed": False,
            "failed_stage": "planar_fss",
            "planar_residual_m": residual,
            "profile": None,
        }

    g = cfg["max_grade_permille"] / 1000.0
    remaining = cfg["lambda_max"] * distance(candidate, end)
    lower = max(parent.z - g * length, end.z - g * remaining)
    upper = min(parent.z + g * length, end.z + g * remaining)
    passed = lower - 1e-6 <= candidate.z <= upper + 1e-6
    return {
        "passed": passed,
        "failed_stage": None if passed else "profile_fss",
        "planar_residual_m": residual,
        "profile": {"lower_m": lower, "upper_m": upper},
    }

def positive_linear_integral(a, b, width):
    """分段线性高差的正值积分，正确处理跨零区间。"""
    if a >= 0 and b >= 0:
        return (a + b) * width / 2.0
    if a <= 0 and b <= 0:
        return 0.0
    positive = max(a, b)
    return width * positive * positive / (2.0 * abs(b - a))

def terrain_check(dem, a, b, interval):
    """纵向高差面积单位为m²，不能代替土方体积m³。"""
    length = distance(a, b)
    count = max(2, int(math.ceil(length / interval)) + 1)
    terrain, differences = [], []

    for index in range(count):
        t = index / (count - 1)
        ground = dem.sample(
            a.x + t * (b.x - a.x),
            a.y + t * (b.y - a.y),
        )
        if ground is None:
            return {"passed": False, "reason": "DEM越界或无效像元"}
        terrain.append(ground)
        differences.append(a.z + t * (b.z - a.z) - ground)

    step = length / (count - 1)
    positive = sum(
        positive_linear_integral(x, y, step)
        for x, y in zip(differences, differences[1:])
    )
    negative = sum(
        positive_linear_integral(-x, -y, step)
        for x, y in zip(differences, differences[1:])
    )
    return {
        "passed": True,
        "sample_count": count,
        "terrain_min_m": min(terrain),
        "terrain_max_m": max(terrain),
        "difference_min_m": min(differences),
        "difference_max_m": max(differences),
        "positive_difference_area_m2": positive,
        "negative_difference_area_m2": negative,
        "fill_volume_m3": None,
        "cut_volume_m3": None,
        "note": "缺少横断面和工程分类；高差不能直接确定桥隧或土方体积",
    }

def grade_sections(points):
    """合并同坡度的采样段，避免将500m节点间距当成设计坡段。"""
    result = []
    for a, b in zip(points, points[1:]):
        length, g = distance(a, b), grade(a, b)
        if result and abs(g - result[-1]["grade_permille"]) <= 1e-6:
            result[-1]["length_m"] += length
        else:
            result.append({"grade_permille": g, "length_m": length})
    return result

def route_structure_checks(points, cfg, structures):
    groups = grade_sections(points)
    violations, pending = [], []

    for index, section in enumerate(groups):
        if section["length_m"] < cfg["min_grade_length_m"] - 1e-6:
            violations.append(f"坡段{index}长度不足")
        if abs(section["grade_permille"]) > cfg["max_grade_permille"] + 1e-6:
            violations.append(f"坡段{index}纵坡超限")

    for previous, current in zip(groups, groups[1:]):
        delta = abs(current["grade_permille"] - previous["grade_permille"])
        if delta > cfg["max_grade_delta_permille"] + 1e-6:
            violations.append("相邻坡段变坡差超限")

    turns = []
    for index in range(1, len(points) - 1):
        angle = turn_angle(points[index - 1], points[index], points[index + 1])
        turns.append({
            "node_index": index,
            "angle_rad": angle,
            "is_straight": angle <= 1e-7,
            "radius_m": None,
        })

    if any(not item["is_straight"] for item in turns):
        pending.append("缺少平面曲线要素，半径/圆曲线长/夹直线长未验证")
    if len(groups) > 1:
        pending.append("存在变坡点，竖曲线衔接尚未验证")

    if structures is None:
        pending.append("桥隧情况未知；缺少完整清单，连续隧道长度未验证")
    else:
        for item in structures:
            if item["type"] == "tunnel":
                if item["length_m"] > cfg["max_tunnel_length_m"] + 1e-6:
                    violations.append(f"隧道{item['id']}长度超限")

    pending.append("未接入禁区、交叉净空、地质和环境空间数据")
    return {
        "violations": violations,
        "pending_checks": pending,
        "grade_sections": groups,
        "turns": turns,
    }

class ConstraintEngine:
    """完整路线核查接口。

    提供结果缓存，但尚非持有增量状态的完整MCTS扩展器。
    配置在实例生命周期内固定；返回深拷贝避免调用方污染缓存。
    """

    def __init__(self, dem, cfg, start, end):
        self.dem = dem
        self.cfg = dict(cfg)
        self.start = start
        self.end = end
        self.expansion_cache = {}
        self.cache_hits = 0

    def simulate(self, parent, candidate, generated):
        return coarse_check(
            self.start, self.end, parent, candidate, generated, self.cfg
        )

    def expand(self, points, structures):
        if len(points) < 2:
            raise ValueError("线路至少需要两个节点")
        if points[0] != self.start or points[-1] != self.end:
            raise ValueError("当前expand核查完整路线，首尾必须匹配实例起终点")

        key = (
            tuple(points),
            json.dumps(structures, sort_keys=True, allow_nan=False),
        )
        if key in self.expansion_cache:
            self.cache_hits += 1
            return copy.deepcopy(self.expansion_cache[key])

        segments, violations = [], []
        generated = 0.0

        for index, (a, b) in enumerate(zip(points, points[1:])):
            length = distance(a, b)
            coarse = self.simulate(a, b, generated)
            segment = {
                "index": index,
                "length_m": length,
                "grade_permille": grade(a, b),
                "coarse": coarse,
                "dem": None,
            }

            if not coarse["passed"]:
                violations.append(f"段{index}未通过{coarse['failed_stage']}")
            elif not self.dem.contains(a.x, a.y) or not self.dem.contains(b.x, b.y):
                violations.append(f"段{index}端点越界")
            else:
                fine = terrain_check(self.dem, a, b, self.cfg["dem_interval_m"])
                segment["dem"] = fine
                if not fine["passed"]:
                    violations.append(f"段{index}DEM无效")

            segments.append(segment)
            generated += length
            if violations:
                break

        if violations:
            structure = {
                "violations": [],
                "pending_checks": ["前置筛选失败，后续结构检查跳过"],
                "grade_sections": None,
            }
        else:
            structure = route_structure_checks(points, self.cfg, structures)

        violations.extend(structure["violations"])
        pending = structure["pending_checks"]
        result = {
            "checked_constraints_passed": not violations,
            "complete": not pending,
            "feasible": False if violations else (None if pending else True),
            "violations": violations,
            "pending_checks": pending,
            "segments": segments,
            "route_structure": structure,
        }
        self.expansion_cache[key] = copy.deepcopy(result)
        return result

def cnn_prior(parent, child, dem, probability_at, type_at, floor=0.01):
    """9.22：沿线10点取地形所需工程类型对应通道，设置概率下限。"""
    if probability_at is None or type_at is None:
        return {
            "available": False,
            "value": None,
            "reason": "缺少CNN查询器或地形工程分类器",
        }

    floor = number(floor, "先验下限")
    if not 0 < floor <= 1:
        raise ValueError("先验下限必须位于(0,1]")

    channels = {"roadbed": 0, "bridge": 1, "tunnel": 2}
    values = []

    for index in range(10):
        t = index / 9.0
        x = parent.x + t * (child.x - parent.x)
        y = parent.y + t * (child.y - parent.y)
        z = parent.z + t * (child.z - parent.z)
        ground = dem.sample(x, y)
        if ground is None:
            return {"available": False, "value": None, "reason": "DEM无效"}

        kind = type_at(x, y, z, ground)
        if kind not in channels:
            return {"available": False, "value": None, "reason": "工程类型未知"}

        raw = probability_at(x, y)
        if raw is None or len(raw) != 3:
            raise ValueError("CNN必须返回三个概率")
        probabilities = [number(v, "CNN概率") for v in raw]
        if any(v > 1 for v in probabilities):
            raise ValueError("CNN概率不能大于1")
        if not math.isclose(sum(probabilities), 1.0, abs_tol=1e-5):
            raise ValueError("CNN三通道概率之和必须为1")
        values.append(max(probabilities[channels[kind]], floor))

    return {"available": True, "value": sum(values) / 10, "sample_count": 10}

def uct_score(q_min, parent_visits, child_visits, cp, prior, initial_cost):
    """未访问节点使用同一公式，避免无穷优先级掩盖先验。"""
    if parent_visits < 1 or child_visits < 0:
        raise ValueError("父访问数至少1，子访问数不能为负")
    q = number(initial_cost if q_min is None else q_min, "成本估值")
    cp = number(cp, "探索系数")
    p = number(prior, "先验")
    if p > 1:
        raise ValueError("先验不能大于1")
    return -q + cp * max(p, 0.01) * math.sqrt(
        math.log1p(parent_visits)
    ) / (1 + child_visits)

def read_route(config, dem):
    raw = config.get("route")
    if raw is not None:
        if not isinstance(raw, list) or len(raw) < 2:
            raise ValueError("route必须至少包含两个节点")
        return [
            Point(float(p["x"]), float(p["y"]), float(p["z"]))
            for p in raw
        ], "configured_route"

    auto = config["auto_route"]
    length = number(auto["length_m"], "测试线长度", 1.0)
    spacing = number(auto["node_spacing_m"], "节点间距", 1.0)
    bounds = dem.src.bounds
    cx = (bounds.left + bounds.right) / 2.0
    cy = (bounds.bottom + bounds.top) / 2.0
    x0, x1 = cx - length / 2, cx + length / 2
    z0, z1 = dem.sample(x0, cy), dem.sample(x1, cy)
    if z0 is None or z1 is None:
        raise ValueError("自动路线端点DEM无效，请配置有效范围内的route")

    count = max(2, int(math.ceil(length / spacing)) + 1)
    points = []
    for index in range(count):
        t = index / (count - 1)
        points.append(Point(x0 + t * length, cy, z0 + t * (z1 - z0)))
    return points, "dem_smoke_test"

def write_json(path, value):
    """输出严格JSON，不允许NaN或Infinity。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("config/project.json"))
    parser.add_argument("--scan-stats", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    root = (
        config_path.parent.parent
        if config_path.parent.name.lower() == "config"
        else config_path.parent
    )

    cfg = config["constraints"]
    expected = {
        "lambda_max", "max_grade_permille", "max_grade_delta_permille",
        "min_grade_length_m", "min_curve_radius_m", "min_curve_length_m",
        "min_tangent_length_m", "max_tunnel_length_m", "dem_interval_m",
    }
    if set(cfg) != expected:
        raise ValueError(f"constraints字段应为{sorted(expected)}")
    cfg = {key: number(value, key, 1e-9) for key, value in cfg.items()}
    if cfg["lambda_max"] < 1:
        raise ValueError("lambda_max不能小于1")

    ideal = number(
        config["soft_constraints"]["ideal_grade_permille"], "理想坡度"
    )
    if ideal > cfg["max_grade_permille"]:
        raise ValueError("理想坡度不能超过硬约束纵坡上限")

    engineering = config["engineering"]
    if not isinstance(engineering, dict):
        raise ValueError("engineering必须是对象")

    with DEM(root / config["dem"]["path"], config["dem"].get("crs")) as dem:
        points, mode = read_route(config, dem)
        lengths = [distance(a, b) for a, b in zip(points, points[1:])]
        if any(length <= 1e-8 for length in lengths):
            raise ValueError("线路存在相邻重复节点")
        if distance(points[0], points[-1]) <= 1e-8:
            raise ValueError("起终点重合")

        total = sum(lengths)
        structures = read_inventory(engineering, total)
        engine = ConstraintEngine(dem, cfg, points[0], points[-1])

        before = dem.hits + dem.misses
        simulate = []
        generated = 0.0
        for a, b, length in zip(points, points[1:], lengths):
            check = engine.simulate(a, b, generated)
            simulate.append(check)
            generated += length
            if not check["passed"]:
                break
        simulate_dem_queries = dem.hits + dem.misses - before

        expand = engine.expand(points, structures)
        cost = evaluate_cost(total, engineering, config["pricing"])
        soft = evaluate_soft(
            engineering,
            config["soft_constraints"],
            grade_sections(points),
        )

        pure = cost["pure_cost_yuan"]
        penalty = soft["soft_penalty_yuan"]
        search_score = (
            pure + penalty
            if pure is not None and penalty is not None
            and expand["checked_constraints_passed"]
            else None
        )

        result = {
            "schema_version": "4.0",
            "status": "completed",
            "task": "railway_constraint_layer",
            "evaluation_mode": mode,
            "tested_dem": dem.metadata(),
            "route": {
                "points": [asdict(p) for p in points],
                "node_count": len(points),
                "segment_count": len(lengths),
                "length_m": total,
                "extension_factor": total / distance(points[0], points[-1]),
            },
            "engineering_inventory": {
                "status": engineering.get("structures_status", "unknown"),
                "structures": structures,
            },
            "simulate": {
                "passed": all(item["passed"] for item in simulate),
                "dem_query_count": simulate_dem_queries,
                "segments": simulate,
            },
            "expand": expand,
            "cost": cost,
            "soft_constraints": soft,
            "scoring": {
                "pure_cost_yuan": pure,
                "soft_penalty_yuan": penalty,
                "search_score_yuan": search_score,
                "engineering_feasibility_certified": expand["feasible"] is True,
            },
            "cnn_prior": {
                "available": False,
                "value": None,
                "reason": "尚未接入实际CNN查询器和地形工程分类器",
            },
            "dem_statistics": dem.statistics() if args.scan_stats else None,
            "cache": {
                "sample_cache_size": len(dem.cache),
                "cache_hits": dem.hits,
                "cache_misses": dem.misses,
                "expansion_cache_size": len(engine.expansion_cache),
                "expansion_cache_hits": engine.cache_hits,
            },
            "feature_status": {
                "fss": "implemented",
                "dem_sampling": "implemented",
                "grade_sections": "implemented",
                "formal_horizontal_curves": "pending_alignment_elements",
                "formal_vertical_curves": "pending_alignment_elements",
                "cnn_prior": "callable_interface_only",
                "uct": "callable_interface_only",
                "mcts": "not_implemented",
                "llm_controller": "not_implemented",
                "bridge_height_100m_hard_rejection": False,
                "dead_end_huge_penalty": False,
            },
        }

    output = root / config["output"]
    write_json(output, result)
    print(f"测试执行完成：{output}")
    print(f"测试模式：{mode}")
    print(f"桥隧清单状态：{result['engineering_inventory']['status']}")
    print(f"理想坡度：{ideal:g}‰")
    print(f"Simulate通过：{result['simulate']['passed']}")
    print(f"已执行精筛通过：{expand['checked_constraints_passed']}")
    print(f"完整工程可行性：{expand['feasible']}")
    print(f"纯造价：{pure}")
    print(f"软惩罚：{penalty}")
    print(f"DEM缓存像元数：{result['cache']['sample_cache_size']}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())