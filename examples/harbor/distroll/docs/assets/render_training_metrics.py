"""Render two offline metric figures from frozen, de-identified numeric data."""

import json
import xml.etree.ElementTree as ET
from html import escape
from pathlib import Path

import gi

gi.require_version("Rsvg", "2.0")
from gi.repository import Rsvg

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "training_metrics.json").read_text())
LAST_STEP = DATA["steps"][-1]
BLUE, TEAL, GRAY = "#3569C8", "#188579", "#64748B"
ET.register_namespace("", "http://www.w3.org/2000/svg")


def text(x, y, value, size=18, color="#334155", anchor="start"):
    return f'<text x="{x}" y="{y}" font-size="{size}" fill="{color}" text-anchor="{anchor}">{escape(str(value))}</text>'


def line(x1, y1, x2, y2, color="#E2E8F0", dash=""):
    return f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="1.3" stroke-dasharray="{dash}"/>'


def save(name, title, subtitle, body, translations, height=650):
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="{height}" viewBox="0 0 1320 {height}">'
        '<rect width="100%" height="100%" fill="#FFFFFF"/>'
        '<g font-family="Noto Sans CJK SC, sans-serif">'
        + text(48, 55, title, 29, "#1C2D47")
        + text(48, 91, subtitle, 17, GRAY)
        + "".join(body)
        + "</g></svg>"
    )
    chinese = ET.fromstring(svg)
    for node in chinese.iter("{http://www.w3.org/2000/svg}text"):
        if not node.text.replace(".", "", 1).isdigit():
            node.text = translations[node.text]
    for filename, document in [
        (name, svg),
        (name + ".zh-CN", ET.tostring(chinese, encoding="unicode")),
    ]:
        (ROOT / f"{filename}.svg").write_text(document)
        raster = document.replace(
            f'width="1320" height="{height}"',
            f'width="1980" height="{int(height * 1.5)}"',
            1,
        )
        Rsvg.Handle.new_from_data(raster.encode()).get_pixbuf().savev(
            str(ROOT / f"{filename}.png"), "png", [], []
        )


def ema(values):
    weighted, weight_sum = 0.0, 0.0
    output = []
    weight = DATA["smoothing"]
    for value in values:
        weighted = weight * weighted + value
        weight_sum = weight * weight_sum + 1
        output.append(weighted / weight_sum)
    return output


def axes(body, x, y, width, height, ymin, ymax, ticks, label):
    px = lambda step: x + width * step / LAST_STEP
    py = lambda value: y + height * (ymax - value) / (ymax - ymin)
    for value in ticks:
        body.extend(
            [
                line(x, py(value), x + width, py(value)),
                text(x - 14, py(value) + 6, f"{value:g}", 16, GRAY, "end"),
            ]
        )
    ticks_x = list(range(0, LAST_STEP + 1, 10))
    if LAST_STEP - ticks_x[-1] < 5:
        ticks_x[-1] = LAST_STEP
    else:
        ticks_x.append(LAST_STEP)
    for step in ticks_x:
        body.extend(
            [
                line(px(step), y, px(step), y + height, "#F1F5F9"),
                text(px(step), y + height + 28, step, 16, GRAY, "middle"),
            ]
        )
    body.extend(
        [
            text(x, y - 16, label, 18),
            text(x + width / 2, y + height + 57, "Training step", 17, GRAY, "middle"),
        ]
    )
    return px, py


def curve(body, values, px, py, color):
    for points, alpha, width in [(values, 0.22, 1.6), (ema(values), 1, 3)]:
        d = " ".join(
            f"{'M' if i == 0 else 'L'}{px(step):.2f},{py(value):.2f}"
            for i, (step, value) in enumerate(zip(DATA["steps"], points))
        )
        body.append(
            f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" opacity="{alpha}" stroke-linejoin="round"/>'
        )


def legend(body, x, y, label, color):
    body.extend([line(x, y - 6, x + 28, y - 6, color), text(x + 38, y, label, 17)])


body = []
px, py = axes(
    body, 95, 180, 1170, 750, 0.4, 1, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], "Mean reward"
)
curve(body, DATA["candidate_reward"], px, py, BLUE)
curve(body, DATA["selected_reward"], px, py, TEAL)
legend(body, 95, 135, "Valid candidates", BLUE)
legend(body, 450, 135, "Selected training trajectories", TEAL)
body.append(
    text(
        48,
        1060,
        "Light: per-step values. Dark: debiased EMA (0.85). Vertical axis: 0.4-1.0.",
        17,
        GRAY,
    )
)
save(
    "reward",
    "Reward: candidates and selected trajectories",
    f"Steps 0-{LAST_STEP}. Both curves exclude failed slots and zero-gradient placeholders.",
    body,
    translations={
        "Reward: candidates and selected trajectories": "奖励：有效候选轨迹与入选训练轨迹",
        f"Steps 0-{LAST_STEP}. Both curves exclude failed slots and zero-gradient placeholders.": f"训练步0–{LAST_STEP}；两条曲线均排除失败slot和零梯度占位。",
        "Mean reward": "平均奖励",
        "Training step": "训练步",
        "Valid candidates": "有效候选轨迹",
        "Selected training trajectories": "入选训练轨迹",
        "Light: per-step values. Dark: debiased EMA (0.85). Vertical axis: 0.4-1.0.": "浅线：逐步原值；深线：去偏EMA（0.85）；纵轴范围0.4–1.0。",
    },
    height=1100,
)

body = []
px, py = axes(body, 95, 180, 1170, 750, 0, 18, [0, 4, 8, 12, 16], "Selected groups")
body.append(line(95, py(16), 1265, py(16), "#94A3B8", "6 5"))
curve(body, DATA["selected_groups"], px, py, BLUE)
body.extend(
    [
        text(48, 133, "Dashed reference: target of 16 groups per step", 17, GRAY),
        text(48, 1060, "Light: per-step values. Dark: debiased EMA (0.85).", 17, GRAY),
    ]
)
save(
    "selected-groups",
    "Groups used for each training update",
    "Up to 32 candidate groups, 8 trajectories per group, target of 16 qualified groups.",
    body,
    translations={
        "Groups used for each training update": "每次参数更新使用的group数量",
        "Up to 32 candidate groups, 8 trajectories per group, target of 16 qualified groups.": "最多32个候选组，每组8条轨迹，目标为16个合格组。",
        "Selected groups": "入选group数",
        "Training step": "训练步",
        "Dashed reference: target of 16 groups per step": "虚线：每步目标16组",
        "Light: per-step values. Dark: debiased EMA (0.85).": "浅线：逐步原值；深线：去偏EMA（0.85）。",
    },
    height=1100,
)
print("Rendered both metric figures in English and Chinese from frozen data.")
