import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Iterable


VEHICLE_CATEGORIES = ["bus", "car", "truck"]
DOMAINS = ["daytime", "night"]
WEATHER_BUCKETS = ["clear", "overcast", "rainy", "snowy", "partly cloudy", "foggy", "undefined"]
SCENE_BUCKETS = ["city street", "highway", "residential", "parking lot", "tunnel", "undefined"]
AREA_BANDS = [
    ("<1024", 0, 1024),
    ("1024-2303", 1024, 2304),
    ("2304-4095", 2304, 4096),
    ("4096-9215", 4096, 9216),
    ("9216-16383", 9216, 16384),
    ("16384-40000", 16384, 40001),
    (">40000", 40001, None),
]
CUMULATIVE_THRESHOLDS = [1024, 2304, 4096, 9216, 16384, 40000]
PROJECTION_THRESHOLDS = [2304, 4096, 9216]
SPLITS = ("train", "val")


def stream_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterable[dict]:
    decoder = json.JSONDecoder()
    buffer = ""
    index = 0
    started = False
    eof = False

    with path.open("r", encoding="utf-8") as handle:
        while True:
            if index >= len(buffer) and eof:
                break

            if index >= len(buffer) - 1 and not eof:
                more = handle.read(chunk_size)
                if more:
                    buffer = buffer[index:] + more
                    index = 0
                else:
                    eof = True
                    if index >= len(buffer):
                        break

            while True:
                while index < len(buffer) and buffer[index].isspace():
                    index += 1

                if not started:
                    if index >= len(buffer):
                        break
                    if buffer[index] != "[":
                        raise ValueError(f"{path} is not a JSON array.")
                    started = True
                    index += 1
                    continue

                if index >= len(buffer):
                    break
                if buffer[index] == ",":
                    index += 1
                    continue
                if buffer[index] == "]":
                    return
                break

            if index >= len(buffer):
                if eof:
                    break
                continue

            try:
                obj, next_index = decoder.raw_decode(buffer, index)
            except json.JSONDecodeError:
                if eof:
                    raise
                more = handle.read(chunk_size)
                if not more:
                    eof = True
                    continue
                buffer = buffer[index:] + more
                index = 0
                continue

            yield obj
            index = next_index

            if index > chunk_size:
                buffer = buffer[index:]
                index = 0


def normalize_domain(raw_timeofday: str | None) -> str:
    value = (raw_timeofday or "").strip().lower()
    if value == "daytime":
        return "daytime"
    if value == "night":
        return "night"
    return "night"


def normalize_weather(raw_weather: str | None) -> str:
    value = (raw_weather or "").strip().lower()
    return value if value in WEATHER_BUCKETS[:-1] else "undefined"


def normalize_scene(raw_scene: str | None) -> str:
    value = (raw_scene or "").strip().lower()
    return value if value in SCENE_BUCKETS[:-1] else "undefined"


def compute_area(box2d: dict | None) -> float | None:
    if not box2d:
        return None
    try:
        width = max(0.0, float(box2d["x2"]) - float(box2d["x1"]))
        height = max(0.0, float(box2d["y2"]) - float(box2d["y1"]))
    except (KeyError, TypeError, ValueError):
        return None
    return width * height


def area_band_label(area: float) -> str:
    for label, lower, upper in AREA_BANDS:
        if upper is None and area >= lower:
            return label
        if lower <= area < upper:
            return label
    return "undefined"


def quantile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return float(ordered[lower])
    weight = pos - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def split_projection(total: int) -> tuple[int, int, int]:
    train = int(total * 0.70)
    val = int(total * 0.15)
    test = total - train - val
    return train, val, test


def format_number(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_table(headers: list[str], rows: list[list[object]]) -> str:
    str_rows = [[format_number(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in str_rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    header_line = " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    sep_line = "-+-".join("-" * widths[idx] for idx in range(len(headers)))
    body_lines = [
        " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))
        for row in str_rows
    ]
    return "\n".join([header_line, sep_line, *body_lines])


def build_report(train_path: Path, val_path: Path) -> str:
    image_counts = Counter()
    annotation_counts_by_category = Counter()
    annotation_counts_by_domain = Counter()
    annotation_counts_by_category_domain = defaultdict(Counter)
    size_band_counts = defaultdict(lambda: defaultdict(Counter))
    cumulative_counts = defaultdict(lambda: defaultdict(Counter))
    occlusion_truncation = defaultdict(lambda: defaultdict(Counter))
    weather_counts = defaultdict(lambda: defaultdict(Counter))
    scene_counts = defaultdict(lambda: defaultdict(Counter))
    quality_counts = defaultdict(Counter)
    image_vehicle_counts = defaultdict(list)
    projection_pool = defaultdict(lambda: defaultdict(Counter))

    for split_name, json_path in [("train", train_path), ("val", val_path)]:
        for image in stream_json_array(json_path):
            image_counts[split_name] += 1

            image_attributes = image.get("attributes", {}) or {}
            domain = normalize_domain(image_attributes.get("timeofday"))
            weather = normalize_weather(image_attributes.get("weather"))
            scene = normalize_scene(image_attributes.get("scene"))

            per_image_vehicle_count = 0

            for label in image.get("labels", []) or []:
                category = (label.get("category") or "").strip().lower()
                if category not in VEHICLE_CATEGORIES:
                    continue

                per_image_vehicle_count += 1
                annotation_counts_by_category[category] += 1
                annotation_counts_by_domain[domain] += 1
                annotation_counts_by_category_domain[category][domain] += 1
                weather_counts[domain][category][weather] += 1
                scene_counts[domain][category][scene] += 1

                quality_counts[category]["manualShape_true" if label.get("manualShape") is True else "manualShape_false"] += 1
                quality_counts[category]["manualAttributes_true" if label.get("manualAttributes") is True else "manualAttributes_false"] += 1

                label_attributes = label.get("attributes", {}) or {}
                occluded = bool(label_attributes.get("occluded", False))
                truncated = bool(label_attributes.get("truncated", False))
                occlusion_truncation[domain][category]["occluded_true" if occluded else "occluded_false"] += 1
                occlusion_truncation[domain][category]["truncated_true" if truncated else "truncated_false"] += 1
                if occluded and truncated:
                    occlusion_truncation[domain][category]["both_true"] += 1

                area = compute_area(label.get("box2d"))
                if area is not None:
                    band = area_band_label(area)
                    size_band_counts[domain][category][band] += 1
                    for threshold in CUMULATIVE_THRESHOLDS:
                        if area >= threshold:
                            cumulative_counts[threshold][domain][category] += 1
                    for threshold in PROJECTION_THRESHOLDS:
                        if area >= threshold:
                            projection_pool[threshold][domain][category] += 1

            image_vehicle_counts[domain].append(per_image_vehicle_count)

    report_lines: list[str] = []
    combined_images = image_counts["train"] + image_counts["val"]

    report_lines.append("BDD100K Vehicle Annotation Report")
    report_lines.append("=" * 80)
    report_lines.append(f"Train JSON: {train_path}")
    report_lines.append(f"Val JSON:   {val_path}")
    report_lines.append("Domain rule: 'daytime' stays daytime; all non-daytime values are folded into 'night' for the required two-domain report.")

    section1_rows = []
    for category in VEHICLE_CATEGORIES:
        day_count = annotation_counts_by_category_domain[category]["daytime"]
        night_count = annotation_counts_by_category_domain[category]["night"]
        section1_rows.append([category, day_count, night_count, day_count + night_count])
    section1_rows.append([
        "ALL",
        annotation_counts_by_domain["daytime"],
        annotation_counts_by_domain["night"],
        sum(annotation_counts_by_category.values()),
    ])
    report_lines.extend([
        "",
        "SECTION 1 - Basic Counts",
        "-" * 80,
        f"Images: train={image_counts['train']} | val={image_counts['val']} | combined={combined_images}",
        render_table(["category", "daytime", "night", "total"], section1_rows),
    ])

    section2_band_rows = []
    band_headers = ["domain", "category"] + [label for label, _, _ in AREA_BANDS]
    for domain in DOMAINS:
        for category in VEHICLE_CATEGORIES:
            row = [domain, category]
            for label, _, _ in AREA_BANDS:
                row.append(size_band_counts[domain][category][label])
            section2_band_rows.append(row)

    section2_cumulative_rows = []
    cumulative_headers = ["threshold"] + [f"{category}_{domain}" for domain in DOMAINS for category in VEHICLE_CATEGORIES]
    for threshold in CUMULATIVE_THRESHOLDS:
        row = [f">={threshold}"]
        for domain in DOMAINS:
            for category in VEHICLE_CATEGORIES:
                row.append(cumulative_counts[threshold][domain][category])
        section2_cumulative_rows.append(row)

    report_lines.extend([
        "",
        "SECTION 2 - Area Distribution Per Category Per Domain",
        "-" * 80,
        render_table(band_headers, section2_band_rows),
        "",
        "Cumulative survival counts by threshold",
        render_table(cumulative_headers, section2_cumulative_rows),
    ])

    section3_rows = []
    for domain in DOMAINS:
        for category in VEHICLE_CATEGORIES:
            counters = occlusion_truncation[domain][category]
            section3_rows.append([
                domain,
                category,
                counters["occluded_true"],
                counters["occluded_false"],
                counters["truncated_true"],
                counters["truncated_false"],
                counters["both_true"],
            ])
    report_lines.extend([
        "",
        "SECTION 3 - Occlusion And Truncation Breakdown",
        "-" * 80,
        render_table(
            ["domain", "category", "occluded_true", "occluded_false", "truncated_true", "truncated_false", "both_true"],
            section3_rows,
        ),
    ])

    section4_rows = []
    for domain in DOMAINS:
        for category in VEHICLE_CATEGORIES:
            row = [domain, category]
            for weather in WEATHER_BUCKETS:
                row.append(weather_counts[domain][category][weather])
            section4_rows.append(row)
    report_lines.extend([
        "",
        "SECTION 4 - Weather Distribution Per Category Per Domain",
        "-" * 80,
        render_table(["domain", "category", *WEATHER_BUCKETS], section4_rows),
    ])

    section5_rows = []
    for domain in DOMAINS:
        for category in VEHICLE_CATEGORIES:
            row = [domain, category]
            for scene in SCENE_BUCKETS:
                row.append(scene_counts[domain][category][scene])
            section5_rows.append(row)
    report_lines.extend([
        "",
        "SECTION 5 - Scene Distribution Per Category Per Domain",
        "-" * 80,
        render_table(["domain", "category", *SCENE_BUCKETS], section5_rows),
    ])

    section6_rows = []
    total_quality = Counter()
    for category in VEHICLE_CATEGORIES:
        counters = quality_counts[category]
        total_quality.update(counters)
        section6_rows.append([
            category,
            counters["manualShape_true"],
            counters["manualShape_false"],
            counters["manualAttributes_true"],
            counters["manualAttributes_false"],
        ])
    section6_rows.append([
        "ALL",
        total_quality["manualShape_true"],
        total_quality["manualShape_false"],
        total_quality["manualAttributes_true"],
        total_quality["manualAttributes_false"],
    ])
    report_lines.extend([
        "",
        "SECTION 6 - Annotation Quality Flags",
        "-" * 80,
        render_table(
            ["category", "manualShape_true", "manualShape_false", "manualAttributes_true", "manualAttributes_false"],
            section6_rows,
        ),
    ])

    section7_rows = []
    decision_rows = []
    for threshold in PROJECTION_THRESHOLDS:
        night_train_counts = {}
        day_train_counts = {}
        night_cap = None
        day_cap = None

        for domain in DOMAINS:
            train_counts = {}
            for category in VEHICLE_CATEGORIES:
                total = projection_pool[threshold][domain][category]
                train_count, val_count, test_count = split_projection(total)
                train_counts[category] = train_count
                section7_rows.append([
                    threshold,
                    domain,
                    category,
                    total,
                    train_count,
                    val_count,
                    test_count,
                ])

            balanced_cap = min(train_counts.values()) if train_counts else 0
            for row in section7_rows[-len(VEHICLE_CATEGORIES):]:
                row.append(balanced_cap)

            if domain == "night":
                night_train_counts = train_counts
                night_cap = balanced_cap
            else:
                day_train_counts = train_counts
                day_cap = balanced_cap

        decision_rows.append([
            threshold,
            night_train_counts.get("bus", 0),
            night_train_counts.get("car", 0),
            night_train_counts.get("truck", 0),
            night_cap or 0,
            day_train_counts.get("bus", 0),
            day_train_counts.get("car", 0),
            day_train_counts.get("truck", 0),
            day_cap or 0,
        ])

    report_lines.extend([
        "",
        "SECTION 7 - Viable Crop Projections At Different Thresholds",
        "-" * 80,
        "Projection basis: combined train+val annotations, then re-split to 70/15/15.",
        render_table(
            ["threshold", "domain", "category", "surviving_total", "projected_train", "projected_val", "projected_test", "balanced_cap"],
            section7_rows,
        ),
    ])

    section8_rows = []
    for domain in DOMAINS:
        counts = image_vehicle_counts[domain]
        if counts:
            section8_rows.append([
                domain,
                len(counts),
                mean(counts),
                median(counts),
                quantile(counts, 0.25),
                quantile(counts, 0.75),
                max(counts),
            ])
        else:
            section8_rows.append([domain, 0, 0, 0, 0, 0, 0])
    report_lines.extend([
        "",
        "SECTION 8 - Per Image Vehicle Count Distribution",
        "-" * 80,
        render_table(["domain", "images", "mean", "median", "p25", "p75", "max"], section8_rows),
    ])

    report_lines.extend([
        "",
        "FINAL SUMMARY RECOMMENDATION TABLE",
        "-" * 80,
        render_table(
            ["threshold", "bus_night_train", "car_night_train", "truck_night_train", "night_cap", "bus_day_train", "car_day_train", "truck_day_train", "day_cap"],
            decision_rows,
        ),
    ])

    return "\n".join(report_lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze BDD100K vehicle annotations from train/val label JSON files.")
    parser.add_argument(
        "--train-json",
        type=Path,
        default=Path("bdd100k_labels_images_train.json"),
        help="Path to bdd100k_labels_images_train.json",
    )
    parser.add_argument(
        "--val-json",
        type=Path,
        default=Path("bdd100k_labels_images_val.json"),
        help="Path to bdd100k_labels_images_val.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_json.exists():
        raise FileNotFoundError(f"Train JSON not found: {args.train_json}")
    if not args.val_json.exists():
        raise FileNotFoundError(f"Val JSON not found: {args.val_json}")

    print(build_report(args.train_json, args.val_json))


if __name__ == "__main__":
    main()
