#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
duplicate_grouper.py
pHash 기반 유사/중복 이미지 그룹화 도구 (YOLO 라벨 지원)

사용법:
  python duplicate_grouper.py -i ./dataset -o ./result
  python duplicate_grouper.py -i ./dataset -o ./result --label-mode unlabeled
  python duplicate_grouper.py -i ./dataset -o ./result --crop --names classes.txt --retry
"""

import os
import sys
import argparse
import shutil
import time
import logging
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from collections import defaultdict

import random
from PIL import Image
import imagehash
from tqdm import tqdm

# ──────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
HASH_SIZE = 8

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# 디렉토리 자동 감지
# ──────────────────────────────────────────────
def detect_dataset_structure(input_dir, labels_override=None):
    input_path = Path(input_dir)

    jpeg_dir = input_path / "JPEGImages"
    if jpeg_dir.exists() and jpeg_dir.is_dir():
        img_dir = jpeg_dir
        logger.info(f"이미지 디렉토리 자동 감지: {jpeg_dir}")
    else:
        img_dir = input_path
        logger.info(f"이미지 디렉토리: {input_path}")

    # labels_override가 있으면 지정 label_dir 반환 (기존 탐색 생략)
    label_dir = None
    if labels_override:
        lp = Path(labels_override)
        if lp.exists():
            label_dir = lp
            logger.info(f"라벨 디렉토리 (직접 지정): {label_dir}")
        else:
            logger.warning(f"지정된 라벨 디렉토리가 없습니다: {labels_override}")

    return img_dir, label_dir


def generate_crops(labeled_files, padding, min_crop_size):
    crops_by_class = defaultdict(list)
    stem_to_path = {}  # source_id -> original image path
    errors = 0

    for img_idx, (img_path, label_path) in enumerate(
        tqdm(
            labeled_files, desc="객체 크롭 생성", unit="img", ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
    ):
        objects = parse_yolo_label(label_path) if label_path else []
        if not objects:
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            errors += 1
            continue

        source_id = f"img{img_idx:08d}"
        stem_to_path[source_id] = img_path
        for obj_idx, (cls_id, cx, cy, bw, bh) in enumerate(objects):
            cropped = crop_object(img, cx, cy, bw, bh, padding, min_crop_size)
            if cropped is not None:
                crops_by_class[cls_id].append((cropped, f"{source_id}_obj{obj_idx}"))
        img.close()

    if errors:
        logger.warning(f"이미지 읽기 실패: {errors}개")
    return crops_by_class, stem_to_path


def save_crop_representative_list(groups, retry_groups, final_unique, hash_dict,
                                   stem_to_path, output_dir, list_name):
    """
    크롭 대표 unique 의 stem 기반 원본경로로 변환해서 저장.
    변환 실패는 경고만 출력하고 건너뜀.
    """
    def stem_key_to_path(key):
        # key: "img_stem_objN" 에서 img_stem 추출
        parts = key.rsplit('_obj', 1)
        if len(parts) == 2:
            return stem_to_path.get(parts[0])
        return None

    all_keys = collect_representative_files(groups, retry_groups, final_unique, hash_dict)
    all_keys = sorted(set(all_keys))

    paths = []
    missing = 0
    for key in all_keys:
        p = stem_key_to_path(key)
        if p:
            paths.append(p)
        else:
            missing += 1

    if missing:
        logger.warning(f"  경로 변환 실패 {missing}개 (stem 매핑 없음)")

    paths = sorted(set(paths))
    list_path = Path(output_dir) / list_name

    n_1st   = len(groups)
    n_retry = len(retry_groups)
    n_uniq  = len(final_unique)

    with open(list_path, 'w', encoding='utf-8') as f:
        f.write(f"# 크롭 대표 이미지 원본 경로 목록\n")
        f.write(f"# 총 {len(paths)}개 "
                f"(1차 대표 {n_1st}개 + retry 대표 {n_retry}개 + 고유 {n_uniq}개)\n")
        f.write(f"# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for p in paths:
            f.write(f"{p}\n")

    logger.info(f"크롭 목록 저장 완료: {list_path} ({len(paths):,}개)")
    return paths


def match_labels(image_files, label_dir=None, images_dir_name="JPEGImages"):
    labeled = []
    unlabeled = []

    for img_path in image_files:
        img_p = Path(img_path)
        img_stem = img_p.stem

        if label_dir:
            label_path = Path(label_dir) / f"{img_stem}.txt"
        else:
            label_path = None

            # images_dir_name 디렉토리 기준으로 상위의 labels/ 탐색
            for parent in img_p.parents:
                if parent.name == images_dir_name:
                    candidate = parent.parent / "labels" / f"{img_stem}.txt"
                    if candidate.exists():
                        label_path = candidate
                    break  # images_dir_name 찾으면 더 이상 탐색하지 않음

            # images_dir_name 없으면 기존 fallback 탐색
            if label_path is None:
                candidate = img_p.parent / "labels" / f"{img_stem}.txt"
                if candidate.exists():
                    label_path = candidate
                else:
                    candidate = img_p.parent.parent / "labels" / f"{img_stem}.txt"
                    if candidate.exists():
                        label_path = candidate

        if label_path and label_path.exists():
            labeled.append((img_path, str(label_path)))
        else:
            unlabeled.append(img_path)

    return labeled, unlabeled


# ──────────────────────────────────────────────
# 이미지/라벨 스캔 및 로딩
# ──────────────────────────────────────────────
def scan_images(img_dir):
    files = []
    for f in Path(img_dir).rglob('*'):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(str(f.resolve()))
    files.sort()
    return files


# ──────────────────────────────────────────────
# YOLO 라벨 파싱 + 크롭
# ──────────────────────────────────────────────
def parse_yolo_label(label_path):
    objects = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                    objects.append((cls_id, cx, cy, w, h))
    except Exception as e:
        logger.warning(f"라벨 파싱 실패 {label_path}: {e}")
    return objects


def crop_object(img, cx, cy, bw, bh, padding, min_crop_size):
    img_w, img_h = img.size
    px_cx, px_cy = cx * img_w, cy * img_h
    px_bw, px_bh = bw * img_w, bh * img_h
    pad_w, pad_h = px_bw * padding, px_bh * padding

    x1 = int(max(0, px_cx - px_bw / 2 - pad_w))
    y1 = int(max(0, px_cy - px_bh / 2 - pad_h))
    x2 = int(min(img_w, px_cx + px_bw / 2 + pad_w))
    y2 = int(min(img_h, px_cy + px_bh / 2 + pad_h))

    if (x2 - x1) < min_crop_size or (y2 - y1) < min_crop_size:
        return None
    return img.crop((x1, y1, x2, y2))


def compute_hash_single(filepath):
    try:
        with Image.open(filepath) as img:
            h = imagehash.phash(img.convert("RGB"), hash_size=HASH_SIZE)
            return (filepath, h)
    except Exception as e:
        return (filepath, None, str(e))


def compute_hashes_parallel(file_list, num_workers):
    results = {}
    errors = []
    with Pool(processes=num_workers) as pool:
        for result in tqdm(
            pool.imap_unordered(compute_hash_single, file_list, chunksize=64),
            total=len(file_list), desc="해시 계산", unit="img", ncols=100,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        ):
            if len(result) == 3:
                errors.append((result[0], result[2]))
            else:
                fp, h = result
                if h is not None:
                    results[fp] = h
    if errors:
        logger.warning(f"해시 계산 실패: {len(errors)}개")
        for fp, err in errors[:3]:
            logger.warning(f"  - {Path(fp).name}: {err}")
        if len(errors) > 3:
            logger.warning(f"  ... 외 {len(errors) - 3}개")
    return results


def compute_hashes_from_pil(pil_items):
    results = {}
    for pil_img, source_info in tqdm(
        pil_items, desc="크롭 해시 계산", unit="crop", ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    ):
        try:
            h = imagehash.phash(pil_img, hash_size=HASH_SIZE)
            results[source_info] = h
        except Exception:
            pass
    return results


# ──────────────────────────────────────────────
# Union-Find
# ──────────────────────────────────────────────
class UnionFind:
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1

    def groups(self):
        grp = defaultdict(list)
        for e in self.parent:
            grp[self.find(e)].append(e)
        return dict(grp)


# ──────────────────────────────────────────────
# 대표 이미지 선정
# ──────────────────────────────────────────────
def select_representative(group, hash_dict):
    if len(group) <= 1:
        return group[0], []
    hashes = {k: hash_dict[k] for k in group}
    min_avg = float('inf')
    rep = group[0]
    for c in group:
        avg = sum(hashes[c] - hashes[o] for o in group if o != c) / (len(group) - 1)
        if avg < min_avg:
            min_avg = avg
            rep = c
    return rep, [k for k in group if k != rep]


# ──────────────────────────────────────────────
# 그룹화 알고리즘
# ──────────────────────────────────────────────
def group_chain(hash_dict, threshold, desc_prefix=""):
    keys = list(hash_dict.keys())
    hashes = [hash_dict[k] for k in keys]
    n = len(keys)
    uf = UnionFind(range(n))
    total_pairs = n * (n - 1) // 2

    pbar = tqdm(
        total=total_pairs, desc=f"그룹화 chain {desc_prefix}", unit="pair", ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    batch = 0
    for i in range(n):
        for j in range(i + 1, n):
            if hashes[i] - hashes[j] <= threshold:
                uf.union(i, j)
            batch += 1
            if batch >= 10000:
                pbar.update(batch)
                batch = 0
    if batch > 0:
        pbar.update(batch)
    pbar.close()

    raw = uf.groups()
    groups, unique = [], []
    for members in raw.values():
        paths = [keys[m] for m in members]
        (groups if len(paths) >= 2 else unique).append(paths if len(paths) >= 2 else paths[0])
    return groups, unique


def group_representative_mode(hash_dict, threshold, desc_prefix=""):
    keys = list(hash_dict.keys())
    groups_list = []

    for key in tqdm(
        keys, desc=f"그룹화 rep {desc_prefix}", unit="img", ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    ):
        h = hash_dict[key]
        matched = False
        for rep_hash, members in groups_list:
            if h - rep_hash <= threshold:
                members.append(key)
                matched = True
                break
        if not matched:
            groups_list.append((h, [key]))

    groups, unique = [], []
    for _, members in groups_list:
        (groups if len(members) >= 2 else unique).append(members if len(members) >= 2 else members[0])
    return groups, unique


def do_grouping(hash_dict, threshold, mode, desc_prefix=""):
    if mode == 'chain':
        return group_chain(hash_dict, threshold, desc_prefix)
    else:
        return group_representative_mode(hash_dict, threshold, desc_prefix)


# ──────────────────────────────────────────────
# 파일 복사
# ──────────────────────────────────────────────
def safe_copy(src, dst_dir, filename):
    dst = dst_dir / filename
    if dst.exists():
        stem = Path(filename).stem
        suffix = Path(filename).suffix
        counter = 1
        while dst.exists():
            dst = dst_dir / f"{stem}_dup{counter}{suffix}"
            counter += 1
    shutil.copy2(src, dst)
    return dst


def copy_groups_with_rep(groups, unique, base_output_dir, hash_dict, pbar, group_prefix="group"):
    """
    그룹별로 지정 출력 디렉토리 구조로 복사.
    group_prefix로 1차(group) / 2차(retry_group) 구분.
    반환: 대표 이미지 경로 목록
    """
    base = Path(base_output_dir)
    base.mkdir(parents=True, exist_ok=True)
    rep_list = []

    for idx, group in enumerate(groups, 1):
        group_dir = base / f"{group_prefix}_{idx:04d}"
        group_dir.mkdir(parents=True, exist_ok=True)

        rep_key, dup_keys = select_representative(group, hash_dict)
        rep_list.append(rep_key)

        rep_ext = Path(rep_key).suffix
        shutil.copy2(rep_key, group_dir / f"representative{rep_ext}")
        pbar.update(1)

        if dup_keys:
            dup_dir = group_dir / "duplicates"
            dup_dir.mkdir(parents=True, exist_ok=True)
            for fp in dup_keys:
                safe_copy(fp, dup_dir, Path(fp).name)
                pbar.update(1)

    if unique:
        unique_dir = base / "unique"
        unique_dir.mkdir(parents=True, exist_ok=True)
        for fp in unique:
            safe_copy(fp, unique_dir, Path(fp).name)
            pbar.update(1)

    return rep_list


def copy_crop_groups_with_rep(groups, unique, crop_pil_map, hash_dict,
                               base_output_dir, pbar, group_prefix="group"):
    base = Path(base_output_dir)
    base.mkdir(parents=True, exist_ok=True)

    for idx, group in enumerate(groups, 1):
        group_dir = base / f"{group_prefix}_{idx:04d}"
        group_dir.mkdir(parents=True, exist_ok=True)

        rep_key, dup_keys = select_representative(group, hash_dict)

        pil_img = crop_pil_map.get(rep_key)
        if pil_img:
            pil_img.save(group_dir / "representative.jpg", quality=95)
        pbar.update(1)

        if dup_keys:
            dup_dir = group_dir / "duplicates"
            dup_dir.mkdir(parents=True, exist_ok=True)
            for si in dup_keys:
                pil_img = crop_pil_map.get(si)
                if pil_img:
                    pil_img.save(dup_dir / f"{si}.jpg", quality=95)
                pbar.update(1)

    if unique:
        unique_dir = base / "unique"
        unique_dir.mkdir(parents=True, exist_ok=True)
        for si in unique:
            pil_img = crop_pil_map.get(si)
            if pil_img:
                pil_img.save(unique_dir / f"{si}.jpg", quality=95)
            pbar.update(1)


# ──────────────────────────────────────────────
# retry 루프
# ──────────────────────────────────────────────
def retry_grouping_files(unique_keys, hash_dict, retry_threshold, mode, desc_prefix=""):
    """
    unique 이미지를 retry_threshold로 재그룹화.
    반환: (retry_groups, final_unique)
    """
    if len(unique_keys) < 2:
        return [], unique_keys

    sub_hash = {k: hash_dict[k] for k in unique_keys if k in hash_dict}
    if len(sub_hash) < 2:
        return [], unique_keys

    logger.info(f"  retry: {len(sub_hash):,}개 unique를 임계값 {retry_threshold}로 재그룹화")
    retry_groups, final_unique = do_grouping(sub_hash, retry_threshold, mode, f"retry {desc_prefix}")
    retry_groups.sort(key=len, reverse=True)

    logger.info(f"  retry 결과: {len(retry_groups):,}개 그룹, {len(final_unique):,}개 최종 고유")
    return retry_groups, final_unique


def retry_grouping_crops(unique_keys, hash_dict, crop_pil_map, retry_threshold, mode, desc_prefix=""):
    """크롭 unique 재그룹화."""
    if len(unique_keys) < 2:
        return [], unique_keys

    sub_hash = {k: hash_dict[k] for k in unique_keys if k in hash_dict}
    if len(sub_hash) < 2:
        return [], unique_keys

    logger.info(f"  retry: {len(sub_hash):,}개 unique 크롭을 임계값 {retry_threshold}로 재그룹화")
    retry_groups, final_unique = do_grouping(sub_hash, retry_threshold, mode, f"retry {desc_prefix}")
    retry_groups.sort(key=len, reverse=True)

    logger.info(f"  retry 결과: {len(retry_groups):,}개 그룹, {len(final_unique):,}개 최종 고유")
    return retry_groups, final_unique


# ──────────────────────────────────────────────
# 썸네일 프리뷰 이미지 생성
# ──────────────────────────────────────────────
def generate_grid_preview_from_files(file_list, output_path, title="preview",
                                      grid_cols=10, grid_rows=10, cell_size=128):
    """
    파일 경로 목록에서 최대 grid_cols*grid_rows 개를 샘플링해
    프리뷰(grid) 형태로 저장하는 썸네일 이미지 생성.
    """
    max_count = grid_cols * grid_rows
    if len(file_list) == 0:
        return None

    samples = random.sample(file_list, min(len(file_list), max_count))

    images = []
    for fp in samples:
        try:
            img = Image.open(fp).convert("RGB")
            img = img.resize((cell_size, cell_size), Image.LANCZOS)
            images.append(img)
        except Exception:
            pass

    if not images:
        return None

    actual_count = len(images)
    cols = min(grid_cols, actual_count)
    rows = (actual_count + cols - 1) // cols

    grid_img = Image.new('RGB', (cols * cell_size, rows * cell_size), (40, 40, 40))

    for idx, img in enumerate(images):
        x = (idx % cols) * cell_size
        y = (idx // cols) * cell_size
        grid_img.paste(img, (x, y))

    save_path = Path(output_path)
    grid_img.save(save_path, quality=90)
    logger.info(f"썸네일 저장: {save_path} ({actual_count}개, {cols}x{rows})")
    return save_path


def generate_grid_preview_from_pil(pil_map, key_list, output_path, title="preview",
                                    grid_cols=10, grid_rows=10, cell_size=128):
    """
    PIL 이미지 맵에서 최대 grid_cols*grid_rows 개를 샘플링해 프리뷰 생성.
    크롭 이미지용.
    """
    max_count = grid_cols * grid_rows
    if not key_list:
        return None

    samples = random.sample(key_list, min(len(key_list), max_count))

    images = []
    for key in samples:
        pil_img = pil_map.get(key)
        if pil_img:
            try:
                img = pil_img.copy().resize((cell_size, cell_size), Image.LANCZOS)
                images.append(img)
            except Exception:
                pass

    if not images:
        return None

    actual_count = len(images)
    cols = min(grid_cols, actual_count)
    rows = (actual_count + cols - 1) // cols

    grid_img = Image.new('RGB', (cols * cell_size, rows * cell_size), (40, 40, 40))

    for idx, img in enumerate(images):
        x = (idx % cols) * cell_size
        y = (idx // cols) * cell_size
        grid_img.paste(img, (x, y))

    save_path = Path(output_path)
    grid_img.save(save_path, quality=90)
    logger.info(f"썸네일 저장: {save_path} ({actual_count}개, {cols}x{rows})")
    return save_path


def collect_representative_files(groups, retry_groups, final_unique, hash_dict):
    """
    전체 대표 이미지 + 최종 unique 수집.
    1차 그룹 대표 + retry 그룹 대표 + 최종 고유 = 전체 제출 목록.
    """
    rep_files = []

    for group in groups:
        rep_key, _ = select_representative(group, hash_dict)
        rep_files.append(rep_key)

    for group in retry_groups:
        rep_key, _ = select_representative(group, hash_dict)
        rep_files.append(rep_key)

    rep_files.extend(final_unique)
    return rep_files


# ──────────────────────────────────────────────
# 대표 이미지 목록 (원본 경로 기준)
# ──────────────────────────────────────────────
def save_representative_list(groups, retry_groups, final_unique, hash_dict,
                              output_dir, list_name):
    """
    1차 대표 + retry 대표 + 최종 unique 의 중복 없는 최종 목록.
    원본 경로 기준.
    """
    all_paths = collect_representative_files(groups, retry_groups, final_unique, hash_dict)
    all_paths = sorted(set(all_paths))

    list_path = Path(output_dir) / list_name

    n_1st_rep   = len(groups)
    n_retry_rep = len(retry_groups)
    n_unique    = len(final_unique)

    with open(list_path, 'w', encoding='utf-8') as f:
        f.write(f"# 대표 이미지 + 고유 이미지 (중복 제거된 최종 목록)\n")
        f.write(f"# 총 {len(all_paths)}개 "
                f"(1차 대표 {n_1st_rep}개 + "
                f"retry 대표 {n_retry_rep}개 + "
                f"고유 {n_unique}개)\n")
        f.write(f"# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for p in all_paths:
            f.write(f"{p}\n")

    logger.info(f"목록 저장 완료: {list_path} ({len(all_paths):,}개)")
    return all_paths


# ──────────────────────────────────────────────
# 리포트
# ──────────────────────────────────────────────
def gen_report(rd, out_dir, args, elapsed):
    rp = Path(out_dir) / "report.txt"
    with open(rp, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  pHash 기반 유사 이미지 그룹화 리포트\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"실행 시각       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"입력 디렉토리   : {args.input}\n")
        f.write(f"출력 디렉토리   : {args.output}\n")
        f.write(f"그룹화 방식     : {args.mode}\n")
        f.write(f"유사도 임계값   : {args.threshold}\n")
        if getattr(args, 'retry', False):
            f.write(f"retry 임계값    : {args.retry_threshold}\n")
        if getattr(args, 'crop', False):
            f.write(f"크롭 임계값     : {args.crop_threshold}\n")
            f.write(f"크롭 패딩       : {args.padding}\n")
            f.write(f"최소 크롭 크기  : {args.min_crop_size}px\n")
        f.write(f"소요 시간       : {elapsed:.1f}초\n\n")

        for sec, d in rd.items():
            gs  = d.get('groups', [])
            uq  = d.get('unique', [])
            rgs = d.get('retry_groups', [])
            fuq = d.get('final_unique', uq)
            hd  = d.get('hash_dict', {})

            total_in       = sum(len(g) for g in gs) + len(uq)
            grouped_imgs   = sum(len(g) for g in gs)
            n_rep_1st      = len(gs)
            n_rep_retry    = len(rgs)
            retry_imgs     = sum(len(g) for g in rgs)
            n_final_unique = len(fuq)
            total_out      = n_rep_1st + n_rep_retry + n_final_unique

            f.write("-" * 70 + "\n")
            f.write(f"  [{sec}]\n")
            f.write("-" * 70 + "\n")
            f.write(f"  입력 이미지       : {total_in:,}개\n")
            f.write(f"  → 1차 그룹        : {len(gs):,}개 그룹 / {grouped_imgs:,}개 → 대표 {n_rep_1st:,}개\n")
            f.write(f"  → 1차 unique      : {len(uq):,}개\n")
            if rgs:
                f.write(f"  → retry 그룹      : {len(rgs):,}개 그룹 / {retry_imgs:,}개 → 대표 {n_rep_retry:,}개\n")
                f.write(f"  → 최종 고유       : {n_final_unique:,}개\n")
            else:
                f.write(f"  → 최종 고유       : {n_final_unique:,}개\n")
            f.write(f"\n")

            if total_in > 0:
                f.write(f"  ※ 최종 결과       : {total_in:,}개 → {total_out:,}개 "
                        f"({100.0 * total_out / total_in:.1f}% 유지 / "
                        f"{100.0 * (1 - total_out / total_in):.1f}% 제거)\n\n")

            if all_sizes := [len(g) for g in gs + rgs]:
                f.write(f"  최대 그룹 크기    : {max(all_sizes):,}\n")
                f.write(f"  평균 그룹 크기    : {sum(all_sizes)/len(all_sizes):.1f}\n\n")

            for idx, g in enumerate(gs[:20], 1):
                rep, dups = select_representative(g, hd) if hd else (g[0], g[1:])
                bh = hd.get(rep)
                nm = Path(rep).name if os.sep in str(rep) or '/' in str(rep) else rep
                f.write(f"  [group_{idx:04d}] ({len(g)}개 → 유지 1개, 제거 {len(g)-1}개)\n")
                f.write(f"    ★ {nm}\n")
                for it in dups[:10]:
                    ih = hd.get(it)
                    dist = (bh - ih) if bh and ih else "?"
                    nm2 = Path(it).name if os.sep in str(it) or '/' in str(it) else it
                    f.write(f"      {nm2:<52s} 거리:{dist}\n")
                if len(dups) > 10:
                    f.write(f"      ... 외 {len(dups)-10}개\n")
                f.write("\n")
            if len(gs) > 20:
                f.write(f"  ... 외 {len(gs)-20}개 그룹 생략\n\n")

            if rgs:
                f.write(f"  --- retry 그룹 ---\n\n")
                for idx, g in enumerate(rgs[:10], 1):
                    rep, dups = select_representative(g, hd) if hd else (g[0], g[1:])
                    bh = hd.get(rep)
                    nm = Path(rep).name if os.sep in str(rep) or '/' in str(rep) else rep
                    f.write(f"  [retry_{idx:04d}] ({len(g)}개 → 유지 1개, 제거 {len(g)-1}개)\n")
                    f.write(f"    ★ {nm}\n")
                    for it in dups[:10]:
                        ih = hd.get(it)
                        dist = (bh - ih) if bh and ih else "?"
                        nm2 = Path(it).name if os.sep in str(it) or '/' in str(it) else it
                        f.write(f"      {nm2:<52s} 거리:{dist}\n")
                    if len(dups) > 10:
                        f.write(f"      ... 외 {len(dups)-10}개\n")
                    f.write("\n")
                if len(rgs) > 10:
                    f.write(f"  ... 외 {len(rgs)-10}개 retry 그룹 생략\n\n")

    logger.info(f"리포트 저장 완료: {rp}")


# ──────────────────────────────────────────────
# 클래스명
# ──────────────────────────────────────────────
def load_class_names(names_path):
    if not names_path:
        return {}
    try:
        names = {}
        with open(names_path, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                name = line.strip()
                if name:
                    names[idx] = name
        logger.info(f"클래스명 로드: {len(names)}개")
        return names
    except Exception as e:
        logger.warning(f"클래스명 파일 로드 실패: {e}")
        return {}


def get_class_name(cls_id, class_names):
    return class_names.get(cls_id, f"class_{cls_id}")


# ──────────────────────────────────────────────
# 단계별 리포트 함수
# ──────────────────────────────────────────────
def _write_report_header(f, title, args, elapsed=None):
    f.write("=" * 70 + "\n")
    f.write(f"  {title}\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"  실행 시각     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"  입력 디렉토리 : {args.input}\n")
    f.write(f"  출력 디렉토리 : {args.output}\n")
    f.write(f"  그룹화 방식   : {args.mode}\n")
    if elapsed is not None:
        f.write(f"  소요 시간     : {elapsed:.1f}초\n")
    f.write("\n")


def gen_report_background(groups, retry_groups, final_unique, hash_dict,
                           out_dir, args, elapsed=None):
    """배경 그룹화 단계 리포트 → report_background.txt"""
    rp = Path(out_dir) / "report_background.txt"
    total_in = sum(len(g) for g in groups + retry_groups) + len(final_unique)
    n_groups  = len(groups)
    n_retry   = len(retry_groups)
    n_unique  = len(final_unique)
    grouped_imgs  = sum(len(g) for g in groups)
    retry_imgs    = sum(len(g) for g in retry_groups)
    total_out     = n_groups + n_retry + n_unique

    with open(rp, 'w', encoding='utf-8') as f:
        _write_report_header(f, "배경(Background) 그룹화 리포트", args, elapsed)
        f.write(f"  유사도 임계값 : {args.threshold}\n")
        if getattr(args, 'retry', False):
            f.write(f"  retry 임계값  : {args.retry_threshold}\n")
        f.write("\n")
        f.write("-" * 70 + "\n")
        f.write(f"  입력 이미지       : {total_in:,}개\n\n")
        f.write(f"  1차 그룹화\n")
        f.write(f"    그룹 수         : {n_groups:,}개\n")
        f.write(f"    그룹 내 이미지  : {grouped_imgs:,}개 → 대표 {n_groups:,}개 (그룹당 1장)\n")
        f.write(f"    1차 unique      : {total_in - grouped_imgs:,}개\n")
        if retry_groups:
            f.write(f"\n  retry 그룹화\n")
            f.write(f"    retry 그룹 수   : {n_retry:,}개\n")
            f.write(f"    retry 내 이미지 : {retry_imgs:,}개 → 대표 {n_retry:,}개\n")
            f.write(f"    최종 unique     : {n_unique:,}개\n")
        else:
            f.write(f"    최종 unique     : {n_unique:,}개\n")
        f.write("\n")
        f.write("-" * 70 + "\n")
        f.write(f"  ※ 출력 (background_list.txt)\n")
        f.write(f"    그룹 대표       : {n_groups + n_retry:,}개  ← background_list.txt 에 포함\n")
        f.write(f"    unique 이미지   : {n_unique:,}개  ← label_list.txt 에 포함\n")
        f.write(f"    합계 출력       : {total_out:,}개\n")
        if total_in > 0:
            f.write(f"    제거율          : {100.0 * (1 - total_out / total_in):.1f}%  "
                    f"({total_in:,} → {total_out:,})\n")
        f.write("\n")

        if all_sizes := [len(g) for g in groups + retry_groups]:
            f.write(f"  최대 그룹 크기  : {max(all_sizes):,}\n")
            f.write(f"  평균 그룹 크기  : {sum(all_sizes)/len(all_sizes):.1f}\n\n")

        f.write("  상위 20개 그룹 미리보기\n")
        f.write("-" * 70 + "\n")
        all_grps = groups[:20] + retry_groups[:max(0, 20 - len(groups))]
        for idx, g in enumerate(all_grps, 1):
            rep, dups = select_representative(g, hash_dict) if hash_dict else (g[0], g[1:])
            nm = Path(rep).name
            f.write(f"  [{idx:>3}] 대표: {nm:<52s} ({len(g)}장)\n")
            for dup in dups[:5]:
                f.write(f"        dup : {Path(dup).name}\n")
            if len(dups) > 5:
                f.write(f"        ... 외 {len(dups)-5}개\n")

    logger.info(f"배경 리포트 저장: {rp}")


def gen_report_label(class_data, bg_unique_count, out_dir, args, elapsed=None):
    """
    크롭(라벨) 그룹화 단계 리포트 → report_label.txt
    class_data: {cls_name: {'groups', 'retry_groups', 'final_unique', 'hash_dict', 'total_crops'}}
    bg_unique_count: int  (배경 unique 이미지 수)
    """
    rp = Path(out_dir) / "report_label.txt"

    total_crops_all  = sum(d['total_crops'] for d in class_data.values())
    total_groups_all = sum(len(d['groups']) + len(d['retry_groups']) for d in class_data.values())
    total_unique_all = sum(len(d['final_unique']) for d in class_data.values())

    # 크롭 그룹 대표 원본 이미지 수 (label_list.txt 기여분)
    label_from_groups = sum(len(d['groups']) + len(d['retry_groups']) for d in class_data.values())
    label_from_unique = sum(len(d['final_unique']) for d in class_data.values())

    with open(rp, 'w', encoding='utf-8') as f:
        _write_report_header(f, "크롭(Label) 그룹화 리포트", args, elapsed)
        f.write(f"  크롭 임계값   : {getattr(args, 'crop_threshold', args.threshold)}\n")
        if getattr(args, 'retry', False):
            f.write(f"  retry 임계값  : {args.retry_threshold}\n")
        f.write(f"  클래스 수     : {len(class_data)}개\n\n")

        f.write("=" * 70 + "\n")
        f.write("  클래스별 결과\n")
        f.write("=" * 70 + "\n")

        for cls_name, d in class_data.items():
            g1    = d['groups']
            rg    = d['retry_groups']
            fuq   = d['final_unique']
            total = d['total_crops']
            g1_imgs  = sum(len(g) for g in g1)
            rg_imgs  = sum(len(g) for g in rg)
            total_out = len(g1) + len(rg) + len(fuq)

            f.write(f"\n  [{cls_name}]\n")
            f.write(f"    입력 크롭       : {total:,}개\n")
            f.write(f"    1차 그룹        : {len(g1):,}개 그룹 / {g1_imgs:,}개 → 대표 {len(g1):,}개\n")
            f.write(f"    1차 unique      : {total - g1_imgs:,}개\n")
            if rg:
                f.write(f"    retry 그룹      : {len(rg):,}개 그룹 / {rg_imgs:,}개 → 대표 {len(rg):,}개\n")
                f.write(f"    최종 unique     : {len(fuq):,}개\n")
            else:
                f.write(f"    최종 unique     : {len(fuq):,}개\n")
            if total > 0:
                f.write(f"    크롭 제거율     : {100.0 * (1 - total_out / total):.1f}%  "
                        f"({total:,} → {total_out:,})\n")

        f.write("\n")
        f.write("=" * 70 + "\n")
        f.write("  전체 요약\n")
        f.write("=" * 70 + "\n")
        f.write(f"  전체 크롭 입력  : {total_crops_all:,}개\n")
        f.write(f"  전체 그룹 수    : {total_groups_all:,}개\n")
        f.write(f"  전체 unique 크롭: {total_unique_all:,}개\n\n")
        f.write(f"  ※ label_list.txt 구성\n")
        f.write(f"    크롭 그룹 대표  : {label_from_groups:,}개 원본 이미지\n")
        f.write(f"    크롭 unique     : {label_from_unique:,}개 원본 이미지\n")
        f.write(f"    배경 unique     : {bg_unique_count:,}개 원본 이미지\n")
        f.write(f"    (중복 제거 후 label_list.txt 에 저장됨)\n")

    logger.info(f"라벨 리포트 저장: {rp}")


def gen_report_representative(cross_groups, cross_retry_groups, cross_unique,
                               input_count, out_dir, args, elapsed=None):
    """
    대표 간 재그룹화 단계 리포트 → report_representative.txt
    input_count: 재그룹화 전 입력 대표 크롭 수
    """
    rp = Path(out_dir) / "report_representative.txt"
    n_groups = len(cross_groups)
    n_retry  = len(cross_retry_groups)
    n_unique = len(cross_unique)
    grouped_imgs = sum(len(g) for g in cross_groups)
    retry_imgs   = sum(len(g) for g in cross_retry_groups)
    total_out    = n_groups + n_retry + n_unique

    with open(rp, 'w', encoding='utf-8') as f:
        _write_report_header(f, "대표(Representative) 간 재그룹화 리포트", args, elapsed)
        f.write(f"  대표 임계값   : {getattr(args, 'rep_threshold', args.threshold)}\n\n")
        f.write("-" * 70 + "\n")
        f.write(f"  입력 대표 크롭    : {input_count:,}개  (label 단계 각 클래스 그룹 대표)\n\n")
        f.write(f"  1차 재그룹화\n")
        f.write(f"    그룹 수         : {n_groups:,}개\n")
        f.write(f"    그룹 내 대표 수 : {grouped_imgs:,}개 → 최종 대표 {n_groups:,}개\n")
        f.write(f"    1차 unique      : {input_count - grouped_imgs:,}개\n")
        if cross_retry_groups:
            f.write(f"\n  retry 재그룹화\n")
            f.write(f"    retry 그룹 수   : {n_retry:,}개\n")
            f.write(f"    retry 내 대표   : {retry_imgs:,}개 → 최종 대표 {n_retry:,}개\n")
            f.write(f"    최종 unique     : {n_unique:,}개\n")
        else:
            f.write(f"    최종 unique     : {n_unique:,}개\n")
        f.write("\n")
        f.write("-" * 70 + "\n")
        f.write(f"  ※ 출력 (representative_list.txt)\n")
        f.write(f"    재그룹 대표     : {n_groups + n_retry:,}개 원본 이미지\n")
        f.write(f"    unique          : {n_unique:,}개 원본 이미지\n")
        f.write(f"    합계 출력       : {total_out:,}개\n")
        if input_count > 0:
            f.write(f"    추가 제거율     : {100.0 * (1 - total_out / input_count):.1f}%  "
                    f"({input_count:,} → {total_out:,})\n")

    logger.info(f"대표 리포트 저장: {rp}")


def save_path_list(paths, out_dir, list_name, comment=""):
    """단순 경로 목록 파일 저장."""
    paths = sorted(set(paths))
    list_path = Path(out_dir) / list_name
    with open(list_path, 'w', encoding='utf-8') as f:
        if comment:
            f.write(f"# {comment}\n")
        f.write(f"# 총 {len(paths):,}개\n")
        f.write(f"# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        for p in paths:
            f.write(f"{p}\n")
    logger.info(f"목록 저장: {list_path} ({len(paths):,}개)")
    return paths


# ──────────────────────────────────────────────
# 파이프라인 선택 인터랙티브
# ──────────────────────────────────────────────
PIPELINE_CHOICES = {
    '1': 'bg_only',
    '2': 'bg_label',
    '3': 'bg_label_rep',
}

def prompt_pipeline(args_pipeline):
    """
    파이프라인 단계 선택 (CLI 인수 없을 경우 대화식 프롬프트).
    bg_only        : 배경 그룹화만
    bg_label       : 배경 + 크롭(라벨) 그룹화
    bg_label_rep   : 배경 + 크롭 + 대표 간 재그룹화
    """
    valid = set(PIPELINE_CHOICES.values())

    if args_pipeline and args_pipeline in valid:
        print(f"  파이프라인   : {args_pipeline} (지정됨)")
        return args_pipeline

    print()
    print("  ─── 파이프라인 선택 ─────────────────────────────")
    print("  1) bg_only      : 배경 그룹화만")
    print("  2) bg_label     : 배경 + 크롭(라벨) 그룹화")
    print("  3) bg_label_rep : 배경 + 크롭 + 대표 간 재그룹화  [권장]")
    print("  ──────────────────────────────────────────────────")

    try:
        user_input = input("  선택 (1/2/3, Enter=3): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        user_input = ""

    pipeline = PIPELINE_CHOICES.get(user_input, 'bg_label_rep')
    print(f"  파이프라인   : {pipeline}")
    return pipeline


# ──────────────────────────────────────────────
# 워커 수 확인 인터랙티브
# ──────────────────────────────────────────────
def prompt_workers(args_workers):
    total_cores = cpu_count()
    default_workers = max(1, total_cores // 2)

    print(f"  CPU 코어     : {total_cores}개 사용 가능")

    if args_workers is not None:
        workers = max(1, min(args_workers, total_cores))
        print(f"  워커 수      : {workers}개 (지정됨)")
        return workers

    print(f"  기본 워커    : {default_workers}개 (CPU절반)")
    print()

    try:
        user_input = input(f"  워커 수 변경 (1~{total_cores}, Enter=기본값 {default_workers}): ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        user_input = ""

    if user_input == "":
        workers = default_workers
    else:
        try:
            workers = int(user_input)
            workers = max(1, min(workers, total_cores))
        except ValueError:
            logger.warning(f"잘못된 입력 '{user_input}', 기본값 {default_workers} 사용")
            workers = default_workers

    print(f"  최종 워커 수  : {workers}개")
    return workers


# ──────────────────────────────────────────────
# GUI에서 직접 호출 가능한 실행 함수
# ──────────────────────────────────────────────
def run_grouper(args, log_fn=None):
    """
    GUI 또는 코드에서 직접 호출 가능한 메인 실행 함수.
    args: argparse.Namespace 또는 유사 속성을 가진 객체
    log_fn: 로그 출력 함수 (없으면 print 사용)
    """
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    # 기본값 처리
    if getattr(args, 'crop_threshold', None) is None:
        args.crop_threshold = args.threshold
    if getattr(args, 'retry_threshold', None) is None:
        args.retry_threshold = args.threshold * 2
    if getattr(args, 'rep_threshold', None) is None:
        args.rep_threshold = getattr(args, 'crop_threshold', args.threshold)

    # 파이프라인 선택 (crop 미사용 시 bg_only 고정)
    if not getattr(args, 'crop', False):
        pipeline = 'bg_only'
    else:
        pipeline = prompt_pipeline(getattr(args, 'pipeline', None))
    args.pipeline = pipeline

    start_time = time.time()

    log("=" * 60)
    log("  pHash 기반 유사 이미지 그룹화")
    log("=" * 60)
    log(f"  입력        : {args.input}")
    log(f"  출력        : {args.output}")
    log(f"  방식        : {args.mode}")
    log(f"  임계값      : {args.threshold}")
    if args.retry:
        log(f"  retry 임계값: {args.retry_threshold}")
    if args.crop:
        log(f"  크롭 임계값 : {args.crop_threshold}")
        log(f"  크롭 패딩   : {args.padding}")
        log(f"  최소 크롭   : {args.min_crop_size}px")
        log(f"  파이프라인  : {pipeline}")
        if pipeline == 'bg_label_rep':
            log(f"  대표 임계값 : {args.rep_threshold}")
    log(f"  워커 수     : {args.workers}")
    log("")

    num_workers = args.workers

    # 단계 1) 디렉토리 감지 및 설정
    img_dir, label_dir = detect_dataset_structure(args.input, getattr(args, 'labels', None))

    # 단계 2) 이미지 스캔 및 목록
    all_images = scan_images(img_dir)
    if not all_images:
        log("오류: 이미지 파일이 없습니다.")
        return False
    log(f"전체 이미지: {len(all_images):,}개")

    # 단계 3) 라벨 매핑 및 분류
    labeled_files, unlabeled_files = match_labels(all_images, label_dir, getattr(args, 'images_dir_name', 'JPEGImages'))
    log(f"라벨 있는 이미지: {len(labeled_files):,}개")
    log(f"라벨 없는 이미지: {len(unlabeled_files):,}개")

    annotated_count = 0
    background_count = 0
    for img_path, lp in labeled_files:
        if lp and parse_yolo_label(lp):
            annotated_count += 1
        else:
            background_count += 1
    log(f"  - 어노테이션 있음: {annotated_count:,}개")
    log(f"  - 배경(어노테이션 없음): {background_count:,}개")

    # 단계 4) 처리 대상 결정
    label_mode = getattr(args, 'label_mode', 'all')
    process_bg    = label_mode in ('all', 'labeled') and labeled_files
    process_ul    = label_mode in ('all', 'unlabeled') and unlabeled_files
    process_crops = getattr(args, 'crop', False) and labeled_files

    if not process_bg and not process_ul and not process_crops:
        log("오류: 처리할 이미지가 없습니다.")
        return False

    report_data = {}
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    # 파이프라인 단계별 공유 변수 초기화
    bg_list_paths  = []   # background_list.txt 경로 목록
    bg_final_unique = []  # 배경 unique 이미지 목록 (label_list 에 포함)

    # 단계 5) unlabeled 그룹화 처리
    if process_ul:
        log("")
        log(f"=== 라벨 없는 이미지 그룹화 ({len(unlabeled_files):,}개) ===")

        ul_hashes = compute_hashes_parallel(unlabeled_files, num_workers)
        if ul_hashes:
            ul_groups, ul_unique = do_grouping(ul_hashes, args.threshold, args.mode, "(unlabeled)")
            ul_groups.sort(key=len, reverse=True)
            log(f"1차 그룹 {len(ul_groups):,}개 / unique {len(ul_unique):,}개")

            ul_retry_groups, ul_final_unique = [], ul_unique
            if getattr(args, 'retry', False) and len(ul_unique) >= 2:
                ul_retry_groups, ul_final_unique = retry_grouping_files(
                    ul_unique, ul_hashes, args.retry_threshold, args.mode, "(unlabeled)"
                )

            total = sum(len(g) for g in ul_groups + ul_retry_groups) + len(ul_final_unique)
            pbar = tqdm(total=total, desc="복사 (unlabeled)", unit="file", ncols=100,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

            copy_groups_with_rep(
                ul_groups, [], str(output_path / "unlabeled"), ul_hashes, pbar, "group"
            )

            if ul_retry_groups:
                copy_groups_with_rep(
                    ul_retry_groups, ul_final_unique,
                    str(output_path / "unlabeled"), ul_hashes, pbar, "retry_group"
                )
            else:
                if ul_final_unique:
                    unique_dir = output_path / "unlabeled" / "unique"
                    unique_dir.mkdir(parents=True, exist_ok=True)
                    for fp in ul_final_unique:
                        safe_copy(fp, unique_dir, Path(fp).name)
                        pbar.update(1)

            pbar.close()

            save_representative_list(
                ul_groups, ul_retry_groups, ul_final_unique, ul_hashes,
                args.output, "unlabeled_representative_list.txt"
            )
            save_representative_list(
                [], [], ul_final_unique, ul_hashes,
                args.output, "unlabeled_unique_list.txt"
            )
            ul_all_rep = collect_representative_files(
                ul_groups, ul_retry_groups, ul_final_unique, ul_hashes
            )
            generate_grid_preview_from_files(
                ul_all_rep, str(output_path / "unlabeled_preview.jpg"), "unlabeled"
            )

            report_data['unlabeled'] = {
                'groups': ul_groups, 'unique': ul_unique,
                'retry_groups': ul_retry_groups, 'final_unique': ul_final_unique,
                'hash_dict': ul_hashes
            }

    # 단계 6) background 그룹화 처리
    if process_bg:
        log("")
        bg_img_paths = [ip for ip, lp in labeled_files]
        log(f"=== background 그룹화 ({len(bg_img_paths):,}개) ===")

        bg_hashes = compute_hashes_parallel(bg_img_paths, num_workers)
        if bg_hashes:
            bg_groups, bg_unique = do_grouping(bg_hashes, args.threshold, args.mode, "(background)")
            bg_groups.sort(key=len, reverse=True)
            log(f"1차 그룹 {len(bg_groups):,}개 / unique {len(bg_unique):,}개")

            bg_retry_groups, bg_final_unique = [], bg_unique
            if getattr(args, 'retry', False) and len(bg_unique) >= 2:
                bg_retry_groups, bg_final_unique = retry_grouping_files(
                    bg_unique, bg_hashes, args.retry_threshold, args.mode, "(background)"
                )

            total = sum(len(g) for g in bg_groups + bg_retry_groups) + len(bg_final_unique)
            pbar = tqdm(total=total, desc="복사 (background)", unit="file", ncols=100,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

            copy_groups_with_rep(
                bg_groups, [], str(output_path / "background"), bg_hashes, pbar, "group"
            )

            if bg_retry_groups:
                copy_groups_with_rep(
                    bg_retry_groups, bg_final_unique,
                    str(output_path / "background"), bg_hashes, pbar, "retry_group"
                )
            else:
                if bg_final_unique:
                    unique_dir = output_path / "background" / "unique"
                    unique_dir.mkdir(parents=True, exist_ok=True)
                    for fp in bg_final_unique:
                        safe_copy(fp, unique_dir, Path(fp).name)
                        pbar.update(1)

            pbar.close()

            # ── background_list.txt : 그룹 대표만 (unique 제외) ──
            bg_rep_only = []
            for g in bg_groups + bg_retry_groups:
                rk, _ = select_representative(g, bg_hashes)
                bg_rep_only.append(rk)
            bg_list_paths = save_path_list(
                bg_rep_only, args.output,
                "background_list.txt",
                "배경 그룹 대표 이미지 (unique 제외 → label_list.txt 에 포함)"
            )

            # 기존 호환 파일도 유지
            save_representative_list(
                bg_groups, bg_retry_groups, bg_final_unique, bg_hashes,
                args.output, "background_representative_list.txt"
            )
            save_representative_list(
                [], [], bg_final_unique, bg_hashes,
                args.output, "background_unique_list.txt"
            )
            bg_all_rep = collect_representative_files(
                bg_groups, bg_retry_groups, bg_final_unique, bg_hashes
            )
            generate_grid_preview_from_files(
                bg_all_rep, str(output_path / "background_preview.jpg"), "background"
            )

            report_data['background'] = {
                'groups': bg_groups, 'unique': bg_unique,
                'retry_groups': bg_retry_groups, 'final_unique': bg_final_unique,
                'hash_dict': bg_hashes
            }

            # 배경 단계 전용 리포트
            gen_report_background(
                bg_groups, bg_retry_groups, bg_final_unique, bg_hashes,
                args.output, args
            )
        else:
            bg_list_paths = []
            bg_final_unique = []

    # 단계 7) 크롭(라벨) 그룹화 처리  (bg_label / bg_label_rep 파이프라인)
    label_list_paths = []      # label_list.txt 에 들어갈 원본 경로
    rep_crop_items   = []      # (crop_pil, source_key) — 대표 간 재그룹화용
    rep_crop_paths   = {}      # source_key → original_img_path
    label_class_data = {}      # gen_report_label 용 데이터
    label_stem_to_path = {}    # 전체 stem→path 통합

    if process_crops and pipeline in ('bg_label', 'bg_label_rep'):
        log("")
        log(f"=== 클래스별 객체 크롭 그룹화 ===")

        class_names = load_class_names(getattr(args, 'names', None))
        crops_by_class, stem_to_path = generate_crops(labeled_files, args.padding, args.min_crop_size)
        label_stem_to_path = stem_to_path
        total_crops = sum(len(v) for v in crops_by_class.values())
        log(f"크롭 생성 완료: {total_crops:,}개 ({len(crops_by_class)}개 클래스)")

        def _si_to_orig(si):
            """source_key(imgXXX_objY) → 원본 이미지 경로"""
            parts = si.rsplit('_obj', 1)
            return stem_to_path.get(parts[0]) if len(parts) == 2 else None

        for cls_id in sorted(crops_by_class.keys()):
            cls_name = get_class_name(cls_id, class_names)
            items = crops_by_class[cls_id]

            if len(items) < 2:
                log(f"  [{cls_name}] {len(items)}개 → 건너뜀")
                # 1개라도 unique로 label_list 에 추가
                for _, si in items:
                    parts = si.rsplit('_obj', 1)
                    if len(parts) == 2:
                        orig = stem_to_path.get(parts[0])
                        if orig:
                            label_list_paths.append(orig)
                continue

            log(f"  [{cls_name}] {len(items):,}개 크롭 처리 중..")

            crop_pil_map = {si: img for img, si in items}
            crop_hashes = compute_hashes_from_pil(items)
            if not crop_hashes:
                continue

            cr_groups, cr_unique = do_grouping(
                crop_hashes, args.crop_threshold, args.mode, f"({cls_name})"
            )
            cr_groups.sort(key=len, reverse=True)
            log(f"    1차 그룹 {len(cr_groups):,}개 / unique {len(cr_unique):,}개")

            cr_retry_groups, cr_final_unique = [], cr_unique
            if getattr(args, 'retry', False) and len(cr_unique) >= 2:
                cr_retry_groups, cr_final_unique = retry_grouping_crops(
                    cr_unique, crop_hashes, crop_pil_map,
                    args.retry_threshold, args.mode, f"({cls_name})"
                )

            total_cr = sum(len(g) for g in cr_groups + cr_retry_groups) + len(cr_final_unique)
            pbar = tqdm(total=total_cr, desc=f"저장 ({cls_name})", unit="crop", ncols=100,
                        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

            copy_crop_groups_with_rep(
                cr_groups, [], crop_pil_map, crop_hashes,
                str(output_path / "crops" / cls_name), pbar, "group"
            )

            if cr_retry_groups:
                copy_crop_groups_with_rep(
                    cr_retry_groups, cr_final_unique, crop_pil_map, crop_hashes,
                    str(output_path / "crops" / cls_name), pbar, "retry_group"
                )
            else:
                if cr_final_unique:
                    unique_dir = output_path / "crops" / cls_name / "unique"
                    unique_dir.mkdir(parents=True, exist_ok=True)
                    for si in cr_final_unique:
                        pil_img = crop_pil_map.get(si)
                        if pil_img:
                            pil_img.save(unique_dir / f"{si}.jpg", quality=95)
                        pbar.update(1)

            pbar.close()

            # 기존 호환 파일 유지
            save_crop_representative_list(
                cr_groups, cr_retry_groups, cr_final_unique, crop_hashes,
                stem_to_path, args.output,
                f"crops_{cls_name}_groups_representative_list.txt"
            )
            save_crop_representative_list(
                [], [], cr_final_unique, crop_hashes,
                stem_to_path, args.output,
                f"crops_{cls_name}_unique_list.txt"
            )

            report_data[f'crops/{cls_name}'] = {
                'groups': cr_groups, 'unique': cr_unique,
                'retry_groups': cr_retry_groups, 'final_unique': cr_final_unique,
                'hash_dict': crop_hashes
            }
            label_class_data[cls_name] = {
                'groups': cr_groups, 'retry_groups': cr_retry_groups,
                'final_unique': cr_final_unique, 'hash_dict': crop_hashes,
                'total_crops': len(items)
            }

            # label_list : 그룹 대표 원본 + unique 원본
            for g in cr_groups + cr_retry_groups:
                rk, _ = select_representative(g, crop_hashes)
                orig = _si_to_orig(rk)
                if orig:
                    label_list_paths.append(orig)
                # 대표 간 재그룹화용 수집
                if pipeline == 'bg_label_rep':
                    pil_img = crop_pil_map.get(rk)
                    if pil_img:
                        rep_crop_items.append((pil_img, rk))
                        if orig:
                            rep_crop_paths[rk] = orig

            for si in cr_final_unique:
                orig = _si_to_orig(si)
                if orig:
                    label_list_paths.append(orig)

            cr_all_rep_keys = []
            for g in cr_groups + cr_retry_groups:
                rk, _ = select_representative(g, crop_hashes)
                cr_all_rep_keys.append(rk)
            cr_all_rep_keys.extend(cr_final_unique)

            generate_grid_preview_from_pil(
                crop_pil_map, cr_all_rep_keys,
                str(output_path / "crops" / f"{cls_name}_preview.jpg"), cls_name
            )

        # bg_final_unique 이미지도 label_list 에 추가
        for fp in bg_final_unique:
            label_list_paths.append(fp)

        # label_list.txt 저장
        label_list_paths_saved = save_path_list(
            label_list_paths, args.output,
            "label_list.txt",
            "크롭 그룹 대표 + 크롭 unique + 배경 unique 원본 이미지"
        )

        # bg_label_list.txt = background_list + label_list (중복 제거)
        bg_label_combined = sorted(set(bg_list_paths) | set(label_list_paths_saved))
        save_path_list(
            bg_label_combined, args.output,
            "bg_label_list.txt",
            "배경 그룹 대표 + 라벨 그룹 대표 + unique 통합 목록"
        )
        log(f"bg_label_list.txt : {len(bg_label_combined):,}개")

        # 라벨 단계 전용 리포트
        gen_report_label(label_class_data, len(bg_final_unique), args.output, args)

    # 단계 8) 대표 간 재그룹화  (bg_label_rep 파이프라인)
    if process_crops and pipeline == 'bg_label_rep' and rep_crop_items:
        log("")
        log(f"=== 대표 크롭 간 재그룹화 ({len(rep_crop_items):,}개) ===")

        rep_hashes = compute_hashes_from_pil(rep_crop_items)
        if rep_hashes:
            cross_groups, cross_unique = do_grouping(
                rep_hashes, args.rep_threshold, args.mode, "(representative)"
            )
            cross_groups.sort(key=len, reverse=True)
            log(f"  재그룹 {len(cross_groups):,}개 / unique {len(cross_unique):,}개")

            cross_retry_groups, cross_final_unique = [], cross_unique
            if getattr(args, 'retry', False) and len(cross_unique) >= 2:
                cross_retry_groups, cross_final_unique = retry_grouping_crops(
                    cross_unique, rep_hashes,
                    {si: img for img, si in rep_crop_items},
                    args.retry_threshold, args.mode, "(representative)"
                )

            # representative_list.txt : 재그룹 대표 + unique 의 원본 경로
            rep_list_paths = []
            for g in cross_groups + cross_retry_groups:
                rk, _ = select_representative(g, rep_hashes)
                orig = rep_crop_paths.get(rk)
                if orig:
                    rep_list_paths.append(orig)
            for si in cross_final_unique:
                orig = rep_crop_paths.get(si)
                if orig:
                    rep_list_paths.append(orig)

            rep_list_saved = save_path_list(
                rep_list_paths, args.output,
                "representative_list.txt",
                "대표 크롭 간 재그룹화 후 최종 대표 원본 이미지"
            )

            # bg_label_rep_list.txt = bg_label_list + rep_list (중복 제거)
            bg_label_saved = [
                line.strip()
                for line in open(output_path / "bg_label_list.txt", encoding='utf-8')
                if line.strip() and not line.startswith('#')
            ] if (output_path / "bg_label_list.txt").exists() else []
            bg_label_rep_combined = sorted(set(bg_label_saved) | set(rep_list_saved))
            save_path_list(
                bg_label_rep_combined, args.output,
                "bg_label_rep_list.txt",
                "배경 + 라벨 + 대표 간 재그룹화 통합 목록 (최종 dedup)"
            )
            log(f"bg_label_rep_list.txt : {len(bg_label_rep_combined):,}개")

            gen_report_representative(
                cross_groups, cross_retry_groups, cross_final_unique,
                len(rep_crop_items), args.output, args
            )

            report_data['representative'] = {
                'groups': cross_groups, 'unique': cross_unique,
                'retry_groups': cross_retry_groups, 'final_unique': cross_final_unique,
                'hash_dict': rep_hashes
            }

    # 단계 9) 통합 리포트 + 통합 목록 저장
    elapsed = time.time() - start_time
    gen_report(report_data, args.output, args, elapsed)

    total_images = set()
    skip = {"total_representative_list.txt"}
    for list_file in sorted(output_path.glob("*_representative_list.txt")) + \
                     sorted(output_path.glob("*_unique_list.txt")):
        if list_file.name in skip:
            continue
        for line in open(list_file, 'r', encoding='utf-8'):
            line = line.strip()
            if line and not line.startswith('#'):
                total_images.add(line)
    if total_images:
        total_sorted = sorted(total_images)
        with open(output_path / "total_representative_list.txt", 'w', encoding='utf-8') as f:
            f.write(f"# 전체 통합 목록 (background + unlabeled + crops 대표경로, 중복제거)\n")
            f.write(f"# 총 {len(total_sorted):,}개\n")
            f.write(f"# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for p in total_sorted:
                f.write(f"{p}\n")
        log(f"전체 통합 리스트 {len(total_sorted):,}개")

    # 최종 결과 요약
    log("")
    log("=" * 60)
    log(f"  완료! (소요 시간: {elapsed:.1f}초)")
    log("=" * 60)
    for section, data in report_data.items():
        g1       = len(data.get('groups', []))
        g1_imgs  = sum(len(g) for g in data.get('groups', []))
        retry_g  = len(data.get('retry_groups', []))
        retry_imgs = sum(len(g) for g in data.get('retry_groups', []))
        final_u  = len(data.get('final_unique', data.get('unique', [])))

        line = f"  [{section}] 1차 그룹: {g1:,}개 ({g1_imgs:,}개)"
        if retry_g:
            line += f" / retry: {retry_g:,}개 ({retry_imgs:,}개)"
        line += f" / 고유: {final_u:,}개"
        log(line)

    # 파이프라인별 결과 파일 안내
    log("")
    log("  생성된 목록 파일")
    log(f"    background_list.txt         : 배경 그룹 대표")
    if pipeline in ('bg_label', 'bg_label_rep'):
        log(f"    label_list.txt              : 크롭 대표 + unique")
        log(f"    bg_label_list.txt           : 배경 + 라벨 통합")
    if pipeline == 'bg_label_rep':
        log(f"    representative_list.txt     : 대표 간 재그룹화")
        log(f"    bg_label_rep_list.txt       : 전체 통합 (최종)")
    log(f"  결과 디렉토리: {args.output}")
    log("=" * 60)

    return True


# ──────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='pHash 기반 유사/중복 이미지 그룹화 도구',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-i', '--input',    required=True,  help='입력 이미지 디렉토리')
    parser.add_argument('-o', '--output',   required=True,  help='출력 디렉토리')
    parser.add_argument('-t', '--threshold', type=int, default=1,
                        help='유사도 임계값 (기본값: 1)')
    parser.add_argument('-m', '--mode', choices=['chain', 'representative'], default='chain',
                        help='그룹화 알고리즘 (기본값: chain)')
    parser.add_argument('--label-mode', choices=['all', 'labeled', 'unlabeled'], default='all',
                        help='처리 대상 (기본값: all)')
    parser.add_argument('--labels',         default=None,   help='라벨 디렉토리 직접 지정 (없으면 자동 감지)')
    parser.add_argument('--names',          default=None,   help='클래스명 파일 (.txt)')
    parser.add_argument('--crop',           action='store_true', help='객체 크롭 그룹화 활성화')
    parser.add_argument('--crop-threshold', type=int, default=None,
                        help='크롭 유사도 임계값 (기본값: threshold와 동일)')
    parser.add_argument('--padding',        type=float, default=0.1,
                        help='크롭 패딩 비율 (기본값: 0.1)')
    parser.add_argument('--min-crop-size',  type=int, default=32, dest='min_crop_size',
                        help='최소 크롭 크기 px (기본값: 32)')
    parser.add_argument('--retry',          action='store_true', help='unique 이미지 재그룹화')
    parser.add_argument('--retry-threshold', type=int, default=None, dest='retry_threshold',
                        help='retry 유사도 임계값 (기본값: threshold * 2)')
    parser.add_argument('--pipeline',
                        choices=['bg_only', 'bg_label', 'bg_label_rep'], default=None,
                        help='처리 파이프라인 선택 (없으면 대화식 선택): '
                             'bg_only=배경만 / bg_label=배경+크롭 / bg_label_rep=배경+크롭+대표재그룹')
    parser.add_argument('--rep-threshold',  type=int, default=None, dest='rep_threshold',
                        help='대표 간 재그룹화 임계값 (기본값: crop_threshold와 동일)')
    parser.add_argument('-w', '--workers',  type=int, default=None,
                        help='병렬 워커 수 (기본값: CPU절반)')
    parser.add_argument('--images-dir-name', default='JPEGImages', dest='images_dir_name',
                        help='이미지 상위 디렉토리명 (기본값: JPEGImages)')

    args = parser.parse_args()

    # 워커 수 결정
    args.workers = prompt_workers(args.workers)

    # 실행
    success = run_grouper(args)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()