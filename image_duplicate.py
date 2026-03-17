#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
image_duplicate.py
pHash 기반 유사/중복 이미지 그룹핑 도구 (YOLO 라벨 지원) - GUI 통합

빌드: pyinstaller --name ImageDuplicate -F --windowed image_duplicate.py
"""

import os
import sys
import io
import shutil
import time
import logging
import random
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from PIL import Image
import imagehash

if sys.stdout is None:
    sys.stdout = io.StringIO()
if sys.stderr is None:
    sys.stderr = io.StringIO()

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff', '.tif'}
HASH_SIZE = 8

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)


def get_cpu_count():
    try:
        return os.cpu_count() or 4
    except Exception:
        return 4


# ══════════════════════════════════════════════
#  진행률 콜백 헬퍼
# ══════════════════════════════════════════════
class ProgressHelper:
    """단계별 진행률을 전체 진행률로 변환 + 남은시간 계산."""
    def __init__(self, log_fn=None, progress_fn=None):
        self.log_fn = log_fn
        self.progress_fn = progress_fn  # progress_fn(percent, stage_text)
        self._stages = []
        self._current_stage = 0
        self._stage_offset = 0.0
        self._start_time = time.time()

    def set_stages(self, stages):
        """stages: [(weight, name), ...] weight 합이 100."""
        self._stages = stages
        self._current_stage = 0
        self._stage_offset = 0.0
        self._start_time = time.time()

    def next_stage(self):
        if self._current_stage < len(self._stages):
            self._stage_offset += self._stages[self._current_stage][0]
            self._current_stage += 1

    def _format_time(self, seconds):
        if seconds < 60:
            return f"{int(seconds)}초"
        elif seconds < 3600:
            return f"{int(seconds//60)}분 {int(seconds%60)}초"
        else:
            return f"{int(seconds//3600)}시간 {int((seconds%3600)//60)}분"

    def update(self, current, total, detail=""):
        if not self.progress_fn or total == 0:
            return
        if self._current_stage < len(self._stages):
            w, name = self._stages[self._current_stage]
            stage_pct = (current / total) * w
            overall = self._stage_offset + stage_pct

            # 남은시간 계산
            elapsed = time.time() - self._start_time
            if overall > 0:
                est_total = elapsed / (overall / 100.0)
                remaining = est_total - elapsed
                remain_str = self._format_time(max(0, remaining))
            else:
                remain_str = "계산 중..."

            stage_num = f"[{self._current_stage+1}/{len(self._stages)}]"
            self.progress_fn(int(min(overall, 100)),
                             f"{stage_num} {name} {detail} | 남은시간: {remain_str}")

    def log(self, msg):
        if self.log_fn:
            self.log_fn(msg)
        else:
            print(msg)


# ══════════════════════════════════════════════
#  엔진
# ══════════════════════════════════════════════
def detect_dataset_structure(input_dir, labels_override=None, img_folder_name="JPEGImages"):
    ip = Path(input_dir)
    jd = ip / img_folder_name
    img_dir = jd if jd.exists() and jd.is_dir() else ip
    label_dir = None
    if labels_override:
        lp = Path(labels_override)
        if lp.exists():
            label_dir = lp
    else:
        # 이미지 폴더명을 labels로 치환하여 라벨 폴더 탐지
        if img_dir != ip:
            auto_label = ip / "labels"
            if auto_label.exists() and auto_label.is_dir():
                label_dir = auto_label
        else:
            al = ip / "labels"
            if al.exists() and al.is_dir():
                label_dir = al
    return img_dir, label_dir


def scan_images(img_dir):
    files = sorted(str(f.resolve()) for f in Path(img_dir).rglob('*')
                   if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS)
    return files


def match_labels(image_files, label_dir):
    labeled, unlabeled = [], []
    for ip in image_files:
        lp = Path(label_dir) / f"{Path(ip).stem}.txt"
        if lp.exists():
            labeled.append((ip, str(lp)))
        else:
            unlabeled.append(ip)
    return labeled, unlabeled


def parse_yolo_label(label_path):
    objs = []
    try:
        with open(label_path, 'r') as f:
            for line in f:
                p = line.strip().split()
                if len(p) >= 5:
                    objs.append((int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])))
    except Exception:
        pass
    return objs


def crop_object(img, cx, cy, bw, bh, padding, min_size):
    iw, ih = img.size
    pw, ph = bw * iw * padding, bh * ih * padding
    x1 = int(max(0, cx * iw - bw * iw / 2 - pw))
    y1 = int(max(0, cy * ih - bh * ih / 2 - ph))
    x2 = int(min(iw, cx * iw + bw * iw / 2 + pw))
    y2 = int(min(ih, cy * ih + bh * ih / 2 + ph))
    if (x2 - x1) < min_size or (y2 - y1) < min_size:
        return None
    return img.crop((x1, y1, x2, y2))


def generate_crops(labeled_files, padding, min_size, ph=None):
    """반환: (crops_by_class, stem_to_path)
    stem_to_path: {img_stem: original_img_path} 역추적용"""
    cbc = defaultdict(list)
    stem_to_path = {}
    total = len(labeled_files)
    for i, (ip, lp) in enumerate(labeled_files):
        objs = parse_yolo_label(lp)
        if not objs:
            continue
        try:
            img = Image.open(ip).convert("RGB")
        except Exception:
            continue
        stem = Path(ip).stem
        stem_to_path[stem] = ip
        for oi, (cid, cx, cy, bw, bh) in enumerate(objs):
            cr = crop_object(img, cx, cy, bw, bh, padding, min_size)
            if cr:
                cbc[cid].append((cr, f"{stem}_obj{oi}"))
        img.close()
        if ph:
            ph.update(i + 1, total, f"{i+1}/{total}")
    return cbc, stem_to_path


def compute_hash_single(filepath):
    try:
        with Image.open(filepath) as img:
            return (filepath, imagehash.phash(img.convert("RGB"), hash_size=HASH_SIZE))
    except Exception as e:
        return (filepath, None, str(e))


def compute_hashes_parallel(file_list, num_workers, ph=None):
    results, errors = {}, []
    total = len(file_list)
    done = 0
    with ThreadPoolExecutor(max_workers=num_workers) as ex:
        futs = {ex.submit(compute_hash_single, fp): fp for fp in file_list}
        for fut in as_completed(futs):
            r = fut.result()
            done += 1
            if len(r) == 3:
                errors.append((r[0], r[2]))
            else:
                fp, h = r
                if h is not None:
                    results[fp] = h
            if ph and done % 50 == 0:
                ph.update(done, total, f"{done}/{total}")
    if ph:
        ph.update(total, total, f"{total}/{total}")
    if errors:
        logger.warning(f"해시 실패: {len(errors)}개")
    return results


def compute_hashes_from_pil(pil_items, ph=None):
    results = {}
    total = len(pil_items)
    for i, (pimg, si) in enumerate(pil_items):
        try:
            results[si] = imagehash.phash(pimg, hash_size=HASH_SIZE)
        except Exception:
            pass
        if ph and (i + 1) % 50 == 0:
            ph.update(i + 1, total, f"{i+1}/{total}")
    if ph:
        ph.update(total, total)
    return results


class UnionFind:
    def __init__(self, elems):
        self.parent = {e: e for e in elems}
        self.rank = {e: 0 for e in elems}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry: return
        if self.rank[rx] < self.rank[ry]: rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]: self.rank[rx] += 1

    def groups(self):
        g = defaultdict(list)
        for e in self.parent: g[self.find(e)].append(e)
        return dict(g)


def select_representative(group, hd):
    if len(group) <= 1:
        return group[0], []
    hs = {k: hd[k] for k in group}
    best, ba = group[0], float('inf')
    for c in group:
        a = sum(hs[c] - hs[o] for o in group if o != c) / (len(group) - 1)
        if a < ba: ba, best = a, c
    return best, [k for k in group if k != best]


def group_chain(hd, threshold, desc="", ph=None):
    keys = list(hd.keys())
    hs = [hd[k] for k in keys]
    n = len(keys)
    uf = UnionFind(range(n))
    total = n * (n - 1) // 2
    done = 0
    for i in range(n):
        for j in range(i + 1, n):
            if hs[i] - hs[j] <= threshold:
                uf.union(i, j)
            done += 1
            if ph and done % 50000 == 0:
                ph.update(done, total, f"{done}/{total}")
    if ph:
        ph.update(total, total)
    groups, unique = [], []
    for ms in uf.groups().values():
        ps = [keys[m] for m in ms]
        if len(ps) >= 2:
            groups.append(ps)
        else:
            unique.append(ps[0])
    return groups, unique


def group_rep_mode(hd, threshold, desc="", ph=None):
    keys = list(hd.keys())
    gl = []
    total = len(keys)
    for i, key in enumerate(keys):
        h = hd[key]
        matched = False
        for rh, ms in gl:
            if h - rh <= threshold:
                ms.append(key)
                matched = True
                break
        if not matched:
            gl.append((h, [key]))
        if ph and (i + 1) % 200 == 0:
            ph.update(i + 1, total, f"{i+1}/{total}")
    if ph:
        ph.update(total, total)
    groups, unique = [], []
    for _, ms in gl:
        if len(ms) >= 2:
            groups.append(ms)
        else:
            unique.append(ms[0])
    return groups, unique


def do_grouping(hd, threshold, mode, desc="", ph=None):
    return group_chain(hd, threshold, desc, ph) if mode == 'chain' \
        else group_rep_mode(hd, threshold, desc, ph)


def safe_copy(src, dst_dir, filename):
    dst = dst_dir / filename
    if dst.exists():
        s, x = Path(filename).stem, Path(filename).suffix
        c = 1
        while dst.exists():
            dst = dst_dir / f"{s}_dup{c}{x}"; c += 1
    shutil.copy2(src, dst)


def copy_groups_with_rep(groups, unique, base_dir, hd, ph=None, prefix="group"):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    total = sum(len(g) for g in groups) + len(unique)
    done = 0
    for idx, g in enumerate(groups, 1):
        gd = base / f"{prefix}_{idx:04d}"
        gd.mkdir(parents=True, exist_ok=True)
        rep, dups = select_representative(g, hd)
        shutil.copy2(rep, gd / f"representative{Path(rep).suffix}")
        done += 1
        if dups:
            dd = gd / "duplicates"; dd.mkdir(parents=True, exist_ok=True)
            for fp in dups:
                safe_copy(fp, dd, Path(fp).name); done += 1
        if ph and done % 20 == 0:
            ph.update(done, total)
    if unique:
        ud = base / "unique"; ud.mkdir(parents=True, exist_ok=True)
        for fp in unique:
            safe_copy(fp, ud, Path(fp).name); done += 1
            if ph and done % 20 == 0:
                ph.update(done, total)
    if ph:
        ph.update(total, total)


def copy_crop_groups_with_rep(groups, unique, pm, hd, base_dir, ph=None, prefix="group"):
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)
    total = sum(len(g) for g in groups) + len(unique)
    done = 0
    for idx, g in enumerate(groups, 1):
        gd = base / f"{prefix}_{idx:04d}"
        gd.mkdir(parents=True, exist_ok=True)
        rep, dups = select_representative(g, hd)
        pi = pm.get(rep)
        if pi: pi.save(gd / "representative.jpg", quality=95)
        done += 1
        if dups:
            dd = gd / "duplicates"; dd.mkdir(parents=True, exist_ok=True)
            for si in dups:
                pi2 = pm.get(si)
                if pi2: pi2.save(dd / f"{si}.jpg", quality=95)
                done += 1
        if ph and done % 20 == 0:
            ph.update(done, total)
    if unique:
        ud = base / "unique"; ud.mkdir(parents=True, exist_ok=True)
        for si in unique:
            pi = pm.get(si)
            if pi: pi.save(ud / f"{si}.jpg", quality=95)
            done += 1
    if ph:
        ph.update(total, total)


def retry_grouping(ukeys, hd, rthresh, mode, desc="", ph=None):
    if len(ukeys) < 2:
        return [], ukeys
    sub = {k: hd[k] for k in ukeys if k in hd}
    if len(sub) < 2:
        return [], ukeys
    rg, fu = do_grouping(sub, rthresh, mode, f"retry {desc}", ph)
    rg.sort(key=len, reverse=True)
    return rg, fu


def collect_reps(groups, retry_groups, final_unique, hd):
    reps = []
    for g in groups:
        r, _ = select_representative(g, hd); reps.append(r)
    for g in retry_groups:
        r, _ = select_representative(g, hd); reps.append(r)
    reps.extend(final_unique)
    return reps


def save_rep_list(groups, rgs, fuq, hd, out_dir, name):
    ap = sorted(set(collect_reps(groups, rgs, fuq, hd)))
    lp = Path(out_dir) / name
    with open(lp, 'w', encoding='utf-8') as f:
        # f.write(f"# 정제 목록: {len(ap)}장 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        for p in ap:
            f.write(f"{p}\n")
    return ap


def gen_grid_files(flist, outpath, cell=128, cols=10, rows=10):
    mx = cols * rows
    if not flist: return
    samp = random.sample(flist, min(len(flist), mx))
    imgs = []
    for fp in samp:
        try: imgs.append(Image.open(fp).convert("RGB").resize((cell, cell), Image.LANCZOS))
        except: pass
    if not imgs: return
    c = min(cols, len(imgs)); r = (len(imgs) + c - 1) // c
    g = Image.new('RGB', (c * cell, r * cell), (40, 40, 40))
    for i, im in enumerate(imgs):
        g.paste(im, ((i % c) * cell, (i // c) * cell))
    g.save(Path(outpath), quality=90)


def gen_grid_pil(pm, keys, outpath, cell=128, cols=10, rows=10):
    mx = cols * rows
    if not keys: return
    samp = random.sample(keys, min(len(keys), mx))
    imgs = []
    for k in samp:
        pi = pm.get(k)
        if pi:
            try: imgs.append(pi.copy().resize((cell, cell), Image.LANCZOS))
            except: pass
    if not imgs: return
    c = min(cols, len(imgs)); r = (len(imgs) + c - 1) // c
    g = Image.new('RGB', (c * cell, r * cell), (40, 40, 40))
    for i, im in enumerate(imgs):
        g.paste(im, ((i % c) * cell, (i // c) * cell))
    g.save(Path(outpath), quality=90)


def gen_report(rd, out_dir, args, elapsed):
    rp = Path(out_dir) / "report.txt"
    with open(rp, 'w', encoding='utf-8') as f:
        f.write(f"pHash 그룹핑 리포트 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
        f.write(f"입력:{args.input} 출력:{args.output} 모드:{args.mode} 임계값:{args.threshold}\n")
        f.write(f"소요:{elapsed:.1f}초\n\n")
        for sec, d in rd.items():
            gs, fu, rgs = d.get('groups', []), d.get('final_unique', d.get('unique', [])), d.get('retry_groups', [])
            hd = d.get('hash_dict', {})
            f.write(f"[{sec}] 그룹:{len(gs)} retry:{len(rgs)} 고유:{len(fu)}\n")
            for idx, g in enumerate(gs[:20], 1):
                rep, dups = select_representative(g, hd) if hd else (g[0], g[1:])
                bh = hd.get(rep)
                nm = Path(rep).name if '/' in str(rep) or os.sep in str(rep) else rep
                f.write(f"  [group_{idx:04d}]({len(g)}장) ★{nm}\n")
                for it in dups:
                    ih = hd.get(it); dist = (bh - ih) if bh and ih else "?"
                    f.write(f"    {Path(it).name if '/' in str(it) or os.sep in str(it) else it} d={dist}\n")
            f.write("\n")


def load_class_names(p):
    if not p: return {}
    try: return {i: l.strip() for i, l in enumerate(open(p, 'r', encoding='utf-8')) if l.strip()}
    except: return {}


# ══════════════════════════════════════════════
#  run_grouper
# ══════════════════════════════════════════════
def run_grouper(args, log_fn=None, progress_fn=None):
    """
    log_fn(msg): 로그 텍스트 콜백
    progress_fn(percent, stage_text): 진행률 콜백 (0~100)
    """
    ph = ProgressHelper(log_fn, progress_fn)

    if getattr(args, 'crop_threshold', None) is None:
        args.crop_threshold = args.threshold
    if getattr(args, 'retry_threshold', None) is None:
        args.retry_threshold = args.threshold * 2

    start = time.time()
    ph.log(f"입력: {args.input}")
    ph.log(f"출력: {args.output}")
    ph.log(f"모드: {args.mode} / 임계값: {args.threshold} / 워커: {args.workers}")
    if getattr(args, 'retry', False):
        ph.log(f"retry 임계값: {args.retry_threshold}")
    ph.log("")

    img_dir, label_dir = detect_dataset_structure(
        args.input, getattr(args, 'labels', None),
        getattr(args, 'img_folder_name', 'JPEGImages')
    )
    all_imgs = scan_images(img_dir)
    if not all_imgs:
        ph.log("오류: 이미지 없음"); return False
    ph.log(f"이미지 폴더: {img_dir}")
    ph.log(f"전체 이미지: {len(all_imgs):,}장")

    labeled, unlabeled = [], []
    if label_dir:
        labeled, unlabeled = match_labels(all_imgs, label_dir)
        ph.log(f"라벨 폴더: {label_dir}")
        ph.log(f"라벨 있음: {len(labeled):,}장 / 없음: {len(unlabeled):,}장")
    else:
        ph.log("라벨 폴더 없음 → 전체를 라벨 없는 것으로 처리")
        unlabeled = all_imgs

    lm = getattr(args, 'label_mode', 'all')
    do_bg = lm in ('all', 'labeled') and labeled
    do_ul = lm in ('all', 'unlabeled') and unlabeled
    do_crop = getattr(args, 'crop', False) and labeled

    if not do_bg and not do_ul and not do_crop:
        ph.log("처리할 이미지 없음"); return False

    # 단계 가중치 설정 (대략적 시간 비율)
    stages = []
    if do_ul: stages.append((20, "unlabeled 해시"))
    if do_ul: stages.append((10, "unlabeled 그룹핑"))
    if do_ul: stages.append((5, "unlabeled 복사"))
    if do_bg: stages.append((20, "background 해시"))
    if do_bg: stages.append((10, "background 그룹핑"))
    if do_bg: stages.append((5, "background 복사"))
    if do_crop: stages.append((15, "크롭 생성"))
    if do_crop: stages.append((10, "크롭 해시/그룹핑"))
    if do_crop: stages.append((5, "크롭 저장"))
    # 남는 비율을 보정
    total_w = sum(s[0] for s in stages)
    if total_w > 0:
        stages = [(int(s[0] * 100 / total_w), s[1]) for s in stages]
    ph.set_stages(stages)

    rd = {}
    op = Path(args.output)
    op.mkdir(parents=True, exist_ok=True)

    def process_section(files, sec_name, subdir):
        ph.log(f"\n=== {sec_name} ({len(files):,}장) ===")

        # 해시
        hashes = compute_hashes_parallel(files, args.workers, ph)
        if not hashes: return
        ph.log(f"해시 완료: {len(hashes):,}장")
        ph.next_stage()

        # 그룹핑
        gs, uq = do_grouping(hashes, args.threshold, args.mode, f"({sec_name})", ph)
        gs.sort(key=len, reverse=True)
        ph.log(f"1차: 그룹 {len(gs):,}개, unique {len(uq):,}개")

        rgs, fuq = [], uq
        if getattr(args, 'retry', False) and len(uq) >= 2:
            rgs, fuq = retry_grouping(uq, hashes, args.retry_threshold, args.mode, f"({sec_name})")
            if rgs:
                ph.log(f"retry: 그룹 {len(rgs):,}개, 최종고유 {len(fuq):,}개")
        ph.next_stage()

        # 복사
        copy_groups_with_rep(gs, [], str(op / subdir), hashes, ph, "group")
        if rgs:
            copy_groups_with_rep(rgs, fuq, str(op / subdir), hashes, ph, "retry_group")
        else:
            if fuq:
                ud = op / subdir / "unique"; ud.mkdir(parents=True, exist_ok=True)
                for fp in fuq:
                    safe_copy(fp, ud, Path(fp).name)

        save_rep_list(gs, rgs, fuq, hashes, args.output, f"{subdir}_representative_list.txt")
        arep = collect_reps(gs, rgs, fuq, hashes)
        gen_grid_files(arep, str(op / f"{subdir}_preview.jpg"), sec_name)
        ph.log(f"{sec_name} 완료: 리스트 {len(arep):,}장")
        ph.next_stage()

        rd[sec_name] = {'groups': gs, 'unique': uq, 'retry_groups': rgs,
                        'final_unique': fuq, 'hash_dict': hashes}

    if do_ul:
        process_section(unlabeled, "unlabeled", "unlabeled")
    if do_bg:
        process_section([ip for ip, lp in labeled], "background", "background")

    if do_crop:
        ph.log(f"\n=== 클래스별 객체 크롭 ===")
        cn = load_class_names(getattr(args, 'names', None))
        cbc, stem_to_path = generate_crops(labeled, args.padding, args.min_crop_size, ph)
        ph.log(f"크롭: {sum(len(v) for v in cbc.values()):,}개 ({len(cbc)}클래스)")
        ph.next_stage()

        # 전체 클래스 통합 원본 이미지 수집용
        all_crop_rep_origins = set()

        for cid in sorted(cbc.keys()):
            cname = cn.get(cid, f"class_{cid}")
            items = cbc[cid]
            if len(items) < 2: continue
            ph.log(f"  [{cname}] {len(items):,}개")
            pm = {si: im for im, si in items}
            ch = compute_hashes_from_pil(items, ph)
            if not ch: continue

            cg, cu = do_grouping(ch, args.crop_threshold, args.mode, f"({cname})", ph)
            cg.sort(key=len, reverse=True)
            crg, cfu = [], cu
            if getattr(args, 'retry', False) and len(cu) >= 2:
                crg, cfu = retry_grouping(cu, ch, args.retry_threshold, args.mode, f"({cname})")

            ph.log(f"    그룹:{len(cg)} retry:{len(crg)} 고유:{len(cfu)}")

            copy_crop_groups_with_rep(cg, [], pm, ch, str(op / "crops" / cname), ph, "group")
            if crg:
                copy_crop_groups_with_rep(crg, cfu, pm, ch, str(op / "crops" / cname), ph, "retry_group")
            else:
                if cfu:
                    ud = op / "crops" / cname / "unique"; ud.mkdir(parents=True, exist_ok=True)
                    for si in cfu:
                        pi = pm.get(si)
                        if pi: pi.save(ud / f"{si}.jpg", quality=95)

            rd[f'crops/{cname}'] = {'groups': cg, 'unique': cu, 'retry_groups': crg,
                                     'final_unique': cfu, 'hash_dict': ch}
            ark = collect_reps(cg, crg, cfu, ch)
            gen_grid_pil(pm, ark, str(op / "crops" / f"{cname}_preview.jpg"), cname)

            # 클래스별 대표 크롭의 원본 이미지 리스트
            cls_origins = set()
            for source_info in ark:
                # source_info: "imgStem_objN" → stem 추출
                parts = source_info.rsplit("_obj", 1)
                if parts:
                    stem = parts[0]
                    orig = stem_to_path.get(stem)
                    if orig:
                        cls_origins.add(orig)
                        all_crop_rep_origins.add(orig)

            # 클래스별 리스트 저장
            cls_list_path = op / "crops" / f"{cname}_representative_list.txt"
            cls_sorted = sorted(cls_origins)
            with open(cls_list_path, 'w', encoding='utf-8') as f:
                f.write(f"# {cname} 대표 크롭이 포함된 원본 이미지 ({len(cls_sorted)}장)\n")
                f.write(f"# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                for p in cls_sorted:
                    f.write(f"{p}\n")
            ph.log(f"    리스트: {cls_list_path.name} ({len(cls_sorted)}장)")

        # 전체 클래스 통합 리스트 (원본 기준 중복 제거)
        if all_crop_rep_origins:
            all_sorted = sorted(all_crop_rep_origins)
            all_list_path = op / "crops_representative_images.txt"
            with open(all_list_path, 'w', encoding='utf-8') as f:
                # f.write(f"# 대표 객체가 포함된 원본 이미지 (전 클래스 통합, 중복 제거)\n")
                # f.write(f"# 총 {len(all_sorted)}장\n")
                # f.write(f"# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                for p in all_sorted:
                    f.write(f"{p}\n")
            ph.log(f"크롭 통합 리스트: {len(all_sorted):,}장")

        ph.next_stage()

    elapsed = time.time() - start
    gen_report(rd, args.output, args, elapsed)

    # 통합 리스트: background 대표 + crops 대표 원본 (중복 제거)
    combined_images = set()
    bg_list_path = op / "background_representative_list.txt"
    if bg_list_path.exists():
        for line in open(bg_list_path, 'r', encoding='utf-8'):
            line = line.strip()
            if line and not line.startswith('#'):
                combined_images.add(line)
    crop_list_path = op / "crops_representative_images.txt"
    if crop_list_path.exists():
        for line in open(crop_list_path, 'r', encoding='utf-8'):
            line = line.strip()
            if line and not line.startswith('#'):
                combined_images.add(line)
    if combined_images:
        combined_sorted = sorted(combined_images)
        comb_path = op / "combined_representative_list.txt"
        with open(comb_path, 'w', encoding='utf-8') as f:
            # f.write(f"# 통합 정제 목록 (background 대표 + crops 대표 원본, 중복 제거)\n")
            # f.write(f"# 총 {len(combined_sorted)}장\n")
            # f.write(f"# 생성: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            for p in combined_sorted:
                f.write(f"{p}\n")
        ph.log(f"통합 리스트: {len(combined_sorted):,}장")

    ph.log(f"\n완료! ({elapsed:.1f}초)")
    for sec, d in rd.items():
        g1 = len(d.get('groups', []))
        g1i = sum(len(g) for g in d.get('groups', []))
        rg = len(d.get('retry_groups', []))
        fu = len(d.get('final_unique', d.get('unique', [])))
        ph.log(f"  [{sec}] 그룹:{g1}({g1i}장) retry:{rg} 고유:{fu}")
    if progress_fn:
        progress_fn(100, "완료")
    return True


# ══════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QPushButton, QLineEdit, QComboBox, QSpinBox,
    QDoubleSpinBox, QCheckBox, QFileDialog, QTextEdit, QProgressBar,
    QScrollArea, QGridLayout, QGroupBox, QMessageBox, QDialog, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QPalette, QColor

STYLE = """
QMainWindow{background:#1e1e2e}
QTabWidget::pane{border:1px solid #313244;background:#1e1e2e}
QTabBar::tab{background:#313244;color:#cdd6f4;padding:10px 20px;margin-right:2px;
  border-top-left-radius:6px;border-top-right-radius:6px;font-size:13px;font-weight:bold}
QTabBar::tab:selected{background:#45475a;color:#89b4fa}
QGroupBox{color:#cdd6f4;border:1px solid #313244;border-radius:8px;margin-top:12px;
  padding-top:20px;font-weight:bold}
QGroupBox::title{subcontrol-origin:margin;left:12px;padding:0 6px}
QLabel{color:#cdd6f4;font-size:12px}
QLineEdit{background:#313244;color:#cdd6f4;border:1px solid #45475a;border-radius:6px;padding:6px 10px}
QLineEdit:focus{border-color:#89b4fa}
QPushButton{background:#89b4fa;color:#1e1e2e;border:none;border-radius:6px;padding:8px 16px;
  font-size:12px;font-weight:bold}
QPushButton:hover{background:#b4d0fb}
QPushButton:disabled{background:#45475a;color:#6c7086}
QPushButton#browseBtn{background:#45475a;color:#cdd6f4;padding:6px 12px}
QPushButton#stopBtn{background:#f38ba8;color:#1e1e2e}
QComboBox{background:#313244;color:#cdd6f4;border:1px solid #45475a;border-radius:6px;padding:6px 10px}
QComboBox QAbstractItemView{background:#313244;color:#cdd6f4}
QSpinBox,QDoubleSpinBox{background:#313244;color:#cdd6f4;border:1px solid #45475a;
  border-radius:6px;padding:6px 10px}
QCheckBox{color:#cdd6f4;font-size:12px}
QTextEdit{background:#11111b;color:#a6e3a1;border:1px solid #313244;border-radius:6px;
  font-family:Consolas,monospace;font-size:11px;padding:8px}
QProgressBar{background:#313244;border:none;border-radius:8px;height:20px;text-align:center;color:#cdd6f4}
QProgressBar::chunk{background:#89b4fa;border-radius:8px}
QScrollArea{background:#1e1e2e;border:none}
QScrollBar:vertical{background:#1e1e2e;width:10px}
QScrollBar::handle:vertical{background:#45475a;border-radius:5px;min-height:30px}
QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0}
"""


class GrouperArgs:
    pass


class GrouperWorker(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, args_obj):
        super().__init__()
        self.args_obj = args_obj
        self._stop = False

    def run(self):
        try:
            def log_cb(msg):
                if self._stop:
                    raise InterruptedError("중지됨")
                self.log_signal.emit(str(msg))

            def prog_cb(pct, stage=""):
                if self._stop:
                    raise InterruptedError("중지됨")
                self.progress_signal.emit(pct, stage)

            ok = run_grouper(self.args_obj, log_fn=log_cb, progress_fn=prog_cb)
            self.finished_signal.emit(bool(ok), "완료" if ok else "실패")
        except InterruptedError as e:
            self.finished_signal.emit(False, str(e))
        except Exception as e:
            self.finished_signal.emit(False, f"오류: {e}")

    def stop(self):
        self._stop = True


class ThumbnailWidget(QLabel):
    clicked = pyqtSignal(str)

    def __init__(self, img_path, size=140):
        super().__init__()
        self.image_path = img_path
        self.setFixedSize(size, size)
        self.setCursor(Qt.PointingHandCursor)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("QLabel{background:#313244;border:2px solid #45475a;border-radius:8px}"
                           "QLabel:hover{border-color:#89b4fa}")
        try:
            px = QPixmap(img_path)
            if not px.isNull():
                self.setPixmap(px.scaled(size-8, size-8, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except:
            self.setText("ERR")

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self.clicked.emit(self.image_path)


class GroupDetailDialog(QDialog):
    def __init__(self, gdir, parent=None):
        super().__init__(parent)
        self.setWindowTitle(Path(gdir).name); self.setMinimumSize(700, 500); self.setStyleSheet(STYLE)
        ly = QVBoxLayout(self)
        reps = list(Path(gdir).glob("representative.*"))
        if reps:
            ly.addWidget(QLabel("★ 대표 이미지"))
            ly.addWidget(ThumbnailWidget(str(reps[0]), 200), alignment=Qt.AlignCenter)
        dd = Path(gdir) / "duplicates"
        if dd.exists():
            dfs = sorted(f for f in dd.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS)
            if dfs:
                ly.addWidget(QLabel(f"중복 ({len(dfs)}장)"))
                sc = QScrollArea(); sc.setWidgetResizable(True)
                sw = QWidget(); gl = QGridLayout(sw); gl.setSpacing(6)
                for i, fp in enumerate(dfs):
                    gl.addWidget(ThumbnailWidget(str(fp), 120), i // 5, i % 5)
                sc.setWidget(sw); ly.addWidget(sc)
        cb = QPushButton("닫기"); cb.clicked.connect(self.close)
        ly.addWidget(cb, alignment=Qt.AlignRight)


class ImageGridViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.vl = QVBoxLayout(self); self.vl.setContentsMargins(0, 0, 0, 0)
        self.info = QLabel(""); self.info.setStyleSheet("color:#a6adc8;padding:4px")
        self.vl.addWidget(self.info)
        self.sc = QScrollArea(); self.sc.setWidgetResizable(True)
        self.sw = QWidget(); self.grid = QGridLayout(self.sw)
        self.grid.setSpacing(8); self.grid.setContentsMargins(8, 8, 8, 8)
        self.sc.setWidget(self.sw); self.vl.addWidget(self.sc)

    def clear(self):
        while self.grid.count():
            it = self.grid.takeAt(0)
            if it.widget(): it.widget().deleteLater()
        self.info.setText("")

    def _make_labeled_thumb(self, img_path, label_text, border_color, size=140):
        """이미지 + 하단 라벨을 가진 위젯 생성."""
        container = QWidget()
        container.setFixedSize(size, size + 20)
        vl = QVBoxLayout(container)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(2)

        t = ThumbnailWidget(img_path, size)
        t.setStyleSheet(f"QLabel{{background:#313244;border:2px solid {border_color};border-radius:8px}}"
                        "QLabel:hover{border-color:#89b4fa}")
        vl.addWidget(t)

        lb = QLabel(label_text)
        lb.setAlignment(Qt.AlignCenter)
        lb.setStyleSheet(f"color:{border_color};font-size:10px;font-weight:bold;padding:0")
        vl.addWidget(lb)

        return container, t

    def load_groups(self, bdir, ts=140):
        self.clear()
        base = Path(bdir)
        if not base.exists():
            self.info.setText("결과 없음"); return
        gds = sorted(d for d in base.iterdir() if d.is_dir() and
                      (d.name.startswith("group_") or d.name.startswith("retry_group_")))
        ud = base / "unique"
        ufs = sorted(f for f in ud.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS) if ud.exists() else []
        self.info.setText(f"그룹 대표: {len(gds)} / 고유: {len(ufs)} / 합계: {len(gds)+len(ufs)}")
        cols, idx = 5, 0

        for gd in gds:
            reps = list(gd.glob("representative.*"))
            if reps:
                # 그룹 내 중복 수 계산
                dup_dir = gd / "duplicates"
                dup_count = len(list(dup_dir.iterdir())) if dup_dir.exists() else 0
                label = f"그룹 ({dup_count+1}장)"
                is_retry = gd.name.startswith("retry_")
                color = "#f9e2af" if is_retry else "#89b4fa"  # retry는 노란색
                if is_retry:
                    label = f"retry ({dup_count+1}장)"

                container, thumb = self._make_labeled_thumb(str(reps[0]), label, color, ts)
                thumb.clicked.connect(partial(self._show, str(gd)))
                thumb.setToolTip(f"{gd.name} | 대표 + 중복 {dup_count}장\n클릭→상세")
                self.grid.addWidget(container, idx // cols, idx % cols); idx += 1

        for uf in ufs:
            container, thumb = self._make_labeled_thumb(str(uf), "고유", "#a6e3a1", ts)
            thumb.setToolTip(f"고유: {uf.name}")
            self.grid.addWidget(container, idx // cols, idx % cols); idx += 1

    def _show(self, gd):
        GroupDetailDialog(gd, self).exec_()


class MainWindow(QMainWindow):
    # def __init__(self):
    #     super().__init__()
    #     self.setWindowTitle("pHash 유사 이미지 그룹핑")
    #     self.setMinimumSize(1100, 750)
    #     self.worker = None
    #     c = QWidget(); self.setCentralWidget(c)
    #     ml = QVBoxLayout(c); ml.setContentsMargins(0, 0, 0, 0)
    #     self.tabs = QTabWidget(); ml.addWidget(self.tabs)
    #     self.tabs.setStyleSheet("QTabBar::tab { padding-left: 20px; padding-right: 20px; min-height: 30px; }")
    #     self._build_settings(); self._build_bg(); self._build_label()
    def __init__(self):
        super().__init__()
        # 1. 기본 창 설정
        self.setWindowTitle("pHash 유사 이미지 그룹핑")
        self.setMinimumSize(1100, 750)
        self.worker = None

        # 2. 중앙 위젯 및 메인 레이아웃 설정
        c = QWidget()
        self.setCentralWidget(c)
        ml = QVBoxLayout(c)
        ml.setContentsMargins(0, 0, 0, 0)

        # 3. 탭 위젯 생성 및 스타일 적용
        self.tabs = QTabWidget()
        
        # 스타일시트: 세로(padding 상하, min-height)와 가로(padding 좌우, min-width) 모두 확장
        self.tabs.setStyleSheet("""
            QTabBar::tab { 
                padding: 12px 25px;      /* 상하 12px, 좌우 25px로 여백 증대 */
                min-width: 120px;       /* 가로 최소 너비 */
                min-height: 25px;       /* 세로 최소 높이를 45px로 설정 */
                font-size: 14px;        /* 글자 크기도 살짝 키워 가독성 확보 */
                font-weight: bold;      /* 글자를 두껍게 하여 잘 보이게 설정 */
            }
        """)
        
        # 탭 바가 압축되지 않도록 설정
        self.tabs.tabBar().setExpanding(False)
        
        ml.addWidget(self.tabs)

        # 4. 각 탭의 UI 빌드
        self._build_settings()
        self._build_bg()
        self._build_label()

    def _build_settings(self):
        tab = QWidget(); ly = QVBoxLayout(tab); ly.setSpacing(8); ly.setContentsMargins(16, 16, 16, 16)
        fg = QGroupBox("폴더 설정"); fl = QGridLayout(fg)
        fl.setColumnStretch(1, 1)  # 입력란이 넓게 펼쳐지도록
        fl.addWidget(QLabel("입력 폴더:"), 0, 0)
        self.inp = QLineEdit(); self.inp.setPlaceholderText("데이터셋 상위 폴더 (하위에 이미지/labels 자동 탐지)")
        fl.addWidget(self.inp, 0, 1)
        b1 = QPushButton("찾기"); b1.setObjectName("browseBtn")
        b1.clicked.connect(lambda: self._br(self.inp)); fl.addWidget(b1, 0, 2)

        fl.addWidget(QLabel("출력 폴더:"), 1, 0)
        self.out = QLineEdit(); self.out.setPlaceholderText("결과 저장 폴더")
        fl.addWidget(self.out, 1, 1)
        b2 = QPushButton("찾기"); b2.setObjectName("browseBtn")
        b2.clicked.connect(lambda: self._br(self.out)); fl.addWidget(b2, 1, 2)

        fl.addWidget(QLabel("라벨 폴더:"), 2, 0)
        self.labels_dir = QLineEdit(); self.labels_dir.setPlaceholderText("비워두면 입력폴더/labels 자동 탐지")
        fl.addWidget(self.labels_dir, 2, 1)
        b_lbl = QPushButton("찾기"); b_lbl.setObjectName("browseBtn")
        b_lbl.clicked.connect(lambda: self._br(self.labels_dir)); fl.addWidget(b_lbl, 2, 2)

        fl.addWidget(QLabel("이미지 폴더명:"), 3, 0)
        self.img_folder = QLineEdit(); self.img_folder.setText("JPEGImages")
        self.img_folder.setPlaceholderText("JPEGImages")
        self.img_folder.setToolTip("입력 폴더 안의 이미지 하위 폴더명 (기본: JPEGImages, 예: images, train)")
        fl.addWidget(self.img_folder, 3, 1, 1, 2)

        fl.addWidget(QLabel("클래스명:(사용 불필요)"), 4, 0)
        self.names = QLineEdit(); self.names.setPlaceholderText("classes.txt (선택)")
        fl.addWidget(self.names, 4, 1)
        b3 = QPushButton("찾기"); b3.setObjectName("browseBtn")
        b3.clicked.connect(self._bf); fl.addWidget(b3, 4, 2)
        ly.addWidget(fg)

        gg = QGroupBox("그룹핑 설정"); gl = QGridLayout(gg)
        gl.addWidget(QLabel("모드:"), 0, 0)
        self.mode = QComboBox()
        self.mode.addItem("chain - 전이적 연결 (정확, 수천 장 이하)", "chain")
        self.mode.addItem("representative - 대표 비교 (빠름, 수만 장 이상)", "representative")
        gl.addWidget(self.mode, 0, 1, 1, 3)
        gl.addWidget(QLabel("임계값(해밍거리)):"), 1, 0)
        self.thresh = QSpinBox(); self.thresh.setRange(0, 64); self.thresh.setValue(5)
        gl.addWidget(self.thresh, 1, 1)
        gl.addWidget(QLabel("라벨 모드[ALL(배경,라벨),배경,라벨 선택]:"), 1, 2)
        self.lmode = QComboBox()
        self.lmode.addItem("all", "all"); self.lmode.addItem("labeled", "labeled")
        self.lmode.addItem("unlabeled", "unlabeled")
        gl.addWidget(self.lmode, 1, 3)
        tc = get_cpu_count()
        gl.addWidget(QLabel(f"워커[CPU CORE 수] (가용:{tc}):"), 2, 0)
        self.wk = QSpinBox(); self.wk.setRange(1, tc); self.wk.setValue(max(1, tc // 2))
        gl.addWidget(self.wk, 2, 1)
        ly.addWidget(gg)

        eg = QGroupBox("라벨/유니크 자동 재시도"); el = QGridLayout(eg)
        self.crop_chk = QCheckBox("클래스별 객체 라벨 분석"); el.addWidget(self.crop_chk, 0, 0, 1, 2)
        self.retry_chk = QCheckBox("유니크(유사 이미지가 없는 이미지) 자동 재시도"); el.addWidget(self.retry_chk, 0, 2, 1, 2)
        el.addWidget(QLabel("크롭 패딩:"), 1, 0)
        self.pad = QDoubleSpinBox(); self.pad.setRange(0, 1); self.pad.setValue(0.1); self.pad.setSingleStep(0.05)
        el.addWidget(self.pad, 1, 1)
        el.addWidget(QLabel("유니크 자동 재시도 임계값:"), 1, 2)
        self.rthresh = QSpinBox(); self.rthresh.setRange(0, 64); self.rthresh.setValue(10)
        el.addWidget(self.rthresh, 1, 3)
        el.addWidget(QLabel("라벨(해밍거리) 임계값:"), 2, 0)
        self.cthresh = QSpinBox(); self.cthresh.setRange(0, 64); self.cthresh.setValue(5)
        el.addWidget(self.cthresh, 2, 1)
        el.addWidget(QLabel("최소 라벨 크기:"), 2, 2)
        self.mincrop = QSpinBox(); self.mincrop.setRange(8, 512); self.mincrop.setValue(32); self.mincrop.setSuffix("px")
        el.addWidget(self.mincrop, 2, 3)
        ly.addWidget(eg)

        rl = QHBoxLayout()
        self.run_btn = QPushButton("▶  실행"); self.run_btn.setFixedHeight(40)
        self.run_btn.setStyleSheet("QPushButton{background:#a6e3a1;color:#1e1e2e;font-size:14px;"
                                    "font-weight:bold;border-radius:8px}"
                                    "QPushButton:hover{background:#b8edb3}"
                                    "QPushButton:disabled{background:#45475a;color:#6c7086}")
        self.run_btn.clicked.connect(self._run); rl.addWidget(self.run_btn)
        self.stop_btn = QPushButton("■  중지"); self.stop_btn.setObjectName("stopBtn")
        self.stop_btn.setFixedHeight(40); self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._stop); rl.addWidget(self.stop_btn)
        self.load_btn = QPushButton("결과 불러오기"); self.load_btn.setFixedHeight(40)
        self.load_btn.clicked.connect(self._load_result); rl.addWidget(self.load_btn)
        ly.addLayout(rl)

        self.prog = QProgressBar(); self.prog.setValue(0); self.prog.setFormat("%p% %v")
        ly.addWidget(self.prog)
        self.stage_label = QLabel(""); self.stage_label.setStyleSheet("color:#a6adc8;font-size:11px")
        ly.addWidget(self.stage_label)
        self.log = QTextEdit(); self.log.setReadOnly(True); self.log.setMaximumHeight(200)
        ly.addWidget(self.log)
        ly.addStretch()
        self.tabs.addTab(tab, "설정 / 실행")

    def _build_bg(self):
        tab = QWidget(); ly = QVBoxLayout(tab); ly.setContentsMargins(8, 8, 8, 8)
        tl = QHBoxLayout(); tl.addWidget(QLabel("보기:"))
        self.bgtype = QComboBox()
        self.bgtype.addItem("background (라벨 있는 이미지)")
        self.bgtype.addItem("unlabeled (라벨 없는 이미지)")
        self.bgtype.currentIndexChanged.connect(self._bg_ch); tl.addWidget(self.bgtype); tl.addStretch()
        ob = QPushButton("폴더 열기"); ob.setObjectName("browseBtn")
        ob.clicked.connect(self._open_bg); tl.addWidget(ob)
        ly.addLayout(tl)
        self.bgv = ImageGridViewer(); ly.addWidget(self.bgv)
        self.tabs.addTab(tab, "Background")

    def _build_label(self):
        tab = QWidget(); ly = QVBoxLayout(tab); ly.setContentsMargins(8, 8, 8, 8)
        tl = QHBoxLayout(); tl.addWidget(QLabel("클래스:"))
        self.cls_cb = QComboBox(); self.cls_cb.setMinimumWidth(200)
        self.cls_cb.currentIndexChanged.connect(self._cls_ch); tl.addWidget(self.cls_cb); tl.addStretch()
        ob = QPushButton("폴더 열기"); ob.setObjectName("browseBtn")
        ob.clicked.connect(self._open_crop); tl.addWidget(ob)
        ly.addLayout(tl)
        self.lblv = ImageGridViewer(); ly.addWidget(self.lblv)
        self.tabs.addTab(tab, "Label (크롭)")

    def _br(self, le):
        f = QFileDialog.getExistingDirectory(self, "폴더")
        if f: le.setText(f)

    def _bf(self):
        f, _ = QFileDialog.getOpenFileName(self, "파일", "", "Text(*.txt);;All(*)")
        if f: self.names.setText(f)

    def _run(self):
        i, o = self.inp.text().strip(), self.out.text().strip()
        if not i or not o:
            QMessageBox.warning(self, "오류", "입출력 폴더를 지정하세요."); return
        a = GrouperArgs()
        a.input, a.output = i, o
        a.threshold = self.thresh.value(); a.mode = self.mode.currentData()
        a.workers = self.wk.value(); a.label_mode = self.lmode.currentData()
        a.labels = self.labels_dir.text().strip() or None
        a.names = self.names.text().strip() or None
        a.img_folder_name = self.img_folder.text().strip() or "JPEGImages"
        a.crop = self.crop_chk.isChecked(); a.padding = self.pad.value()
        a.crop_threshold = self.cthresh.value(); a.min_crop_size = self.mincrop.value()
        a.retry = self.retry_chk.isChecked(); a.retry_threshold = self.rthresh.value()

        self.log.clear(); self.prog.setValue(0); self.stage_label.setText("")
        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.worker = GrouperWorker(a)
        self.worker.log_signal.connect(self._on_log)
        self.worker.progress_signal.connect(self._on_prog)
        self.worker.finished_signal.connect(self._on_done)
        self.worker.start()

    def _stop(self):
        if self.worker: self.worker.stop()
        self.stop_btn.setEnabled(False)

    def _on_log(self, t):
        self.log.append(t)
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def _on_prog(self, pct, stage):
        self.prog.setValue(pct)
        if stage:
            self.stage_label.setText(stage)

    def _on_done(self, ok, msg):
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)
        self.prog.setValue(100 if ok else self.prog.value())
        self.stage_label.setText("완료" if ok else msg)
        self.log.append(f"\n{'✅' if ok else '❌'} {msg}")
        if ok:
            self._load_results(); self.tabs.setCurrentIndex(1)

    def _load_result(self):
        o = self.out.text().strip()
        if not o:
            f = QFileDialog.getExistingDirectory(self, "결과 폴더")
            if f: self.out.setText(f)
            else: return
        self._load_results(); self.tabs.setCurrentIndex(1)

    def _load_results(self):
        o = self.out.text().strip()
        if not o: return
        self._bg_ch(self.bgtype.currentIndex())
        self.cls_cb.blockSignals(True); self.cls_cb.clear()
        cd = Path(o) / "crops"
        if cd.exists():
            for d in sorted(cd.iterdir()):
                if d.is_dir(): self.cls_cb.addItem(d.name, str(d))
        self.cls_cb.blockSignals(False)
        if self.cls_cb.count() > 0: self._cls_ch(0)

    def _bg_ch(self, idx):
        o = self.out.text().strip()
        if not o: return
        self.bgv.load_groups(str(Path(o) / ("background" if idx == 0 else "unlabeled")))

    def _open_bg(self):
        o = self.out.text().strip()
        if not o: return
        t = Path(o) / ("background" if self.bgtype.currentIndex() == 0 else "unlabeled")
        if t.exists() and sys.platform == 'win32': os.startfile(str(t))

    def _cls_ch(self, idx):
        if idx < 0: return
        d = self.cls_cb.currentData()
        if d: self.lblv.load_groups(d)

    def _open_crop(self):
        d = self.cls_cb.currentData()
        if d and Path(d).exists() and sys.platform == 'win32': os.startfile(d)


def gui_main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    p = QPalette()
    p.setColor(QPalette.Window, QColor("#1e1e2e"))
    p.setColor(QPalette.WindowText, QColor("#cdd6f4"))
    p.setColor(QPalette.Base, QColor("#313244"))
    p.setColor(QPalette.Text, QColor("#cdd6f4"))
    p.setColor(QPalette.Highlight, QColor("#89b4fa"))
    app.setPalette(p)
    w = MainWindow(); w.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    gui_main()
