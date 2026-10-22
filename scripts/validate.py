#!/usr/bin/env python3
"""Read-only checks for the combined roadmap. No network or course execution."""
from pathlib import Path
from urllib.parse import unquote, urlsplit
import collections
import csv
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
errors = []
manifest = json.loads((ROOT / 'ORIGINALS_MANIFEST.json').read_text())
expected = {entry['archived_path'] for entry in manifest['files']}
for entry in manifest['files']:
    path = ROOT / entry['archived_path']
    if not path.is_file():
        errors.append(f'Missing original: {path}')
    elif path.stat().st_size != entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        errors.append(f'Original content changed: {path}')
original_dirs = {entry['archived_path'].split('/')[1] for entry in manifest['files']}
actual = {p.relative_to(ROOT).as_posix() for name in original_dirs for p in (ROOT / 'originals' / name).rglob('*') if p.is_file()}
if actual != expected:
    errors.append(f'Original inventory mismatch: missing={expected-actual}, extra={actual-expected}')
if manifest['file_count'] != len(expected):
    errors.append('Manifest count does not match its entries')

with (ROOT / 'schedule.csv').open(newline='') as f:
    rows = list(csv.DictReader(f))
if len(rows) != 273 or [int(r['day']) for r in rows] != list(range(1,274)):
    errors.append('Schedule must contain consecutive days 1–273')
weekly = collections.defaultdict(list)
minutes = collections.Counter()
for row in rows:
    w = int(row['week']); weekly[w].append(row)
    fields = ('technical_minutes','career_minutes','review_minutes')
    amounts = [int(row[k]) for k in fields]
    if sum(amounts) != int(row['total_minutes']):
        errors.append(f'Day {row["day"]}: time allocation mismatch')
    for k, amount in zip(fields, amounts): minutes[k] += amount
    if row['kind'] == 'study':
        for k in ('topic','local_resources','external_entry','task','completion_check','career_task','career_resource'):
            if not row[k]: errors.append(f'Day {row["day"]}: missing {k}')
    elif row['kind'] != 'rest' or int(row['total_minutes']) != 0:
        errors.append(f'Day {row["day"]}: invalid rest entry')
    for target in filter(None, row['local_resources'].split(';') + [row['career_resource']]):
        if not (ROOT / target).is_file(): errors.append(f'Day {row["day"]}: missing resource {target}')
    filename, anchor = row['week_file'].split('#')
    p = ROOT / filename
    if not p.is_file() or f'id="{anchor}"' not in p.read_text():
        errors.append(f'Day {row["day"]}: missing weekly anchor')
if sorted(weekly) != list(range(1,40)): errors.append('Week numbering mismatch')
for w, group in weekly.items():
    if len(group) != 7 or sum(r['kind']=='study' for r in group) != 6 or sum(int(r['total_minutes']) for r in group) != 1020:
        errors.append(f'Week {w}: expected six sessions + rest and 1,020 minutes')
    totals = [sum(int(r[k]) for r in group) for k in ('technical_minutes','career_minutes','review_minutes')]
    if totals != ([720,240,60] if w <= 16 else [840,120,60]):
        errors.append(f'Week {w}: unexpected category totals {totals}')
if sum(minutes.values()) != 39780: errors.append('Expected 663 total hours')

with (ROOT / 'trackers/daily-progress.csv').open(newline='') as f:
    progress = list(csv.DictReader(f))
# Extra user-added repair rows are allowed; scheduled rows must remain present.
progress_ids = {r['day'] for r in progress}
if not {r['day'] for r in rows}.issubset(progress_ids): errors.append('Progress tracker lacks scheduled days')

coverage = (ROOT / 'COVERAGE.md').read_text()
for prefix, count in [('P',62),('I',48),('E',24),('C',16)]:
    for i in range(1,count+1):
        code = f'{prefix}{i:02}'
        if len(re.findall(r'^\| '+code+r' \|', coverage, re.M)) != 1:
            errors.append(f'Coverage must contain exactly one row for {code}')

# Validate new Markdown references. Original documents are immutable inputs.
link_count = 0
new_markdown = [p for p in ROOT.rglob('*.md') if 'originals' not in p.relative_to(ROOT).parts]
new_markdown.append(ROOT / 'originals/README.md')
for p in new_markdown:
    for target in re.findall(r'\[[^\]]*\]\(([^\s)]+)\)',p.read_text()):
        if urlsplit(target).scheme or target.startswith('//'): continue
        target = unquote(target)
        filename, _, anchor = target.partition('#')
        dest = (p.parent / filename).resolve() if filename else p
        link_count += 1
        if not dest.exists(): errors.append(f'Broken local link in {p.relative_to(ROOT)}: {target}')
        elif anchor and dest.suffix == '.md' and 'originals' not in dest.relative_to(ROOT).parts:
            txt = dest.read_text()
            if f'id="{anchor}"' not in txt:
                # New docs link only explicit day anchors; heading support is useful for later edits.
                headings = re.findall(r'^#{1,6}\s+(.+)$', txt, re.M)
                slugs = [re.sub(r'[^\w\- ]','',h.lower()).replace(' ','-') for h in headings]
                if anchor not in slugs: errors.append(f'Broken anchor in {p.relative_to(ROOT)}: {target}')

if errors:
    print('\n'.join(errors))
    raise SystemExit(1)
print(f'PASS: {len(expected)} original files retain their exact size and SHA-256.')
print(f'PASS: 39 weeks; 234 study sessions + 39 rest days; 663 hours.')
print(f'PASS: time split = {minutes["technical_minutes"]/60:g} technical + {minutes["career_minutes"]/60:g} career + {minutes["review_minutes"]/60:g} review hours.')
print('PASS: all 150 original teaching units accounted for; active tracker contains all scheduled days.')
print(f'PASS: {link_count} local Markdown link occurrences and all CSV resource/anchor paths resolve.')
print('Not checked: external URL availability, original-document links, software execution, hardware compatibility or learning outcomes.')
