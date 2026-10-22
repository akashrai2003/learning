# Combined AI systems study roadmap

**39 weeks · about 9 months · 17 hours/week · 663 planned hours**

A single plan for your PyTorch, inference engineering, C++, deployment and AI career materials. It follows your chosen **15–20 hour weekly budget** and combines **ML systems/inference engineering with broad PyTorch competence and career preparation**. Edge deployment is a later applied branch of the same work.

Start with [START_HERE.md](START_HERE.md), then open [Week 01, Day 001](weeks/week-01.md#day-001). Use this plan as your schedule. The original schedules are a reference library, not additional homework.

## Your main navigation

| Need | Open |
|---|---|
| Understand the sequence and monthly outcomes | [Full roadmap](ROADMAP.md) |
| Find today's reading, task and completion check | [39 weekly daily plans](WEEK_INDEX.md) |
| Filter all 273 calendar days in a spreadsheet | [Daily schedule](schedule.csv) |
| Record actual work and blockers | [Daily tracker](trackers/daily-progress.csv) and [weekly progress](PROGRESS.md) |
| Decide what counts as done | [Milestones](MILESTONES.md) |
| Fit study and career work around your job | [Study system and pacing](STUDY_SYSTEM.md) |
| Know what to build and reuse | [Connected projects](PROJECTS.md) |
| Find resources and understand verification limits | [Resource guide](RESOURCES.md) |
| See where every original topic went | [Coverage map](COVERAGE.md) |
| Continue into deeper specializations | [Extensions](EXTENSIONS.md) |

## The working rhythm

- Five 2.5-hour sessions and one 4.5-hour session, which you can split into two blocks.
- One protected rest day. The long session includes the weekly review.
- Weeks 1–16: **12 technical + 4 career + 1 review hour**.
- Weeks 17–39: **14 technical + 2 career + 1 review hour**.
- Weeks **12, 24 and 36** are repair/consolidation weeks, already included in the total.

There are **234 study sessions and 39 rest days**, with no assumed start date. Use Day 001 on the day you actually start; use the day labels rather than trying to synchronize with dates in an older curriculum. Optional work up to your 20-hour ceiling goes to an unfinished prerequisite, not another course.

## What this plan aims to produce

A reproducible serving benchmark and reliability lab; a verified reference decoder; a quantization and optimization workbench; a focused deployment/capstone dossier; and one upstream-quality contribution. Interviews, applications and portfolio evidence start early and use those same artifacts.

This is selected depth across all four curricula. It does **not** promise completion of every original exercise, mastery of every backend, an accepted PR or a job offer in nine months. The [coverage map](COVERAGE.md) distinguishes focused assignments, surveys and deferred work.

## Original materials

All four original directories now live together in [originals/](originals/README.md). Their **250 files are byte-for-byte preserved**, including the hidden `.zip` file. Sibling-directory relationships were retained, so original relative cross-curriculum links keep their structure. The move changes absolute paths and old bookmarks; no compatibility symlinks were added.

[ORIGINALS_MANIFEST.json](ORIGINALS_MANIFEST.json) records old paths, new paths, sizes and SHA-256 checksums. Original content may contain dated links, old schedules or machine-specific assumptions; follow the combined plan's pacing and verify hardware/software when a lab begins.

To check preservation, schedule totals and local links again:

```bash
python3 learning/scripts/validate.py
```

Run that command from the parent of `learning/`, or run `python3 scripts/validate.py` from inside it. See [VALIDATION.md](VALIDATION.md) for what was actually checked.
