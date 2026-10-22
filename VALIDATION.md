# Validation record

Created 22 October 2026. Run `python3 scripts/validate.py` from `learning/` to repeat the read-only checks.

The validator checks:

- Every one of the 250 original files against the pre-move byte count and SHA-256, including hidden files; no extra/missing files inside the four archived packs.
- 39 consecutive weeks and 273 consecutive days, with 234 study sessions and 39 zero-hour rest days.
- Exactly 17 hours reserved each week, totaling 663 hours: 514 technical, 110 career and 39 review.
- Required task, completion-check, reading and career fields on every study day.
- Local resource paths and daily navigation anchors in the schedule.
- A coverage row for all 62 PyTorch modules, 48 inference modules, 24 edge weeks and 16 career weeks.
- Every new Markdown local link, including links into the archived source library.
- Presence of all planned day IDs in the active progress tracker; additional repair rows are allowed.

The curriculum was also reviewed for dependency order, reuse of projects, career time within the overall budget, explicit survey/deferred scope and hardware fallbacks. This is editorial validation, not proof that an estimated session duration will fit every learner.

Core external entry points were inspected separately as documented in [RESOURCES.md](RESOURCES.md). The validator does not access the network or validate all external links, old original-document links, builds, benchmarks, devices, models, hiring claims or learning outcomes. No course assignments or projects have been executed for you.
