# Mutation execution evidence and changed-line gate — 2026-10-22

Base: `3f017b521df7c3f330182229c63d9d0211fc8e9d`, the
[date/calendar follow-up](mutation-calendar-2026-10-22.md) in draft PR #172.
This change improves execution records and failure summaries. Rust production
code, Rust tests, mutation selection and acceptance rules are unchanged.

## Evidence survives abrupt termination

Previously `run.json` was written only in the runner's final cleanup. A process
termination that bypassed Python's exception/finally handlers could leave the
inventory and partial tool output without the runner's source/command record.
This was also observed during the real changed-line run below: `selected.json`
and the baseline log existed while `run.json` did not yet exist.

The runner now writes checkpoints before tool discovery, selection, execution
and result validation, retaining the metadata available at each stage. It writes
to a temporary file and atomically replaces `run.json`. A stop partway through
the next write therefore preserves the previous valid JSON. Normal completion
and handled failures add a finish timestamp and normalized runner exit code.

The stages `preparing`, `selecting`, `running` and `validating` describe the last
checkpoint. They do not establish that a process is still alive or that a run
passed. A hard stop leaves no finish timestamp or runner exit code. Handled
keyboard interrupts remain `interrupted` with exit 130. Each invocation still
gets a fresh directory, so it cannot inherit a previous passing result.

The regression test launches a child Python process and calls `os._exit(143)`
during selection and execution. Against the original runner, both cases lack
`run.json`; against the updated runner, both retain the source HEAD, command,
start time and nonterminal stage, with the inventory/count present once selected.
A separate test interrupts a checkpoint write and verifies preservation of the
previous JSON and recovery on the next successful write. These tests do not
depend on a graceful shutdown.

## Actionable, bounded failure summaries

After complete evidence passes the existing checker, `run.json` retains every
missed, timeout and unviable identity, the last execution phase and its log/patch
paths. The local and GitHub summaries display up to 20 entries, with missed
mutations first. The display limit never removes entries from the retained JSON.
Mutation names containing table delimiters, backticks or line breaks are escaped.

Partial or invalid reports do not receive a validated failure list. Build
failures, build timeouts and test timeouts retain their original categories;
they do not become caught mutations. Selection identity, package selection,
baseline checks and exit/count reconciliation are unchanged.

The updated formatter/checker was exercised on the retained complete calendar
and numerical reports, without rerunning or replacing those surveys:

| Report | Caught | Missed | Timeout | Unviable | Retained outstanding |
| --- | ---: | ---: | ---: | ---: | ---: |
| Date/calendar, 135 selected | 122 | 0 | 5 | 8 | 13 |
| Numerics, 952 selected | 931 | 11 | 3 | 7 | 21 |

All referenced log and patch files exist. The numerical summary shows 20 entries
and explicitly reports the additional entry retained in JSON. Both diagnostic
surveys retain exit 3 and remain failed; this reporting change does not fix or
reclassify their outstanding mutations.

## Broad changed-line gate

The normal runner was executed against the actual stacked PR base:

```bash
CARGO_PROFILE_TEST_DEBUG=0 python scripts/run_mutation_tests.py changed \
  --base 376c92624f88083b46d86e1087d2b0f03e202b91 \
  --output /tmp/pricing-calendar-changed-gate
```

This run started on the clean base commit before the reporting-only edits.
It uses the normal test scope, **both `pricing` and `pricing-numerics`**, including
unit, integration and doc-test targets, with all features and locked dependencies.
The complete report is revalidated with the updated checker. It is one execution
of the broad gate, not two executions or a new full-file survey.

The unmutated baseline passes **1,116 tests**, with **91 existing tests ignored**,
zero failures and zero filtered tests. The doc-test targets are included but
currently contain zero cases. The JSON retains every test-target result and the
actual baseline/mutant Cargo commands.

The changed-line selection contains exactly two mutations:

| Mutation | Outcome |
| --- | --- |
| Negate the corrected year/month comparison in `Calendar::adjust` | Caught by failed calendar assertions |
| Replace `Calendar::adjust` with `Ok(Default::default())` | Unviable: `Date` does not implement `Default` |

The changed-line gate **passes with exit 0: one caught and one unviable**, with
zero missed and zero timeout outcomes. The unviable mutation is not counted as
caught. This limited selection does not establish that the 135- or 952-mutation
diagnostic surveys pass, nor does it cover every library file.

## Verification

- Runner unit tests: **24 passed**, including both hard-exit stages, interrupted
  checkpoint writes, handled errors/interrupts, phase categories and bounded
  summaries with the full JSON list preserved.
- Installed cargo-mutants fixture: all four scenarios pass (four caught, four
  missed, a failed downstream baseline and an empty shard). Additional assertions
  verify terminal timestamps/exits and existing log/patch paths.
- Documentation links, staged source archive and diff checks pass.

The [JSON record](mutation-reporting-2026-10-22.json) pins reporting-source hashes,
regression logs, the complete changed-line inventory/outcomes, baseline target
results and the revalidated survey evidence. Earlier validation records are
preserved. Ignored acceptance and installed Python-wheel tests were not executed
by this gate. Remote CI is reported separately; the PR remains a draft.
