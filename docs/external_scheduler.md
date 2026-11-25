# External scheduler for GitHub Actions

GitHub-hosted cron schedules in `.github/workflows/*.yml` routinely drift by
5–30+ minutes during busy periods. The in-repo crons stay as a safety net,
but the primary trigger is an external scheduler (any always-on host: a
small VPS, home server, `cron` on a laptop that is left on, or a cloud
scheduler such as GitHub Actions on another repo, Cloudflare Workers Cron,
PythonAnywhere scheduled tasks, etc.) that calls the GitHub REST API at the
times below.

All three workflows accept `workflow_dispatch` with an optional `run_date`
input. When that input is omitted:

* **weekly_run** resolves `run_date` to the most-recent Sunday in UTC
  (same rule as the Monday backup cron), so Monday morning fires reprocess
  Sunday’s data rather than today’s.
* **watchdog** resolves `run_date` to the most-recent Sunday in UTC.
* **monday_orders** reads `data/weekly/` and picks the latest
  `data/weekly/<Sunday>/` whose `summary.json` still has at least one
  strategy without a live orders file. If every recent Sunday is already
  fully submitted the script early-outs with “nothing to do”, so firing the
  retry slots on a day when the first attempt succeeded is a no-op.

## Dispatch calls

Each workflow is triggered via the REST endpoint

```
POST /repos/miloszksiazek/llm-analyst-thesis/actions/workflows/<file>/dispatches
```

with headers

```
Accept: application/vnd.github+json
Authorization: Bearer <TOKEN>
X-GitHub-Api-Version: 2022-11-28
```

and a JSON body of the form

```json
{ "ref": "main", "inputs": { ... } }
```

GitHub returns `204 No Content` on success. There is no polling needed —
check Actions UI for the resulting run.

### weekly_run

```bash
curl -sS -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/miloszksiazek/llm-analyst-thesis/actions/workflows/weekly_run.yml/dispatches \
  -d '{"ref":"main","inputs":{}}'
```

Optional inputs:
* `run_date` (string, `YYYY-MM-DD`) — override the default most-recent
  Sunday.
* `mode` (`production` | `test`) — `production` writes to `data/weekly/`,
  `test` writes to `data/test_runs/` and never touches `data/weekly/`.
* `allow_late` (`true` | `false`) — permit running after the first NYSE
  open on-or-after `run_date` (normally refused to avoid leakage).

### watchdog

```bash
curl -sS -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/miloszksiazek/llm-analyst-thesis/actions/workflows/watchdog.yml/dispatches \
  -d '{"ref":"main","inputs":{}}'
```

Optional input:
* `run_date` (string, `YYYY-MM-DD`) — override the default most-recent
  Sunday.

### monday_orders

```bash
curl -sS -X POST \
  -H "Accept: application/vnd.github+json" \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  https://api.github.com/repos/miloszksiazek/llm-analyst-thesis/actions/workflows/monday_orders.yml/dispatches \
  -d '{"ref":"main","inputs":{}}'
```

Optional inputs:
* `run_date` (string, `YYYY-MM-DD`) — override the default
  “latest-incomplete-Sunday”.
* `dry_run` (`true` | `false`) — force dry-run even if
  `config/settings.yaml:broker_live=true`. Dry-run ledgers land in
  `data/scratch/orders_dryrun/<run_date>/` and never touch `data/weekly/`.

## Token

Create a **fine-grained personal access token** scoped to the single repo
`miloszksiazek/llm-analyst-thesis`:

* Resource owner: `miloszksiazek`
* Repository access: *Only select repositories* → `llm-analyst-thesis`
* Repository permissions:
  * **Actions: Read and write** (required to dispatch workflows)
  * All other permissions: *No access*
* Account permissions: *No access*
* Expiration: set the shortest window that covers the live test (28 weeks
  of forward test + buffer; set a calendar reminder to rotate before
  expiry).

Store the token in the scheduler host’s secret manager (or a `.env` with
600 perms), never in the repo. Rotate immediately if it leaks.

## Recommended trigger times (Europe/Warsaw)

GitHub cron times are in UTC and drift. The external scheduler should fire
in local time so it tracks DST without re-editing cron lines. Times below
are Europe/Warsaw; they translate to the Actions runs appearing within a
minute or two of the local time, not within 5–30 minutes of a UTC slot.

| Workflow        | Day(s) | Local time (Europe/Warsaw) | Purpose |
|-----------------|--------|----------------------------|---------|
| `weekly_run`    | Sun    | 18:40                      | Primary Sunday run — package data, agents, XGBoost |
| `weekly_run`    | Mon    | 06:10                      | Backup — reprocesses Sunday’s data if Sunday slot was skipped (resolves `run_date` to Sunday automatically) |
| `watchdog`      | Mon    | 08:20                      | First check that `data/weekly/<Sunday>/summary.json` is present and all strategies are `ok` |
| `watchdog`      | Mon    | 12:20                      | Second check before NYSE opens |
| `monday_orders` | Mon–Fri| 16:05                      | Primary order submission — just after NYSE open (14:30 UTC winter, 13:30 UTC summer) |
| `monday_orders` | Mon–Fri| 18:05                      | Retry if the 16:05 attempt failed or the live ledger was only partially written |
| `monday_orders` | Mon–Fri| 20:05                      | Backup retry late in the US session |

`monday_orders` is safe to fire on Tue–Fri: the broker script exits early
if the market is closed, and the new default `run_date` resolution exits
early if every strategy in the latest `data/weekly/<Sunday>/` already has
a live orders file. The idempotency gate (live orders files under
`data/weekly/<Sunday>/orders/<strategy>.json`) prevents duplicate
submissions even across concurrent dispatches.

## Example: minimal shell cron on a scheduler host

```
# /etc/cron.d/llm-analyst-thesis  (TZ=Europe/Warsaw)
TZ=Europe/Warsaw
GITHUB_TOKEN=... # from secret manager
REPO=miloszksiazek/llm-analyst-thesis

40 18 * * 0  user  curl -sS -X POST -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/$REPO/actions/workflows/weekly_run.yml/dispatches -d '{"ref":"main","inputs":{}}'
10  6 * * 1  user  curl -sS -X POST -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/$REPO/actions/workflows/weekly_run.yml/dispatches -d '{"ref":"main","inputs":{}}'
20  8 * * 1  user  curl -sS -X POST -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/$REPO/actions/workflows/watchdog.yml/dispatches -d '{"ref":"main","inputs":{}}'
20 12 * * 1  user  curl -sS -X POST -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/$REPO/actions/workflows/watchdog.yml/dispatches -d '{"ref":"main","inputs":{}}'
 5 16 * * 1-5 user curl -sS -X POST -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/$REPO/actions/workflows/monday_orders.yml/dispatches -d '{"ref":"main","inputs":{}}'
 5 18 * * 1-5 user curl -sS -X POST -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/$REPO/actions/workflows/monday_orders.yml/dispatches -d '{"ref":"main","inputs":{}}'
 5 20 * * 1-5 user curl -sS -X POST -H "Authorization: Bearer $GITHUB_TOKEN" -H "Accept: application/vnd.github+json" -H "X-GitHub-Api-Version: 2022-11-28" https://api.github.com/repos/$REPO/actions/workflows/monday_orders.yml/dispatches -d '{"ref":"main","inputs":{}}'
```

The in-repo cron schedules remain as a backup so a failure of the external
scheduler host does not silently drop a week of data.
