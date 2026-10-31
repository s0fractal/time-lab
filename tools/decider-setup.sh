#!/bin/bash
# One-time owner setup of time-lab-decider (METHOD §2), run after decider-key.yml is on main:
#   1. the environment `decider`, deployable from main alone;
#   2. the owner's gh token lent to one decider-key.yml run as TL_SECRETS_TOKEN: piped from `gh auth token` into the
#      environment secret, never written to disk or printed, and deleted on exit whatever happens;
#   3. the run makes TL_DECIDER_SEED on the runner and prints its public key, which this script prints last.
# Binding that key under time-lab-decider in .warrant/v3.json is a separate, reviewed change.
set -euo pipefail
R=s0fractal/time-lab

gh api -X PUT "repos/$R/environments/decider" -F 'deployment_branch_policy[protected_branches]=false' \
  -F 'deployment_branch_policy[custom_branch_policies]=true' > /dev/null
gh api "repos/$R/environments/decider/deployment-branch-policies" --jq '.branch_policies[] | [.name, .type] | @tsv' \
  | grep -qx $'main\tbranch' \
  || gh api -X POST "repos/$R/environments/decider/deployment-branch-policies" -f name=main -f type=branch > /dev/null

trap 'gh secret delete TL_SECRETS_TOKEN --env decider --repo "$R" > /dev/null 2>&1 || true' EXIT
gh auth token | gh secret set TL_SECRETS_TOKEN --env decider --repo "$R"

before=$(gh run list --repo "$R" --workflow decider-key.yml --limit 1 --json databaseId --jq '.[0].databaseId // 0')
gh workflow run decider-key.yml --repo "$R" --ref main
id=$before
for _ in $(seq 60); do
  id=$(gh run list --repo "$R" --workflow decider-key.yml --limit 1 --json databaseId --jq '.[0].databaseId // 0')
  [ "$id" != "$before" ] && break
  sleep 2
done
[ "$id" != "$before" ] || { echo "decider-key.yml did not start" >&2; exit 1; }
gh run watch "$id" --repo "$R" --exit-status > /dev/null
gh secret delete TL_SECRETS_TOKEN --env decider --repo "$R"
echo "run $id; TL_SECRETS_TOKEN deleted"
gh run view "$id" --repo "$R" --log | grep -o 'time-lab-decider public key [0-9a-f]\{64\}' | tail -1
