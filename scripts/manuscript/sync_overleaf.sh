#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  scripts/manuscript/sync_overleaf.sh paper-en <overleaf-git-url>
  scripts/manuscript/sync_overleaf.sh thesis-th <overleaf-git-url>

Examples:
  scripts/manuscript/sync_overleaf.sh paper-en https://git.overleaf.com/PROJECT_ID
  scripts/manuscript/sync_overleaf.sh thesis-th https://git.overleaf.com/PROJECT_ID

Optional environment variables:
  REMOTE_BRANCH=master

This replaces the Overleaf project contents with the selected manuscript folder
plus manuscript/shared, then commits and pushes the result.
USAGE
}

if [[ $# -ne 2 ]]; then
  usage
  exit 2
fi

target="$1"
remote_url="$2"
remote_branch="${REMOTE_BRANCH:-master}"

case "$target" in
  paper-en|thesis-th)
    ;;
  *)
    echo "Unknown target: $target" >&2
    usage
    exit 2
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
manuscript_dir="$repo_root/manuscript"

if [[ ! -d "$manuscript_dir/$target" ]]; then
  echo "Missing manuscript folder: $manuscript_dir/$target" >&2
  exit 1
fi

if [[ ! -d "$manuscript_dir/shared" ]]; then
  echo "Missing shared manuscript folder: $manuscript_dir/shared" >&2
  exit 1
fi

tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/gbm-overleaf-${target}.XXXXXX")"
cleanup() {
  rm -rf "$tmp_root"
}
trap cleanup EXIT

project_dir="$tmp_root/project"
git clone "$remote_url" "$project_dir"

find "$project_dir" -mindepth 1 \
  ! -name .git \
  ! -path "$project_dir/.git/*" \
  -exec rm -rf {} +

cp -R "$manuscript_dir/$target" "$project_dir/$target"
cp -R "$manuscript_dir/shared" "$project_dir/shared"

(
  cd "$project_dir"
  git add -A
  if git diff --cached --quiet; then
    echo "No manuscript changes to push."
    exit 0
  fi

  git commit -m "Update $target manuscript from local workspace"
  git push origin "HEAD:$remote_branch"
)
