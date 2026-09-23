#!/usr/bin/env bash
set -euo pipefail
# Official rhysd/actionlint release asset digest, verified 2026-09-23.
# Update the version and digest together after reviewing the official release.
actionlint_tmp=$(mktemp -d)
trap 'rm -rf "$actionlint_tmp"' EXIT
curl --fail --silent --show-error --location --retry 3 --max-time 60 \
  https://github.com/rhysd/actionlint/releases/download/v1.7.12/actionlint_1.7.12_linux_amd64.tar.gz \
  -o "$actionlint_tmp/actionlint.tar.gz"
printf '%s  %s\n' '8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8' \
  "$actionlint_tmp/actionlint.tar.gz" | sha256sum --check --status
tar -xzf "$actionlint_tmp/actionlint.tar.gz" -C "$actionlint_tmp" actionlint
"$actionlint_tmp/actionlint" -color
