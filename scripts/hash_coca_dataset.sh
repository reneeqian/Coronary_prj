#!/usr/bin/env bash
# Hash key COCA dataset annotation files for DHF data integrity record.
# Run when /Volumes/rqian1TB is mounted. Output goes to DHF:
#   COCA-prj-DHF/13_ai_ml/training_data_description.md §6

set -euo pipefail

COCA_ROOT="${1:-/Volumes/rqian1TB/coca/cocacoronarycalciumandchestcts-2}"
GATED_XML="${COCA_ROOT}/Gated_release_final/calcium_xml"
NONGATED_SCORES="${COCA_ROOT}/deidentified_nongated/scores.xlsx"

if [[ ! -d "${COCA_ROOT}" ]]; then
    echo "ERROR: COCA root not found at ${COCA_ROOT}" >&2
    echo "Mount the external drive and retry." >&2
    exit 1
fi

echo "=== COCA Dataset SHA-256 Hashes ==="
echo "Date: $(date -u '+%Y-%m-%d')"
echo ""

echo "--- Gated XML manifest (calcium_xml/) ---"
# Hash a manifest of all XML files (one SHA per file, then hash the manifest itself)
MANIFEST=$(find "${GATED_XML}" -type f -name "*.xml" | sort | xargs shasum -a 256)
MANIFEST_HASH=$(echo "${MANIFEST}" | shasum -a 256 | awk '{print $1}')
echo "XML manifest SHA-256: ${MANIFEST_HASH}"
echo ""

echo "--- Nongated scores.xlsx ---"
shasum -a 256 "${NONGATED_SCORES}"
echo ""

echo "=== Paste the values above into COCA-prj-DHF/13_ai_ml/training_data_description.md §6 ==="
