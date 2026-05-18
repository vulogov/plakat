#!/usr/bin/env bash
# Fetch public-domain exemplar images from Wikimedia Commons for the style-
# detection spike. Idempotent — re-running skips files already on disk.
#
# Output: tests/fixtures/style_catalog/{watercolor,photorealistic,holdout}/
#
# Two styles in the spike, each backed by 4 exemplar images, plus 2 holdout
# images (one per style) for the smoke test. All images are public domain —
# either pre-1928 watercolor paintings or NASA works (US-Government PD).
# See tests/fixtures/style_catalog/ATTRIBUTION.md for full attribution.

set -euo pipefail

UA="plakat-spike/0.1 (https://github.com/vulogov/plakat)"
WIDTH=1024
FIXTURES="$(cd "$(dirname "$0")/.." && pwd)/tests/fixtures/style_catalog"

fetch() {
    local out_path="$1"
    local wm_file="$2"
    local full="$FIXTURES/$out_path"
    if [[ -s "$full" ]]; then
        echo "  skip   $out_path"
        return
    fi
    local url="https://commons.wikimedia.org/wiki/Special:FilePath/${wm_file}?width=${WIDTH}"
    echo "  fetch  $out_path"
    curl -sL -H "User-Agent: $UA" -o "$full" "$url"
    # Be polite to Wikimedia's rate limiter.
    sleep 1
}

echo "==> watercolor exemplars"
fetch "watercolor/01_durer_hare.jpg"           "Albrecht_D%C3%BCrer_-_Hare,_1502_-_Google_Art_Project.jpg"
fetch "watercolor/02_sargent_alligators.jpg"   "Sargent_-_Muddy_Alligators.jpg"
fetch "watercolor/03_sargent_fountain.jpg"     "John_Singer_Sargent_-_Spanish_Fountain.jpg"
fetch "watercolor/04_homer_northwoods.jpg"     "Winslow_Homer_-_The_North_Woods_(1884).jpg"

echo "==> photorealistic exemplars"
fetch "photorealistic/01_apollo11_step.jpg"    "Apollo_11_first_step.jpg"
fetch "photorealistic/02_apollo8_earthrise.jpg" "NASA-Apollo8-Dec24-Earthrise.jpg"
fetch "photorealistic/03_pillars_creation.jpg" "Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg"
fetch "photorealistic/04_buzz_flag.jpg"        "Buzz_salutes_the_U.S._Flag.jpg"

echo "==> holdout (smoke-test queries)"
fetch "holdout/watercolor_sargent_willows.jpg" "John_Singer_Sargent_-_Under_the_Willows_-_Google_Art_Project.jpg"
fetch "holdout/photo_apollo17_earth.jpg"       "The_Earth_seen_from_Apollo_17.jpg"

echo
echo "Done. Fixture sizes:"
du -h "$FIXTURES"/*/*.jpg | sort -k2
