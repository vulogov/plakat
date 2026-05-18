#!/usr/bin/env bash
# Fetch public-domain exemplar images from Wikimedia Commons for the style
# catalog. Idempotent — re-running skips files already on disk.
#
# Output: tests/fixtures/style_catalog/<style>/ + holdout/
#
# Each style has 4 exemplar images; holdouts feed the smoke test (one per
# style currently). All images are public domain — either pre-1928
# paintings/prints whose authors died long enough ago, or NASA works
# (US-Government PD). See tests/fixtures/style_catalog/ATTRIBUTION.md
# for full attribution.

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

echo "==> oil_painting exemplars"
fetch "oil_painting/01_rembrandt_self_portrait.jpg" \
    "Rembrandt_van_Rijn_-_Self-Portrait_-_Google_Art_Project.jpg"
fetch "oil_painting/02_vermeer_pearl_earring.jpg" \
    "Johannes_Vermeer_(1632-1675)_-_The_Girl_With_The_Pearl_Earring_(1665).jpg"
fetch "oil_painting/03_sargent_lady_agnew.jpg" \
    "John_Singer_Sargent_-_Lady_Agnew_of_Lochnaw.jpg"
fetch "oil_painting/04_waterhouse_lady_shalott.jpg" \
    "John_William_Waterhouse_-_The_Lady_of_Shalott_-_Google_Art_Project.jpg"

echo "==> ukiyo_e exemplars"
fetch "ukiyo_e/01_hokusai_great_wave.jpg" \
    "Tsunami_by_hokusai_19th_century.jpg"
fetch "ukiyo_e/02_hiroshige_arashiyama.jpg" \
    "Arashiyama_by_Hiroshige_(Yamagata).jpg"
fetch "ukiyo_e/03_hiroshige_shellfish.jpg" \
    "Gathering_Shellfish_at_Shinagawa_Gotenyama_by_Hiroshige.jpg"
fetch "ukiyo_e/04_hokusai_saigyo.jpg" \
    "Katsushika_Hokusai_Poet_Saigyo.jpg"

echo "==> art_nouveau exemplars (Alphonse Mucha posters)"
fetch "art_nouveau/01_mucha_dame_camelias.jpg" \
    "Alfons_Mucha_-_1896_-_La_Dame_aux_Cam%C3%A9lias_-_Sarah_Bernhardt.jpg"
fetch "art_nouveau/02_mucha_biscuits.jpg" \
    "Alfons_Mucha_-_1896_-_Biscuits_Lef%C3%A8vre-Utile.jpg"
fetch "art_nouveau/03_mucha_salammbo.jpg" \
    "Alfons_Mucha_-_1896_-_Salammb%C3%B4.jpg"
fetch "art_nouveau/04_mucha_imprimerie.jpg" \
    "Alfons_Mucha_-_Poster_for_'Imprimerie_Cassan_Fils'_(1896).jpg"

echo "==> holdout (smoke-test queries)"
fetch "holdout/watercolor_sargent_willows.jpg" \
    "John_Singer_Sargent_-_Under_the_Willows_-_Google_Art_Project.jpg"
fetch "holdout/photo_apollo17_earth.jpg" \
    "The_Earth_seen_from_Apollo_17.jpg"
fetch "holdout/oil_painting_rembrandt63.jpg" \
    "Rembrandt,_Self_Portrait_at_the_Age_of_63.jpg"
fetch "holdout/ukiyo_e_great_wave_bm.jpg" \
    "Great_Wave_Hokusai_BM_1906.1220.0.533_n02.jpg"
fetch "holdout/art_nouveau_mucha_salon_cent.jpg" \
    "Alfons_Mucha_-_Salon_des_Cent_20th_Exhibition,_1896.jpg"

echo
echo "Done. Fixture sizes:"
du -h "$FIXTURES"/*/*.jpg | sort -k2
