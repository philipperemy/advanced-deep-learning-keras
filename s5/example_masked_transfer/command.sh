#!/usr/bin/env bash
python3 INetwork.py images/inputs/content/Japanese-cherry-widescreen-wallpaper-Picture-1366x768.jpg \
                    images/inputs/style/metals/burnt_gold.jpg \
                    "Cherry_Blossom_inverted" \
                    --style_masks images/inputs/mask/cherry-blossom-1.jpg --preserve_color True

python3 INetwork.py images/inputs/content/Japanese-cherry-widescreen-wallpaper-Picture-1366x768.jpg \
                    images/inputs/style/metals/burnt_gold.jpg \
                    "Cherry_Blossom" \
                    --style_masks images/inputs/mask/cherry-blossom-2.jpg \
                    --preserve_color True
