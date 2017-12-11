#!/usr/bin/env bash
python3 INetwork.py images/inputs/content/blue-moon-lake.jpg \
                    images/inputs/style/starry_night.jpg \
                    images/inputs/style/red-canna.jpg \
                    "Starry_night_dominant" \
                    --style_weight 1.0 0.2 \
                    --pool_type "ave" \
                    --num_iter 50

python3 INetwork.py images/inputs/content/blue-moon-lake.jpg \
                    images/inputs/style/starry_night.jpg \
                    images/inputs/style/red-canna.jpg \
                    "Canna_dominant" \
                    --style_weight 0.2 1.0 \
                    --pool_type "ave" \
                    --num_iter 50
