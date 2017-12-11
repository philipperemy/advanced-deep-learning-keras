#!/usr/bin/env bash
python3 INetwork.py images/inputs/content/Dipping-Sun.jpg \
                    images/inputs/style/misty-mood-leonid-afremov.jpg \
                    "dominant_style" \
                    --content_weight 0.0001

python3 INetwork.py images/inputs/content/Dipping-Sun.jpg \
                    images/inputs/style/misty-mood-leonid-afremov.jpg \
                    "dominant_content" \
                    --content_weight 10

