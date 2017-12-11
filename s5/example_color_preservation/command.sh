#!/usr/bin/env bash
python3 INetwork.py images/inputs/content/Kinkaku-ji.jpg \
                    images/inputs/style/water-lilies-1919-2.jpg "NoColorPres"

python3 INetwork.py images/inputs/content/Kinkaku-ji.jpg \
                    images/inputs/style/water-lilies-1919-2.jpg "ColorPres" \
                    --preserve_color True
