#!/bin/bash
source venv/bin/activate
python ai/train_manual.py --level 1 --trains 2 --steps 500000 --no-early-stop
python ai/train_manual.py --level 2 --trains 5 --steps 700000 --load ai/models/L1_2Trains_Best/best_model --no-early-stop
python ai/train_manual.py --level 3 --trains 7 --steps 700000 --load ai/models/L2_5Trains_Best/best_model --no-early-stop
python ai/train_manual.py --level 4 --trains 10 --steps 1000000 --load ai/models/L3_7Trains_Best/best_model --no-early-stop
