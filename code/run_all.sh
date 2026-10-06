#!/bin/bash
# Order: Granville-table windows (1e7, 1e8), all even n to 1e8, the eight pre-registered fresh windows, then 1e9 and 1e10.
python3 -u goldbach_rev.py window 10000000
python3 -u goldbach_rev.py window 100000000
python3 -u goldbach_rev.py full 100000000
for A in 125894656 199524352 316227584 501186560 794329088 1258921984 1995259904 3162275840; do
  python3 -u goldbach_rev.py window $A
done
python3 -u goldbach_rev.py window 1000000000
python3 -u goldbach_rev.py window 10000000000
echo ALL DONE
