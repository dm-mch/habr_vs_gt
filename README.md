# Habrahabr vs geektime classifier
Learning for classifier text on 2 categoris: habrahabr vs geektimes
## Features
- Download articles from habrahabr and geektimes
- Build model for classification
- Train and score
- Test new text file

## Dependence
- Python 3.5
- numpy
- sklearn
- scrapy
- matplotlib
- argparse
- pickle

## Sampe usage
```
habrgeek.py ---download 1000 --train
habrgeek.py --train -f habrahabr.json geektimes.json
habrgeek.py --test -f input.txt
habrgeek.py ---download 1000
```
