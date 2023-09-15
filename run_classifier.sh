#!/bin/bash
#
# Run classifier on Woebot Message Data.

echo "Pre-processing"
echo ""
python3 "./src/pre-processing.py"

echo "Generating visualizations for data exploration (view in /figures/distributions folder)"
echo ""
python3 "./src/visualization/data_distribution.py"

echo "Running polarity classifiers"
python3 "./src/models/polarity_classification.py"

echo ""
echo "Generating visualizations for results from polarity classifier(view in /figures/results folder)"

echo ""
echo "Running empathy classifier"
echo "Results:"
python3 -W "ignore" "./src/models/empathy_classification.py"


