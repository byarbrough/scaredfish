#!/bin/bash
echo test
# Run threshold experiments for both relaxed and alarmed states
# This script runs the threshold.py experiment twice with different Couzin parameters

set -e  # Exit on error

echo "========================================================================"
echo "RUNNING THRESHOLD EXPERIMENTS - BOTH STATES"
echo "========================================================================"
echo ""
echo "This will run two experiments:"
echo "  1. Relaxed state (repulsion=5, orientation=12, attraction=36)"
echo "  2. Alarmed state (repulsion=2.75, orientation=12, attraction=48)"
echo ""
echo "========================================================================"
echo ""

# Run relaxed state
echo "========================================================================"
echo "EXPERIMENT 1/2: RELAXED STATE"
echo "========================================================================"
echo ""
time python threshold.py --state relaxed

echo ""
echo ""

# Run alarmed state
echo "========================================================================"
echo "EXPERIMENT 2/2: ALARMED STATE"
echo "========================================================================"
echo ""
time python threshold.py --state alarmed

echo ""
echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "========================================================================"
echo ""
echo "Output files:"
echo "  - threshold_relaxed_detailed_results.csv"
echo "  - threshold_relaxed_summary.csv"
echo "  - threshold_alarmed_detailed_results.csv"
echo "  - threshold_alarmed_summary.csv"
echo ""
echo "Next step: Analyze results in threshold_analysis.ipynb"
echo "========================================================================"
