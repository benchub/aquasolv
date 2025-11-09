#!/usr/bin/env python3
import re

with open('/tmp/claude/test_results.txt', 'r') as f:
    text = f.read()

# Find all accuracy percentages
accuracies = re.findall(r'(\d+\.\d+)%', text)
accuracies = [float(x) for x in accuracies]

if accuracies:
    mean = sum(accuracies) / len(accuracies)
    print(f"Mean accuracy: {mean:.2f}%")
    print(f"Median accuracy: {sorted(accuracies)[len(accuracies)//2]:.2f}%")
    print(f"Count: {len(accuracies)}")
else:
    print("No accuracies found")
