#!/usr/bin/env python3
"""Analyze the failing images to find patterns"""

failing = [
    ("apple ii.png", 90.77),
    ("murky wisdom.png", 91.11),
    ("arch.png", 91.30),
    ("aurora borealis planet.png", 93.08),
    ("blasted farewell.png", 94.50),
    ("double cleanse.png", 94.68),
    ("climate groundhog.png", 95.52),
    ("glacier of debt.png", 95.59),
    ("hidden entitlements.png", 95.59),
    ("climate hell.png", 95.67),
    ("flying tiger.png", 95.78),
    ("nature's pyrotechnics.png", 96.08),
    ("garbage wave theory.png", 96.11),
    ("molten home.png", 96.38),
    ("double blockage.png", 96.59),
    ("pretentious silence.png", 96.65),
    ("grant's march.png", 96.79),
    ("democratic party's polls.png", 96.84),
    ("ca.png", 96.85),
    ("healthcare.png", 96.85),
    ("dystopian novel.png", 96.96),
]

print("Failing Images by Category:\n")

print("CRITICAL (<92%): Need urgent attention")
for name, acc in failing:
    if acc < 92:
        print(f"  {acc:.2f}% - {name}")

print("\nBAD (92-94%): Significant issues")
for name, acc in failing:
    if 92 <= acc < 94:
        print(f"  {acc:.2f}% - {name}")

print("\nMARGINAL (94-96%): Close to passing")
for name, acc in failing:
    if 94 <= acc < 96:
        print(f"  {acc:.2f}% - {name}")

print("\nVERY CLOSE (96-97%): Just under threshold")
for name, acc in failing:
    if 96 <= acc < 97:
        print(f"  {acc:.2f}% - {name}")

print(f"\nTotal failing: {len(failing)}")
print(f"Critical: {sum(1 for _, acc in failing if acc < 92)}")
print(f"Bad: {sum(1 for _, acc in failing if 92 <= acc < 94)}")
print(f"Marginal: {sum(1 for _, acc in failing if 94 <= acc < 96)}")
print(f"Very close: {sum(1 for _, acc in failing if 96 <= acc < 97)}")
