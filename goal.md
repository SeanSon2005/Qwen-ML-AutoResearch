# Experiment Goal

## Objective
In the BCI Competition IV 2a EEG motor-imagery dataset, motor signals are often decoded using channels such as C3 and C4, which are strongly associated with hand movement and motor imagery. We want to replicate a similar motor-imagery classification task, but with a focus on more distributed or planning-related neural activity. To do this, we will omit the C3 and C4 channels and train a model using the remaining EEG channels. The goal is to evaluate whether comparable classification performance can be achieved without relying on the primary sensorimotor channels most directly associated with hand movement.

Please help design an experimental pipeline for this task, including preprocessing, channel selection, feature extraction, model training, evaluation metrics, and a comparison against a baseline model that includes C3 and C4.

This dataset comprises 9 MATLAB (.mat) files from the BCI Competition IV – Data Set 2a. Each file contains EEG recordings from a different subject, recorded at the Graz University of Technology. The recordings include 22 EEG channels and 3 EOG channels sampled at 250 Hz while subjects performed four distinct motor imagery tasks:
* Left hand
* Right hand
* Both feet
* Tongue

With sessions recorded on different days, the dataset is ideal for studying continuous classification challenges and session-to-session transfer in brain–computer interface research. This resource is perfect for developing and benchmarking machine learning and signal processing algorithms for EEG-based BCIs.

## Optimization Metric
- Determine a metric to optimize.

## Constraints
- memory_gb: <= 23.0

## Notes
None
