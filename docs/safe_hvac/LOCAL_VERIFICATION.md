# Preparation-machine verification

Date: 11 October 2026.

No EnergyPlus or BOPTEST process was started. No building rollout, experimental training job, GPU pilot, or performance comparison was executed. Small synthetic fixtures exercised numerical calculations, isolated optimizer updates, SB3's collector with a fake simulator-contract environment, checkpoint loading, and result-file validation.

## Completed checks

| Check | Result |
|---|---|
| Corrected-code regression suite | **20 passed** |
| Target SB3/Gymnasium fixture versions | SB3 **2.8.0**, Gymnasium **1.2.3**, loaded from isolated temporary wheels |
| Host numerical libraries used by those tests | Python **3.12.12**, PyTorch **2.5.1**, NumPy **1.26.4** |
| Fresh original HVAC source + portable patch | Applies cleanly; reverse check recognizes an already applied patch |
| Bootstrap on the existing corrected clones | Recognizes the pinned commits and applied adapters |
| Command-line help checks | All **11** modern/legacy train/eval/check entry points load without simulator initialization |
| GPU shell scripts | Shell syntax checks passed |
| Experiment configuration loading | All seven YAML presets resolve and pass budget/configuration validation |
| Extracted publication bundle | All **20 tests pass** from the extracted source, independently of the preparation checkout |

The suite covers:

- The `[0.5,1.5] * [0,2]` safety example gives `1.5`, with ratio gradients `[0,1]`; a `[B,1]` residual is rejected, and barrier gradients are detached during actor weighting.
- GAE includes the final reward, separates terminal from truncated bootstrap, and stops across resets.
- Positive margins reject the zero barrier as a solution for both trusted classes.
- One/multiple-horizon successors retain final transitions and mask reset crossings.
- Replay capacity remains bounded, including the case where one class is missing.
- Physical occupancy labels and the reward use the same bounds; energy uses W-to-kWh integration.
- Action scaling, actual successor capture before autoreset, clock advance and timeout bootstrap are checked against a synthetic Sinergym-shaped contract.
- Policy action-head and barrier parameters change in isolated update fixtures; barrier-only pretraining leaves policy parameters unchanged.
- Policy/critic, barrier, dual and optimizer state reload correctly; the dual optimizer remains bound to its restored parameter. Further fixture updates succeed after reload.
- Evaluation uses saved normalization without updating it and reports all configured seeds; checkpoint evaluation preserves the training RNG sequence.
- Comparison rejects different tasks, duplicate inputs and missing matched environment seeds; test seeds are aggregated within independently trained models.

## Limits

The full pinned GPU environment also uses newer PyTorch/NumPy packages and EnergyPlus 25.1. It has not been installed or tested on this preparation machine. These checks establish numerical and software-contract behavior, not live simulator compatibility or safety/performance. The GPU runbook therefore requires the doctor, real physical preflight and both smoke runs before a pilot.

There are no actual Sinergym learning curves or energy/comfort results to report yet. The portable bundle contains source, configs, tests, instructions and the legacy patch; it contains no trained model or synthetic performance presented as an experiment.
