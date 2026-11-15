| How the planner learns what other agents measured | Controls set to the measured-best value | Alive write switched on (measured harmful) | Seconds per decision | Prompt size per decision | Input cost per 1,000 decisions |
|---|---|---|---|---|---|
| No shared memory (own menu of beliefs) | 37 of 48 (77%) | 11 of 12 cases | 1.5 | about 784 tokens | $0.78 |
| Whole shared log pasted into the prompt (68 notes) | 48 of 48 (100%) | 0 of 12 cases | 2.1 | about 6,480 tokens | $6.48 |
| past.dev recall (author, kind, time, supersededAt) | 44 of 48 (92%) | 0 of 12 cases | 3.2 | about 1,349 tokens | $1.35 |
