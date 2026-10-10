# D6 Memo: which drift hurts most, and what would I change?

**Answer: the telegraph frequency jump hurts the algorithm most, not the gain drift.** In our simulation (21 seeds; controls set perfectly at t=0, then frozen; p=1 QAOA on the 4-ring), the 0.8 MHz jump alone lowers the cost by 0.176% of the ideal on average (SD 0.065%). All gain drift gives 0.061% and the OU frequency wander 0.027%. By raw gate error the order flips: gain (mean error 5.9e-4) beats the jump (4.3e-4). Gate-error rankings should therefore not be used alone to prioritise algorithm-level fixes. We did not isolate why; a plausible cause is that detuning error acts on every state-preparation and mixer rotation.

**What I would change in the control stack**

1. *Calibrate along a dependency graph* (Kelly et al., 2018). We ran frequency, then amplitude, then DRAG. The DRAG optimum (-0.313 ns) was flat enough (a 0.05 ns error costs about 5e-5 in error) to calibrate once.
2. *Do not over-engineer triggering.* A recalibration costs about 520 shots (0.26 s), so simply recalibrating every 30-60 min reached mean error 1.7-2.2e-4, near the 0.94e-4 floor. Our health-check policy did not beat this when its shots count as downtime (paired difference +6.7e-5, 95% CI [3.2e-5, 1.0e-4], 21 seeds).
3. *Fix the pulse, not just the parameters.* First-order DRAG (beta = 1/|alpha|; Motzoi et al., 2009) overestimated the optimum by about 70% (0.53 vs 0.313 ns) because the dominant error was a Stark-type coherent tilt, not leakage. Leakage sits at a floor of 1.7e-4 set by truncating the Gaussian. A smooth envelope with Stark compensation (virtual-Z or frequency offset) should lower both.

**What I would change in hardware**

- Reduce telegraph jumps at the source: fewer TLS defects (materials and interfaces) and frequency-allocation choices that avoid defects.
- Temperature-stabilise the drive electronics to flatten the +/-2% gain swing; its slow, predictable sinusoid is cheap to track.

**Caveats.** These are simulation results under the specified drift model. Follow-up tests should vary drift amplitudes, measurement noise, and circuit depth before generalising these conclusions beyond this four-qubit benchmark. The 4-ring p=1 QAOA is forgiving: with no recalibration the average shortfall is only 0.23%. Idle-detuning phase between gates and decoherence inside the gate error are not modelled. The hardware suggestions are hypotheses; this study did not test them.

**References.** Kelly et al., arXiv:1803.03226 (2018). Motzoi et al., Phys. Rev. Lett. 103, 110501 (2009).
