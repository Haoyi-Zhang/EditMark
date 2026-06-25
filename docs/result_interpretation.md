# Result Interpretation

The result surface separates detector evidence from provenance evidence. A clean detector score says that a watermark signal is visible before software change. A post-edit provenance claim additionally requires retained evidence after test-passing edits, support, negative controls, utility, and cost.

The tables should therefore be read component-wise. A method can be useful under one edit family and weak under another. A method can retain evidence but still carry a cost or false-positive boundary. The artifact is meant to prevent those distinctions from being collapsed into a single ranking.

The compact score is a secondary navigation index, not the paper's claim. The released score multiplies the gate by a geometric combination of headline core evidence and headline generalization after the scorecard's soft-floor component scaling. The core evidence itself combines detection, robustness, utility, control behavior, and efficiency. Use it as a navigation cue, then inspect the component tables before interpreting any provenance claim.
