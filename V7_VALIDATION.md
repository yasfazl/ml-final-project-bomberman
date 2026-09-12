# V7 validation record

Validation completed in the packaging workspace:

- every Python source and test file compiled successfully;
- the actual `potential_shaping.py` implementation passed direct checks for
  endgame gating, bounded values, attack-readiness ordering, terminal zero,
  a `+4` ready-to-post-bomb transition, discounted telescoping, and a
  non-profitable closed loop;
- the actual `train.py` reward integration passed direct checks that the old
  immediate targeted-bomb bonus is replaced only in safe-attack mode;
- the actual checkpoint verifier accepted an adapter-only change and rejected
  changes to protected input columns and later layers;
- the archive contains no `.pt`, `.pkl`, generated statistics, or logs.

The packaging workspace does not contain PyTorch or pytest, so the focused
behavioral suite could not be executed there.  Run the exact pytest command in
`V7_INSTALL.md` in the existing Mac virtual environment before training.  Do
not train if any test fails.
