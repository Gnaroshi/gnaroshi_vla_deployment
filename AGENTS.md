# Deployment skeleton contract

Read `README.md`, `contracts.py`, the selected embodiment adapter and the owning
project's deployment instructions before changing a deployment.

- The shared default branch contains the reusable GUI, session, recording,
  adapter contracts and CPU mock. Keep concrete model/paper/task implementations,
  checkpoint locations, personal paths and hardware identities out of this branch.
- Extend a clone inside the owning project under `<backbone>/<paper>/`.
  Keep its policy factory, shell launcher, process lifecycle and task configuration
  together. Project-specific changes belong to that project, not the shared template.
- Generic operator state, GUI, recording and metrics stay in `src/gnaroshi_deployment`.
  Hardware adapters must not import model implementations. Implement a new adapter
  through `contracts.py`; the mock is the runnable example. Do not duplicate the GUI.
- Trace image roles/RGB/resize, state/action dimensions, coordinates, units,
  normalization, temporal horizon, executed chunk length, reset and checkpoint
  identity from the actual selected policy. Do not copy another policy's shapes.
- Let each project supply robot address, camera identities and task home. Keep
  task/method/checkpoint/target Hz/max steps/cameras/output visible in its shell.
  Record the model and method clearly. Never silently change the requested rate
  to the data recording rate, or report requested rate as achieved command rate.
- Start homes then resets a fresh episode. Success/Failure stop commands, save
  the manual outcome and return home. Stop/Retry discard the attempt and remain
  stationary. Max steps means waiting for an operator outcome, never automatic success.
- An active attempt interrupted by a robot, policy or device fault is saved as an
  automatic failure with its cause and prior observations/actions/media. Keep
  automatic failures distinct from manual outcomes and preserve accepted outcomes
  when only the return home fails. Idle faults must not invent a rollout.
- Stop/close cancels in-flight predictions before they can issue a command.
  Raise the hardware stop latch before acquiring a command lock and check it
  inside apply. Failed or unconfirmed stops block further motion. Cleanup must
  run even when stop itself raises. Use finite network/camera deadlines.
- Importing, config inspection, CPU tests and mock GUI must not connect to a
  model server, robot, gripper or camera. Real hardware requires the project's
  explicit live action. Start authorizes episode preparation. A separate idle Home
  authorizes arm homing after readiness checks; use home_idle when available to
  preserve the gripper. Both operations remain cancellable by Stop/Exit.
- Preserve the established operator flow. Do not add repeated TCP-bound or
  emergency-stop confirmation prompts. Deletion of saved results/media requires
  GUI confirmation. Communication errors during an active attempt are automatic
  failures with a recorded cause, rather than operator-classified outcomes.
- Let the caller choose output storage with `DEPLOYMENT_OUTPUT_ROOT` or explicit
  config. Default generated files stay under ignored `.runtime`. Never commit
  checkpoints, caches, videos, logs, credentials or research outputs.
- Respect the user's current GPU allocation in each process. Visibility does not
  reserve a device. Preserve other researchers' jobs, data, ports and containers.
- Verify session/adapter changes with CPU state/recording tests and mock GUI smoke.
  Mock/preflight success does not establish physical task success. Report the
  tested scope precisely and keep physical validation separate.
