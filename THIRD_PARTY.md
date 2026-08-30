# Third-party source

`src/lerobot/` contains the LeRobot runtime snapshot used by this project so the
custom DinoFlow policy and its training processor can run independently of the
parent `openpi_repo` checkout. The upstream LeRobot portions retain their
original copyright notices and Apache-2.0 license.

The files under `src/lerobot/policies/dino_flow/` and the DinoFlow-specific
training/deployment scripts are the project-specific implementation.
