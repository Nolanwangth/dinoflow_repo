# Deployment

`server.py` only needs the `dinoflow_env` and a trained checkpoint:

```bash
cd /home/nolan/vla/dinoflow_repo
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python deployment/server.py \
  --model-path /path/to/checkpoints/030000/pretrained_model
```

`client_mock.py` and `client.py` are the AgiBot WBC adapters. They are optional
and are not needed for training. To use them from this independent repository,
point `DINOFLOW_ROBOT_ROOT` at the checkout that provides `agibot/`,
`agibot_gdk/`, and `wbc_gdk.WbcGdk`:

```bash
export DINOFLOW_ROBOT_ROOT=/path/to/robot/checkout
python deployment/client_mock.py --host 127.0.0.1 --port 9001
```

Run the mock first. It reads robot state/cameras but does not send motor
commands. Only use `client.py` after the robot-side safety checks are complete.

The default deployment schedule predicts 50 actions, refreshes the chunk after
20 control ticks, and applies RTC over the next 20 overlapping ticks. The
initial inference delay is configured as 3 ticks; adjust
`--inference-delay-steps` after checking the server's measured latency. The
robot client enables strict sensor validation automatically. The mock can use
the same checks with `--strict-sensors`.
