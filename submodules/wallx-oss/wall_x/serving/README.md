# Wall-X Model Serving

This directory contains scripts for serving Wall-X models via a websocket server, allowing remote clients to connect and get action predictions from observations.

## Overview

The serving infrastructure consists of three main components:

1. **WebsocketPolicyServer** (`wall_x/serving/websocket_policy_server.py`): Generic websocket server that can serve any policy implementing the `BasePolicy` interface
2. **WallXPolicy** (`wall_x/serving/policy/wall_x_policy.py`): Policy wrapper that adapts the Wall-X model to the `BasePolicy` interface
3. **launch_serving.py**: Main script for starting the server with various configurations

## Quick Start

### Basic Usage

Serve a model with default LIBERO configuration:

```bash
cd /x2robot_v2/vincent/workspace/opensource
python -m wall_x.serving.launch_serving \
  --env libero \
  --model-config.model-path /path/to/libero_model_stuff \
  --model-config.action-tokenizer-path /path/to/fast/ \
  --model-config.train-config-path /path/to/config.yml
```

### Specify Environment

Serve with a specific environment preset:

```bash
# LIBERO (single arm, 7 DOF)
python -m wall_x.serving.launch_serving --env libero

# ALOHA (dual arm, 14 DOF)
python -m wall_x.serving.launch_serving --env aloha
```

### Custom Configuration

Serve with custom model paths and settings:

```bash
python -m wall_x.serving.launch_serving \
  --model-config.model-path /path/to/model \
  --model-config.action-tokenizer-path /path/to/tokenizer \
  --model-config.train-config-path /path/to/train_config.yml \
  --model-config.action-dim 7 \
  --model-config.state-dim 8 \
  --model-config.pred-horizon 32 \
  --model-config.camera-key front_view left_wrist_view \
  --port 8000
```

## Command Line Arguments

### Basic Arguments

- `--env {libero,aloha}`: Environment mode (default: libero)
- `--port PORT`: Port to serve on (default: 8000)
- `--host HOST`: Host to bind to (default: 0.0.0.0)
- `--default-prompt TEXT`: Default text prompt if not provided in observation
- `--debug`: Enable debug logging

### Model Configuration

All model configuration arguments use the `--model-config.` prefix:

- `--model-config.model-path PATH`: Path to pretrained model checkpoint (required)
- `--model-config.action-tokenizer-path PATH`: Path to action tokenizer (required)
- `--model-config.train-config-path PATH`: Path to train config YAML file (required)
- `--model-config.action-dim INT`: Action space dimension (default: 7)
- `--model-config.state-dim INT`: Robot state dimension (default: 8)
- `--model-config.pred-horizon INT`: Prediction horizon (default: 32)
- `--model-config.device {cuda,cpu}`: Device to run on (default: cuda)
- `--model-config.dtype {bfloat16,float16,float32}`: Model dtype (default: bfloat16)
- `--model-config.predict-mode {fast,diffusion}`: Prediction mode (default: fast)
- `--model-config.camera-key KEY1 KEY2 ...`: Camera keys for observation images

### Camera Keys

The `camera-key` parameter specifies which camera views are expected in the observation dictionary. This is **critical** for proper operation:

- Keys must match between server configuration and client observations
- Order matters: keys are processed in the order specified
- Common keys: `front_view`, `left_wrist_view`, `right_wrist_view`, `face_view`

Example:
```bash
--model-config.camera-key front_view left_wrist_view
```

Client must send observations with matching keys:
```python
obs = {
    "front_view": image1,        # Must match camera-key[0]
    "left_wrist_view": image2,   # Must match camera-key[1]
    "prompt": "task description",
    "state": robot_state,
}
```

## Default Configurations

### LIBERO (Single Arm)

```python
ModelConfig(
    model_path="/path/to/model",
    action_tokenizer_path="/path/to/action_tokenizer",
    train_config_path="/path/to/train_config",
    state_dim=8,
    action_dim=7,
    pred_horizon=32,
    device="cuda",
    dtype="bfloat16",
    predict_mode="fast",
    camera_key=["front_view", "left_wrist_view"],
)
```

### ALOHA (Dual Arm)

```python
ModelConfig(
    model_path="/path/to/model",
    action_tokenizer_path="/path/to/action_tokenizer",
    train_config_path="/path/to/train_config",
    state_dim=14,
    action_dim=14,
    pred_horizon=32,
    device="cuda",
    dtype="bfloat16",
    predict_mode="fast",
    camera_key=["face_view", "left_wrist_view", "right_wrist_view"],
)
```

## Server Protocol

### Connection Flow

1. Client connects to `ws://host:port`
2. Server sends metadata JSON with policy information
3. Client sends observation (msgpack-encoded)
4. Server responds with action prediction (msgpack-encoded)
5. Repeat steps 3-4 for each inference

### Observation Format

Observations must be a dictionary with camera keys matching server configuration:

```python
obs = {
    # Image observations - keys must match server's camera_key configuration
    "front_view": np.ndarray,      # (H, W, 3) uint8 or float
    "left_wrist_view": np.ndarray, # (H, W, 3) uint8 or float

    # Required fields
    "prompt": str,                 # Task description
    "dataset_names": List[str],    # Dataset/robot name, e.g., ["physical-intelligence/libero"]
    "state": np.ndarray,           # Robot proprioception state (state_dim,)
}
```

**Important**: The image keys (`front_view`, `left_wrist_view`, etc.) must exactly match the `camera_key` parameter configured on the server.

### Action Response Format

Actions are returned as a dictionary:

```python
{
    "action": np.ndarray,       # Predicted action [pred_horizon, action_dim]
    "server_timing": {
        "infer_ms": float,      # Inference time in milliseconds
        "prev_total_ms": float, # Total time for previous request
    }
}
```

### Server Metadata

When connecting, the server sends metadata:

```python
{
    "action_dim": int,          # Action space dimension
    "pred_horizon": int,        # Number of future actions predicted
    "device": str,              # Device model runs on
    "predict_mode": str,        # Prediction mode (fast/diffusion)
    "env": str,                 # Environment name
}
```

### Health Check

HTTP health check endpoint available at:
```
http://host:port/healthz
```

Returns `200 OK` if the server is running.

## Client Example

### Synchronous Python Client

For synchronous usage, see `wall_x/serving/client.py`:

```python
from wall_x.serving.client import WallXClient

# Create and connect
client = WallXClient(uri="ws://localhost:8000")
client.connect_sync()

# Prepare observation
obs = {
    "front_view": image1,
    "left_wrist_view": image2,
    "prompt": "task description",
    "state": robot_state,
    "dataset_names": ["physical-intelligence/libero"],
}

# Get prediction
response = client.predict_sync(obs)
action = response["action"]

# Close connection
client.close_sync()
```

## Architecture

### WebsocketPolicyServer

Generic websocket server that:
- Handles websocket connections with msgpack serialization
- Tracks inference timing and performance metrics
- Provides health check endpoint
- Handles errors gracefully with proper logging
- Supports concurrent client connections

### WallXPolicy

Policy wrapper that:
- Loads and manages the Wall-X model from pretrained checkpoint
- Processes multi-camera observations
- Handles image preprocessing (smart resize, normalization)
- Manages device placement and dtype conversion
- Provides policy metadata to clients
- Supports both fast tokenizer and diffusion prediction modes

### Image Processing Pipeline

1. **Camera Key Matching**: Extracts images from observation dict using configured camera keys
2. **Format Conversion**: Converts numpy arrays to PIL Images
3. **Smart Resize**: Applies Qwen's smart resize algorithm based on min/max pixels
4. **Vision Token Formatting**: Inserts vision tokens in text prompt
5. **Batch Preparation**: Creates model-ready BatchFeature input
