# SimTok 
---


## Table of Contents 📑

---

1. What is SimTok?
2. System overview
3. System requirements
4. How to use it?


## What is SimTok?

SimTok is a tool for generating thermal data using `Issacsim` based on `Isaac Core 2023`.
## System overview
### Pipeline

#### 1. Simulating and collecting data
The tool captures grayscale videos of a wall usd that are suppposed to mimic real thermal camera videos and preserve metadata.
The videos are captured from two `POVS`:
  1. A **fixed** camera `POV` directed right at the target (a wall).
  2. A camera `POV` that varies in angles and distance from target.

The metadata is beings saved as `pkl` files, preserving `Pose` and `Bbox`.
`Pose` being camera orientation and global location.
`Bbox` being distance from target in each axis, whether target is in frame, whether its visible (target may be in frame but hidden behind another object).

#### 2. Adding noise to videos

The captured videos are being processed and new copies of them are being created with pre-configured noise.
Noise models may vary and work in their own sort of `pipeline`.
Noise models may include:
`DeadPixelsNoise`, `HotPixelsNoise`, `GaussianNoise` etc...

#### 3. Saving MetaData as json

The `pkl` files are being processed and the `metadata` is being saved as `json` files. The `json` files contain only the required metadata and exclude irrelevant information such as `global camera location`.

### Scripts and nodes

As well as files included in `Isaac Core 2023`, `SimTok` adds multiple scripts that some of them contain several nodes.

- simtok_cli.py

Creates a user interface cli tool to activate and pass parameters to each of the steps in the `pipeline` mentioned in `Pipeline` chapter.
Essentially wraps all steps in a configurable `pipeline`.

- collect.py

Handles running the simulation and collecting data.

Provides an `API` `DataCollector` `ROS2` node.
the node uses `HostIsaacManager` to run isaac sim with a pre-configured `usda` file.
uses, `VideoCapture`, `PoseCapture` and `BboxCapture` for filming and collecting metadata.

###### Note:  Oscillations - a calculated change of shade in the usd prim.

Requests new oscillations whenever a new `sample` begins.
A `Sample` is the combination of two `Povs` sharing the same oscillation.

Resets oscillation between each `POV` to keep the same exact data just from a different angle.

- gray_scale.py

A `script node` running from an action graph in `Isaacsim` that creates oscillations, and restarts them according to the `isaac_core/bbox topic` from which the `euclidean distance` is calculated, from the distance the gray value is being calculated in the following formula: 

- noise_processor.py

Provides an `API` for a video processing `pipeline` that adds pre-configured `thermal noise`.

- noise_models.py

Provides `noise models` classes that inherit from an abstract `NoiseModel` class.
Each class adds a different type of `thermal noise`.

- json_creation.py

Provides an `API` for a json convertor of pkl `metadata` to `json`, extracts only necessary data.

- simtok_config.toml

