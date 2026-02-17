### AI2-THOR Unity Environment Setup

To run the AI2-THOR simulation, the Unity executable is required.

Please download the precompiled Unity build from:
[Download link](http://s3-us-west-2.amazonaws.com/ai2-thor-public/builds/thor-Linux64-f0825767cd50d69f666c7f282e54abfe58f1e917.zip)

Alternatively, you can use the provided installation script:

```bash
bash tools/installing/thor/download_ai2thor.sh
```

By default, the Unity build will be extracted to submodules/thor/.
To install it in a different location, modify the TARGET_DIR variable in the script.