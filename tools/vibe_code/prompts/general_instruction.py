text_prompt = """
You are an expert software developer and system architect.
Your task is to help me construct a modular world model framework by transforming and completing code according to the specifications below.
You will be provided with a code repository and required to adapt or generate code that conforms strictly to the following architecture and file formats.

====================================================
1. Test File
====================================================

You need to provide a test file that calls a pipeline file to validate the system.
The test code should be as concise as possible. Please follow the reference example below:

```python
from diffusers.utils import export_to_video
from PIL import Image
from sceneflow.pipelines.matrix_game.pipeline_matrix_game_2 import MatrixGame2Pipeline

image_path = "./data/test_case1/ref_image.png"
input_image = Image.open(image_path).convert("RGB")

pretrained_model_path = "Skywork/Matrix-Game-2.0"
pipeline = MatrixGame2Pipeline.from_pretrained(
    synthesis_model_path=pretrained_model_path,
    mode="universal",
    device="cuda"
)
output_video = pipeline(
    input_image=input_image,
    num_output_frames=150,
    interaction_signal=[
        "forward", "left", "right",
        "forward_left", "forward_right",
        "camera_l", "camera_r"
    ]
)
export_to_video(output_video, "matrix_game_2_demo.mp4", fps=12)
```

====================================================
2. Pipeline File
====================================================

The pipeline file is the core interface invoked by the test file.
It should follow the structure below:
```python
class PipelineABC:
    def __init__(self):
        pass

    @classmethod
    def from_pretrained(cls):
        ###### Load different categories of pretrained models here ######
        return cls()
    
    def process(self, *args, **kwds):
        ###### Process interaction signals using operators here ######
        pass
    
    def __call__(self, *args, **kwds):
        ###### This is the main interface called by the test file.
        ###### It should internally invoke the process() function.
        pass

    def stream(self, *args, **kwds) -> Generator[torch.Tensor, List[str], None]:
        ###### This function supports multi-round interactive inputs.
        ###### It should call __call__ internally.
        ###### Memory management must be handled here via the Memory module.
        pass
```

====================================================
3. Operator File
====================================================

The operator is responsible for handling interactions and managing input signals.
Operator file format:

```python
class BaseOperator(object):
    def __init__(self, operation_types=[]):
        #####
        operation types include:
            - textual_instruction
            - visual_instruction
            - action_instruction (e.g., mouse and keyboard input
              for trajectory and viewpoint control)
        #####
        self.interaction_template = []
        self.current_interaction = []
        self.interaction_history = []

    def interaction_template_init(self):
        if type(self.interaction_template) is not list:
            raise ValueError("interaction_template should be a list")

    def get_interaction(self, interaction):
        ##### Use this function to update the interaction list
        ##### This function should call check_interaction internally
        pass

    def check_interaction(self, interaction):
        ##### Use this function to validate the interaction
        ##### Called inside get_interaction
        pass

    def process_interaction(self):
        ##### Use this function to process interaction signals
        pass

    def process_perception(self):
        ##### Use this function to process visual and audio signals
        ##### This is different from process_interaction and is designed
        ##### for real-time perceptual updates
        pass
```

====================================================
4. Memory File
====================================================

The memory module is primarily used inside the pipeline's stream() function
for interaction history and state management. Format:

```python
class BaseMemory(object):
    ###### Generic Multimodal Memory System Template
    ###### Designed for VLM, VLA, and generative/reasoning tasks
    ###### NOTE:
    ###### - record() and select() are the primary interfaces used by the pipeline
    ###### - compress() and process() are internal functions called by select()
    ###### - manage() handles lifecycle and memory consolidation

    def __init__(self, capacity=None, **kwargs):
        #### Initialize storage structures and resource constraints
        self.storage = []
        self.capacity = capacity

    def check_template(self, **kwargs):
        #### The template of self.storage should be:
        #### [
        ####   {
        ####     'content': ...,
        ####     'type': <type>,
        ####     'timestamp': <timestamp>,
        ####     'metadata': <metadata>
        ####   },
        ####   ...
        #### ]
        #### Allowed types:
        #### ['image', 'video', 'text', 'audio', 'action', 'other']
        pass

    def record(self, data, metadata=None, **kwargs):
        #### 1. Recording (Ingestion)
        #### Purpose:
        ####   Ingest raw interaction data (image, action, text, depth, etc.)
        #### Logic:
        ####   Assign metadata and insert into memory storage
        pass

    def select(self, context_query, **kwargs):
        #### 2. Selection (Retrieval)
        #### Purpose:
        ####   Retrieve relevant memory entries based on task context
        #### Logic:
        ####   Similarity matching, temporal correlation,
        ####   or importance-based filtering
        pass

    def compress(self, memory_items, **kwargs):
        #### 3. Compression (Refinement)
        #### Purpose:
        ####   Reduce memory size or distill key information
        #### Logic:
        ####   Text summarization, feature extraction, etc.
        pass

    def process(self, refined_data, target_format="kv_cache", **kwargs):
        #### 4. Processing (Adaptation)
        #### Purpose:
        ####   Convert refined memory into model-ready representations
        #### Logic:
        ####   e.g., KV cache, latent tokens, embeddings
        pass

    def manage(self, **kwargs):
        #### 5. Management (Lifecycle & Consolidation)
        #### Purpose:
        ####   Maintain long-term memory health
        #### Logic:
        ####   Memory merging, eviction (forgetting),
        ####   STM to LTM transfer
```

====================================================
5. Other Files
====================================================

Additional files are required to:
- Receive outputs from operators inside the pipeline
- Perform task-specific generation, reasoning, and representation learning
- Wrap various generative and inference models used by the world model
For this code construction task, you are required to provide the following "other files":
"""
