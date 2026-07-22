## OpenWorldLib Planning

### Methods Being Integrated

We are continuously expanding the **World Model** method library in our framework.  
The representative methods currently being integrated are listed below. We warmly welcome more developers to submit PRs and help us improve the ecosystem together!

| Method | Paper | Code Repository | Main Direction / Task |
|---|---|---|---|
| **Qwen-RobotWorld** | [paper](https://arxiv.org/abs/2606.17030) | [code](#) | Embodied AI (language-conditioned video world model for robot manipulation) |
| **LingbotMap** | [paper](https://arxiv.org/abs/2604.14141) | [code](https://github.com/ironmanizawesome/LingbotMap) | 3D Scene Generation (feed-forward 3D foundation model for streaming reconstruction) |
| **AlayaWorld** | [paper](https://arxiv.org/abs/2607.06291) | [code](https://github.com/AlayaLab/AlayaWorld) | Navigation Video Generation (long-horizon playable video world generation) |

> If you have methods you’d like to integrate (e.g., new world models, evaluation benchmarks, or inference pipelines), feel free to open an Issue or submit a PR directly.


### Todo List

Status Legend: ✅ Completed; 🚧 In Progress; 🔄 Ongoing; 📝 Not Started.

- ✅ Integrate multiple world model tasks into a single unified framework.
- 🔄 Continuously add cutting-edge and valuable world model-related works to OpenWorldLib.
- 🚧 Support more works related to long-video generation.
- 🚧 Support more works related to autonomous driving.
- 🚧 Optimize the contribution workflow to make it easier for other contributors to submit PRs, and continuously refactor the OpenWorldLib codebase.
- 🔄 Provide comprehensive documentation and detailed explanations for the interactions within each pipeline.
- 📝 Support inference data parallelism across all pipelines.
- 🔄 Streamline the codebase and simplify the outputs of the pipelines.


### Code Optimization
- The current repository may still contain some redundant code, which we will further streamline;
- The interactions between different modules are not yet seamless enough. We will work towards enabling more flexible and natural integration between components;
- The performance of some existing methods still requires further improvement, specifically:
  - The rendering effectiveness of vggt should improve;


### Roadmap

- **Continuous Benchmark Enhancement**  
  We will further improve the current benchmark suite by expanding task coverage, refining evaluation dimensions, and strengthening data quality control. We also plan to increase dataset scale to improve robustness and discriminative power of evaluation results.

- **Optimization for Multi-Stream Generation**  
  Since interaction signals are typically variable-length inputs, the current framework is not yet optimal for high-concurrency generation. We will improve support for multi-stream generation through better scheduling, batch alignment, and throughput optimization.

- **Training Framework Development**  
  OpenWorldLib primarily aims to provide a standardized and extensible framework for world models and their inference pipelines, with a strong emphasis on comprehensiveness. In the next phase, we will introduce **world model training frameworks** built on several mainstream foundation models.

- **Datasets and Tooling Release**  
  Along with the training framework, we will release corresponding datasets, preprocessing pipelines, and starter examples to help users with reproduction, fine-tuning, and further development.
