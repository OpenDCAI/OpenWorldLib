## OpenWorldLib Planning

### Methods Being Integrated

We are continuously expanding the **World Model** method library in our framework.  
The representative methods currently being integrated are listed below. We warmly welcome more developers to submit PRs and help us improve the ecosystem together!

| Method | Paper | Code Repository | Main Direction / Task |
|---|---|---|---|
| **Solaris** | [paper](https://arxiv.org/pdf/2602.22208) | [code](https://github.com/solaris-wm/solaris) | Navigation Video Generation (multiplayer Minecraft world modeling) |
| **FantasyWorld** | [paper](https://arxiv.org/pdf/2509.21657) | [code](https://github.com/Fantasy-AMAP/fantasy-world) | 3D Scene Generation (unified video and 3D prediction) |
| **Ctrl-World** | [paper](https://arxiv.org/pdf/2510.10125) | [code](https://github.com/Robert-gyj/Ctrl-World) | VLA / Robot Manipulation (controllable generative world model) |

> If you have methods you’d like to integrate (e.g., new world models, evaluation benchmarks, or inference pipelines), feel free to open an Issue or submit a PR directly.


### Code Optimization
- The current repository may still contain some redundant code, which we will further streamline;
- The interactions between different modules are not yet seamless enough. We will work towards enabling more flexible and natural integration between components;
- The performance of some existing methods still requires further improvement, specifically:
  - The sky layers generated in wonderworld are not yet satisfactory;


### Roadmap

- **Continuous Benchmark Enhancement**  
  We will further improve the current benchmark suite by expanding task coverage, refining evaluation dimensions, and strengthening data quality control. We also plan to increase dataset scale to improve robustness and discriminative power of evaluation results.

- **Optimization for Multi-Stream Generation**  
  Since interaction signals are typically variable-length inputs, the current framework is not yet optimal for high-concurrency generation. We will improve support for multi-stream generation through better scheduling, batch alignment, and throughput optimization.

- **Training Framework Development**  
  SceneFlow primarily aims to provide a standardized and extensible framework for world models and their inference pipelines, with a strong emphasis on comprehensiveness. In the next phase, we will introduce **world model training frameworks** built on several mainstream foundation models.

- **Datasets and Tooling Release**  
  Along with the training framework, we will release corresponding datasets, preprocessing pipelines, and starter examples to help users with reproduction, fine-tuning, and further development.
