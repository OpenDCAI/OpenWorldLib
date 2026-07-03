## OpenWorldLib Installation Details

In this document, we list the installation requirements and installation scripts for different methods, as shown in the table below.

<table>
<thead>
  <tr>
    <th align="center">Method</th>
    <th align="center">Python</th>
    <th align="center">CUDA</th>
    <th align="center">Key Dependencies</th>
    <th align="center">Install Command</th>
    <!-- <th align="center">Docs</th> -->
  </tr>
</thead>
<tbody>
  <tr>
    <td colspan="5" align="center"><b>🧭 Navigation Video Generation</b></td>
  </tr>
  <tr>
    <td align="center">MatrixGame2</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
    <!-- <td align="center"><a href="envs/env_method_a.md">📖</a></td> -->
  </tr>
  <tr>
    <td align="center">MatrixGame3</td>
    <td align="center">3.12 (recommended)</td>
    <td align="center">12.1+</td>
    <td>PyTorch 2.10.0, transformers 4.57.3, FlashAttention (optional but recommended)</td>
    <td><code>bash scripts/setup/matrix_game_3_install.sh</code><br/><code>bash scripts/test_inference/test_nav_video_gen.sh matrix-game-3</code></td>
  </tr>
  <tr>
    <td align="center">Hunyuan-GameCraft</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">HunyuanWorld-Voyager</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/lower_trans_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Hunyuan-WorldPlay</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Astra</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Yume</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">LingBot-World</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Sana-WM</td>
    <td align="center">3.10</td>
    <td align="center">12.1+</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/sana_wm_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Infinite-World</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1, flash-attn</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Cosmos3</td>
    <td align="center">3.11</td>
    <td align="center">12.8+</td>
    <td>PyTorch, Diffusers main, Transformers, Cosmos Guardrail</td>
    <td><code>bash scripts/setup/cosmos3_install.sh</code><br/><code>bash scripts/test_inference/test_nav_video_gen.sh cosmos3</code></td>
  </tr>
  <tr>
    <td align="center">GammaWorld</td>
    <td align="center">3.10</td>
    <td align="center">12.8</td>
    <td>PyTorch, Transformer Engine, Gamma-World package</td>
    <td><code>bash scripts/setup/gamma_world_install.sh</code><br/><code>bash scripts/test_inference/test_nav_video_gen.sh gamma-world</code></td>
  </tr>
  <tr>
    <td align="center">Solaris</td>
    <td align="center">3.10</td>
    <td align="center">12.1+</td>
    <td>JAX 0.6.2, Flax 0.10.6, PyTorch 2.8.0</td>
    <td><code>bash scripts/setup/solaris_install.sh</code><br/><code>bash scripts/test_inference/test_nav_video_gen.sh solaris</code></td>
  </tr>
  <tr>
    <td align="center">MemFlow</td>
    <td align="center">3.10</td>
    <td align="center">12.8</td>
    <td>PyTorch 2.8.0, Diffusers 0.31.0, FlashAttention</td>
    <td><code>bash scripts/setup/memflow_install.sh</code><br/><code>bash scripts/test_inference/test_nav_video_gen.sh memflow</code></td>
  </tr>
  <tr>
    <td colspan="5" align="center"><b>🎨 3D Scene Generation</b></td>
  </tr>
  <tr>
    <td align="center">FlashWorld</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/flash_world_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">HunyuanWorld-Mirror</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/hunyuan_mirror_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Hunyuan-worldplay2 / HY-World-2.0</td>
    <td align="center">3.11.15</td>
    <td align="center">12.8</td>
    <td>PyTorch, WorldMirror 2.0, gsplat</td>
    <td><code>bash scripts/setup/hunyuan_worldplay2_install.sh</code><br/><code>bash scripts/test_inference/test_3d_scene_gen.sh hunyuan-worldplay2</code></td>
  </tr>
  <tr>
    <td align="center">VGGT</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_3D_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Pi3</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_3D_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Pi3X</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_3D_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">LoGeR</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.6.0</td>
    <td><code>bash scripts/setup/loger_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">InfiniteVGGT</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_3D_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">FantasyWorld</td>
    <td align="center">3.10</td>
    <td align="center">12.1+</td>
    <td>PyTorch 2.4.0+, Diffusers 0.31.0+, Transformers 4.49.0</td>
    <td><code>bash scripts/setup/fantasy_world_install.sh</code><br/><code>bash scripts/test_inference/test_3d_scene_gen.sh fantasy-world</code></td>
  </tr>
  <tr>
    <td colspan="5" align="center"><b> 🤖 Vision Language Action</b></td>
  </tr>
  <tr>
    <td align="center">&pi;0</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">&pi;0.5</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">GigaBrain-0</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Spirit v1.5</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh 
    bash scripts/setup/libero_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Lingbot-va</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_lingbot_va.sh</code></td>
  </tr>
  <tr>
    <td align="center">Ctrl-World</td>
    <td align="center">3.11</td>
    <td align="center">12.6+</td>
    <td>PyTorch 2.7.1, Diffusers 0.34.0, Transformers 4.48.1</td>
    <td><code>bash scripts/setup/ctrl_world_install.sh</code><br/><code>bash scripts/test_inference/test_vla.sh ctrl-world</code></td>
  </tr>
  <tr>
    <td colspan="5" align="center"><b>🎓 Multimodal Reasoning</b></td>
  </tr>
  <tr>
    <td align="center">OmniVinci</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/omnivinci_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Qwen2.5-Omni</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.6.0</td>
    <td><code>bash scripts/setup/default_audio_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">SpatialLadder</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">SpatialReasoner</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td colspan="5" align="center"><b>🧩 Interactive Video</b></td>
  </tr>
  <tr>
    <td align="center">Sora 2</td>
    <td align="center">3.10</td>
    <td align="center"></td>
    <td></td>
    <td>Only need API</td>
  </tr>
  <tr>
    <td align="center">Veo 3</td>
    <td align="center">3.10</td>
    <td align="center"></td>
    <td></td>
    <td>Only need API</td>
  </tr>
  <tr>
    <td align="center">Wan 2.5</td>
    <td align="center">3.10</td>
    <td align="center"></td>
    <td></td>
    <td>Only need API</td>
  </tr>
  <tr>
    <td align="center">Wan 2.2</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">WoW</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Cosmos-Predict 2.5</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">Recammaster</td>
    <td align="center">3.10</td>
    <td align="center">12.1</td>
    <td>PyTorch 2.5.1</td>
    <td><code>bash scripts/setup/default_install.sh</code></td>
  </tr>
  <tr>
    <td colspan="5" align="center"><b>⚙️ Simulation Environment</b></td>
  </tr>
  <tr>
    <td align="center">AI2thor</td>
    <td align="center">3.9</td>
    <td align="center"> </td>
    <td> </td>
    <td><code>bash scripts/setup/ai2thor_install.sh</code></td>
  </tr>
  <tr>
    <td colspan="5" align="center"><b>🎵 Audio Generation</b></td>
  </tr>
  <tr>
    <td align="center">MMAudio</td>
    <td align="center">3.10</td>
    <td align="center">12.1 </td>
    <td> PyTorch 2.6.0</td>
    <td><code>bash scripts/setup/audio_generation_default_install.sh</code></td>
  </tr>
  <tr>
    <td align="center">ThinkSound</td>
    <td align="center">3.10</td>
    <td align="center">12.1 </td>
    <td> PyTorch 2.6.0</td>
    <td><code>bash scripts/setup/audio_generation_default_install.sh</code></td>
  </tr>
</tbody>
</table>
