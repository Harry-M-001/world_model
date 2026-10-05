# 低精度世界模型的可用性判据与设计方法
## Usability Criteria and Design Methodology for Low-Precision World Models

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23134088.svg)](https://doi.org/10.5281/zenodo.23134088)

四个阶段、四篇预印本的完整代码与论文集合。核心问题：**低精度世界模型何时可用、如何设计**——当推理端量化（INT8/FP8/FP4）压缩世界模型的潜空间时，预测误差每步被注入并沿自回归轨迹累积，直至超出容差判据；本工作给出度量、判据、闭环验证、设计方法与模态边界的完整回答。

## 论文链条

| 篇 | 文件 | 主题 | 页数 |
|---|---|---|---|
| P1 | `paper/world_model_usability_chinaxiv.pdf` | 可用性度量（λ₁ 误差放大率、崩坏视界 T_crash、临界位宽 b\*、floor 判据） | 26 |
| P2 | `paper/chinaxiv_second_phase.pdf` | 判据框架两域验证 + 设计轨三层裁决（损失塑形 ✗ / 更新几何 ✗ / 架构 ✓） | 6 |
| P3 | `paper/chinaxiv_third_phase.pdf` | 解析核适用边界（四点单调刻画）+ 混合架构（公式打底+有界残差） | 5 |
| P4 | `paper/chinaxiv_fourth_phase.pdf` | 音频模态判据可行性（负结果链 + 两段式视界模型 + 音画探针） | 5 |

全部实验**预注册**（跑前写死判据与门限）、**数字审计**（关键数字逐一对拍归档 JSON）、**可复算**（固定 seed + 归档产物）。

## 核心结论

1. **架构层是唯一出口**：损失塑形与更新几何在受控实验中无效，有界残差 Koopman（+谱约束）通过预注册门限；
2. **解析核价值 = f(解析形式覆盖度)** 单调：潜空间坐标错（正确口径 a=0.062，近中性）→ 显式坐标形式错（a=0.69，旧口径待重测）→ 形式完整（0 完美）→ 部分已知（0.83，混合臂修复至 0.56）；
3. **混合架构全线胜学习基线**：合成域 32×/100×/2.7×，真实域（PushT）action 驱动解析 a=0.02 近临界、纯解析覆盖 96%；
4. **判据边界补丁**：cos 阈值有效步数在任务随机性饱和区是钝感指标（X5 因果实验）；两段式视界模型 h\*(η) = min(T_crash(a,η), h_sat)（B1）；
5. **音频模态**：mel 能量潜码口径下判据不可测（五次判据门负结果链，失败模式三分类可诊断）——模态无关性有边界，波形级潜码/真实音频数据是出路；
6. **实时闭环**：8GB 单卡 59.1 FPS（18.15ms/步），架构+重置组合有效步数 8.9×。

## 代码结构

```
code/
├── 基础设施
│   ├── mvmnist.py            # Moving MNIST 数据（生成器逻辑与 rng 语义）
│   ├── vae.py                # VAE 编解码与潜码提取
│   ├── multiphys.py          # 多物理域合成（匀速/重力/弹性碰撞）
│   ├── q12_min_train.py      # 判据口径参考实现（a/margin/h*）
│   ├── q40_koopman.py        # 有界残差 Koopman（谱约束）
│   └── warn_component.py     # 崩坏预警组件（CollapseWarner）
├── P2 判据两域验证
│   ├── t2b_route.py          # 多物理域路由
│   ├── q2 系列               # scene 三件套（margin/a/量化）
│   └── q3 系列               # 渲染闭环（对齐 22.4dB / 59.1FPS）
├── P3 解析核
│   ├── r0/r1/r1b/r3/r3b      # 负结果两例 + 三档梯度 + 混合臂
│   ├── x3_koop_closedloop.py # koop 进闭环（重置收益 12.7×）
│   └── x5_decouple.py        # η 注入因果实验（判据边界补丁）
├── P4 音频模态
│   ├── m1_audio_probe.py     # 判据门（正弦纯音，FAIL）
│   ├── m1b/m1c/m1d           # 噪声/谐波/零训练（FAIL 链）
│   ├── m1e_deep.py           # 深化验证（D2 信号不足）
│   ├── m1h_pca16.py          # PCA16 信息集中（G2/G3 首过）
│   ├── m1i_a_alt.py          # Jacobian 口径（判据门 PASS）
│   ├── m2_audio_suite.py     # 音频三判据全套 + 模态对照表
│   ├── m0_full_av.py         # 音画探针（全事件对齐断言）
│   ├── b1_a_hstar.py         # a–h* 定量关系（两段式模型）
│   └── x6c_simulator_replay.py # 仿真器重放（一致率 1.0）
└── d2b_pusht_suite.py        # pusht 三件套（第二真实域验证）
```

## 复现

```bash
# 环境：Python 3.11 + PyTorch 2.x + CUDA（8GB 即可）
pip install torch numpy scipy matplotlib

# 判据参考实现（视觉 MM 域）
python code/q12_min_train.py --arm all

# P3 解析核主实验（四点边界）
python code/r3_multiphys_core.py
python code/r1b_hybrid_action.py

# P4 音频判据门链
python code/m1_audio_probe.py   # 判据门 FAIL（预期）
python code/m1i_a_alt.py        # 判据门 PASS（修正口径）
python code/m2_audio_suite.py   # 主结果

# 闭环
python code/x3_koop_closedloop.py
```

所有实验产物落盘 `logs/*.json`，关键数字有对应审计脚本（`_paper{2,3,4}_number_check.py`）。

## 引用

```bibtex
@misc{ma2026usability,
  title={低精度世界模型的可用性度量},
  author={马浩睿},
  year={2026},
  note={ChinaXiv 预印本（P1）},
  doi={10.5281/zenodo.23134088},
  url={https://doi.org/10.5281/zenodo.23134088}
}
```

## 许可

仅供学术研究使用。


## ⚠ 测量口径修订（2026-10-05）

本仓库论文与代码的一次系统性勘误。所有 `λ` / `a` 读数原先由**孪生两条分支共享 prev 帧**的旧写法测得，
它测的是 `ρ(∂f/∂st)` 而非模型自回归的误差放大，会把 `λ≈0` 的模型读成 `+0.53 ~ +0.67`。
已改为：两条分支各带自己的历史 + 两帧提升的 augmented 范数 + 固定窗口（见 `code/_twinlib.py`，含自检）。

| 口径 | 测的是什么 | 修正后实测 |
|---|---|---|
| λ₁（Jacobian 谱半径） | 模型映射本身的放大率 | **+0.022** |
| λ₁（孪生·各带历史） | 同上，独立印证 | **−0.006** |
| λ₁（崩坏轨迹 OLS 反解 ln ρ̂） | 同上，第三条证据 | **−0.0005** |
| 全误差 rollout 增长率 | 含每步注入，**三阶段** | 早期 +0.114 → 饱和 +0.005 |
| ~~孪生·共享 prev（旧）~~ | ~~ρ(∂f/∂st)~~ | ~~+0.53 ~ +0.67（伪读数，已废止）~~ |

**修正 ⇒ 模型固有放大率 λ₁ ≈ 0，崩坏由「每步注入 + 中性传播」驱动，不是指数放大。**
零参数闭式 `T_crash = ln(1+δ*(e^λ−1)/η)/λ` 在 λ≈0 下失效（崩坏预警命中率 81.2%→35.3%）；
改用**注入-累积模型** `δ_t ≈ δ₁ + ĉ·t` 重标定后命中率 **81.6%**、误报 17→6
⇒ 判据的正确形式不是「指数放大视界」而是「注入累积视界」。

修正涉及：第一/三阶段论文正文与摘要、`results/r0_analytic_core.json`、
`results/t14_warn_dense_h480_act.json`、以及代码中 `q12_min_train.py` / `r0_analytic_core.py` /
`r3_multiphys_core.py` / `t14_warn.py` 的孪生口径。**新增 `code/_twinlib.py` 作为唯一正确口径库。**

## 代码包

- `world_model_core_code.zip` —— 核心代码包（53 文件：26 脚本 + 归档 JSON + 文档）
- `world_model_release.zip` —— GitHub Release 用打包（与仓库内容一致）
