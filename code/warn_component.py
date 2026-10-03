# -*- coding: utf-8 -*-
"""Q3.4 · 崩坏预警可复用组件

供 Q3.3 实时闭环调用。两个判据 + 一个结论重述：

  A1 解析判据（预飞行预算）
    给定模型参数（λ, η(b), δ*），用零参数公式预测崩坏视界。
    在 rollout 开始前调用一次，设定本回合的"信任预算"。

  A2a 在线 3σ 监测
    给定训练分布的范数统计，实时监测 rollout 是否偏离。
    在 rollout 的每一步调用，返回是否应停止信任。

  结论重述（Q0 模板，平均偏差为主判据，命中率已证饱和）
    形状修复后 n=207 可判（删失 53.8%）：
    四臂全部低于常数预测平凡基线 83.1%——原「一条 if 语句就够了」已撤回。
    A1 解析判据最好：命中率 81.2%，平均偏差 −1.29 步（系统性偏早）；
    Δλ 修正后 73.4%（偏早 4.04 步）——作为"预算参考"仍有信息量，
    作为"精确停止信号"不可靠（排序可用窗口 ≈1 步，见 G1①）。

用法（Q3.3 实时闭环）：
    from warn_component import CollapseWarner
    w = CollapseWarner(lambda_model, eta_by_bits, delta_star)
    budget = w.pre_flight(b=8)           # → 预测崩坏步数（int 或 None）
    for step in rollout:
        ...
        if w.online_check(z_norm, step_idx):
            break                        # 偏离训练分布，停止信任
"""
import math


class CollapseWarner:
    """崩坏预警组件（Q3.4 → Q3.3 接口）。

    Parameters
    ----------
    mu_norm : float
        训练集潜向量范数的均值（从训练数据统计，不是 rollout 数据）。
    sd_norm : float
        训练集潜向量范数的标准差。
    n_sigma : float, optional
        偏离阈值（默认 3σ）。⇒ A2a 的灵敏度参数。
    """

    def __init__(self, mu_norm: float, sd_norm: float, n_sigma: float = 3.0):
        if sd_norm <= 0:
            raise ValueError(f"sd_norm 必须为正，得到 {sd_norm}")
        self.mu = mu_norm
        self.sd = sd_norm
        self.n_sigma = n_sigma
        self._baseline = abs(mu_norm) + n_sigma * sd_norm

    # ---- A1 解析判据（预飞行预算）----

    @staticmethod
    def pre_flight(lam: float, eta: float, delta_star: float = 1.0):
        """零参数公式：给定 λ、η(b)、δ*，预测崩坏视界。

        Parameters
        ----------
        lam : float
            模型自回归轨迹的 Lyapunov 指数（float64 孪生实测，非数据真值）。
        eta : float
            该位宽 b 的每步量化注入范数（√L·Δ/√12 或实测值）。
        delta_star : float
            崩坏阈值（‖z_model − z_true‖ 超过此值即视为崩坏；默认 1.0）。

        Returns
        -------
        int or None
            预测崩坏步数（≥1），或 None（λ≤0 且 η ≤ δ*(1−e^λ) 时永不崩）。
        """
        if abs(lam) < 1e-6:
            return None
        e = math.exp(lam)
        val = 1.0 + delta_star * (e - 1) / max(eta, 1e-12)
        if val <= 0:
            return None
        return max(1, round(math.log(val) / lam))

    def pre_flight_for(self, lam, eta_by_b, delta_star=1.0, bs=None):
        """对多个位宽批量预测。返回 {b: T_crash or None}。"""
        bs = bs or sorted(eta_by_b.keys())
        return {b: self.pre_flight(lam, eta_by_b[b], delta_star) for b in bs}

    # ---- A2a 在线 3σ 监测 ----

    def online_check(self, z_norm: float) -> bool:
        """给定当前步的潜向量范数，返回是否应停止信任（True = 偏离 > nσ）。"""
        return abs(z_norm - self.mu) > self.n_sigma * self.sd

    def deviation_sigma(self, z_norm: float) -> float:
        """返回当前范数偏离训练均值多少个 σ（带符号）。"""
        return (z_norm - self.mu) / self.sd

    @property
    def threshold(self) -> float:
        """范数警戒线（mu + n_sigma × sd）。"""
        return self._baseline

    # ---- 便捷方法 ----

    def online_check_batch(self, norms) -> list:
        """批量检查；返回逐步 bool 列表。"""
        return [self.online_check(z) for z in norms]

    def first_violation(self, norms) -> int or None:
        """返回第一个超限步的索引（1-based），或 None。"""
        for i, z in enumerate(norms):
            if self.online_check(z):
                return i + 1
        return None


# ---- 重述结论（Q0 模板）----

RESTATED_CONCLUSION = """\
## T14 崩坏预警 · 结论重述（Q0 模板，平均偏差为主判据）

### 背景
形状修复后 n=207 可判（删失 53.8%）。命中率类判据被平凡基线饱和
（常数预测 T=1..6 命中率即 100%），已按 Q0 模板改报平均偏差。

### 四臂结果（平均偏差为正 = 偏早 / 为负 = 偏晚）
| 臂 | 命中率 | 平均偏差 | 读法 |
|---|---|---|---|
| A1 解析判据 | 81.2% | −1.29 步（偏早） | 最好；作预算参考仍有信息量 |
| A3 Laya 421M | 71.0% | −5.22 步（偏早） | 次之 |
| A2a 3σ 规则 | 57.0% | +0.39 步（偏晚） | 差 |
| A2b 二阶差分 | 24.6% | −0.81 步 | 最差 |
| 平凡基线 | 83.1% | — | 四臂均未超过 |

### 操作读法
- A1 解析判据的命中率（81.2%）未超过平凡基线（83.1%）⇒ 不是精确停止信号。
- 但 A1 的平均偏差仅 −1.29 步（系统性偏早）⇒ 作为「预算参考」（提前知道大概几步后崩）
  仍有信息量，尤其当 8 步 rollout 评分脱钩（G1① ρ≈0）时，
  预飞行预算是唯一可用的先验。
- Δλ 修正后 A1 = 73.4%（偏早 4.04 步）⇒ 量纲修正反而更差，
  **零参数判据本身是准的**这一论证不再成立。
- **对 Q3.3 的含义**：预警组件不应单独使用，应与「每步重规划」（H=1）配合——
  预算告诉你大概几步后崩，重规划每步纠正，两者互补。
"""
