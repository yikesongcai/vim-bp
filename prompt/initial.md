**Role:**
你是一位精通 PyTorch 和联邦学习框架（特别是 EasyFL/FLGo）的高级研究工程师。我正在进行一项关于 **“可验证联邦遗忘学习激励机制 (VIM-BP)”** 的研究。

**Context:**
我已经下载了 `EasyFL` (即 `FLGo`) 的源代码。现在的目标是基于该框架实现一个 Demo，验证我的核心 Idea：

1. **验证机制**：利用“放射性数据（Radioactive Data）”作为纠缠特征，验证客户端是否真的执行了遗忘。
2. **激励机制**：利用“背包约束下的多臂老虎机（Knapsack UCB）”算法，在有限预算下动态选择诚实的客户端。

**Task:**
请根据以下 **4个模块** 的需求，为我编写/修改相应的 Python 代码。请尽量遵循 EasyFL 的类继承结构（`BasicServer`, `BasicClient`）。

---

#### 🛠️ Module 1: 数据集改造 (Radioactive Poisoning)

**目标**：在 CIFAR-10 数据集上，为特定类别（如 Class 0）的样本注入微弱的“放射性噪声”，使其与正常数据在特征上纠缠。
**代码要求**：

* 编写一个 `RadioactiveTransform` 类，继承自 `torchvision.transforms`。
* 逻辑：生成一个固定的全局 Gaussian Noise 掩码（Trigger），将其以极小的权重（epsilon=0.05）叠加到图像上。
* 提供一个函数 `get_verification_set(test_data)`，专门提取出带有 Trigger 的样本作为服务器端的探测集（Probe Set）。

#### 🛠️ Module 2: 异质客户端模拟 (Client Variants)

**目标**：模拟不同类型的客户端行为，用于测试验证机制的有效性。
**代码要求**：

* 创建一个 `VIMClient` 类，继承自 EasyFL 的 `BasicClient`。
* 增加一个属性 `client_type`，支持以下模式：
* `'honest'`: 收到遗忘指令后，对目标数据执行 Gradient Ascent（梯度上升）操作。
* `'free_rider_lazy'`: 不执行任何遗忘操作，直接返回旧模型（假装已遗忘）。
* `'free_rider_smart'`: 对模型参数添加高斯噪声（试图通过破坏模型来混淆视听）。


* 重写 `reply()` 或 `unpack()` 方法，使其根据 `client_type` 返回不同的更新。

#### 🛠️ Module 3: 验证与激励核心 (Server Logic)

**目标**：这是核心创新点。服务器需要在聚合前验证模型，并基于结果更新 MAB 算法。
**代码要求**：

* 创建一个 `VIMServer` 类，继承自 `BasicServer`。
* **实现 `verify_submission(client_model)` 方法**：
1. 计算该模型在 `Radioactive Probe Set` 上的 Loss。
2. 如果 Loss **高于** 设定阈值（说明遗忘成功），返回 `verified=True`。
3. 同时检查在 `Normal Test Set` 上的准确率（Utility Check），防止模型被破坏。


* **实现 `KnapsackMAB` 类**：
1. 维护每个客户端的 `Q_value` (验证通过率) 和 `Cost` (报价)。
2. 实现 `select_clients(budget)` 方法，使用 **UCB-BwK** 逻辑（优先选择$\frac{UpperConfidenceBound}/{Cost}$  高的客户端）。


* **重写 `iterate()` 流程**：
* 在每轮开始时，调用 MAB 选择客户端。
* 发送遗忘请求。
* 收到更新后，逐个调用 `verify_submission`。
* 根据验证结果（0或1）更新 MAB 的状态。
* 仅聚合验证通过的模型。



#### 🛠️ Module 4: 实验启动脚本 (Runner)

**目标**：将上述模块串联起来运行。
**代码要求**：

* 使用 `flgo.gen_task` 生成一个 Non-IID 的 CIFAR-10 任务。
* 初始化 `VIMServer` 和 100 个 `VIMClient`。
* 设置 30% 的客户端为 `free_rider`，70% 为 `honest`。
* 运行 50 轮通信，并绘制“遗忘成功率”和“预算消耗效率”的曲线。

---
