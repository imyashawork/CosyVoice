# CosyVoice 使用指南

本指南基于 `example.py`，详细介绍 CosyVoice 各模型的 API 用法。

## 目录

- [1. 环境准备](#1-环境准备)
- [2. 模型概览](#2-模型概览)
- [3. AutoModel 工厂函数](#3-automodel-工厂函数)
- [4. 通用返回格式](#4-通用返回格式)
- [5. 推理方法详解](#5-推理方法详解)
  - [5.1 SFT 预训练音色合成 (inference_sft)](#51-sft-预训练音色合成-inference_sft)
  - [5.2 零样本语音克隆 (inference_zero_shot)](#52-零样本语音克隆-inference_zero_shot)
  - [5.3 跨语种复刻 (inference_cross_lingual)](#53-跨语种复刻-inference_cross_lingual)
  - [5.4 自然语言指令控制 (inference_instruct, v1 专用)](#54-自然语言指令控制-inference_instruct-v1-专用)
  - [5.5 指令控制 2 (inference_instruct2, v2/v3)](#55-指令控制-2-inference_instruct2-v2v3)
  - [5.6 语音转换 (inference_vc)](#56-语音转换-inference_vc)
- [6. 说话人保存与复用](#6-说话人保存与复用)
- [7. 流式推理与双向流式](#7-流式推理与双向流式)
- [8. 细粒度控制标记](#8-细粒度控制标记)
- [9. 指令控制参考列表](#9-指令控制参考列表)
- [10. Web UI 启动](#10-web-ui-启动)
- [11. FastAPI 服务部署](#11-fastapi-服务部署)
- [12. 注意事项](#12-注意事项)

---

## 1. 环境准备

```bash
# 激活 conda 环境
conda activate cosyvoice

# 所有示例都需要将 Matcha-TTS 加入路径
# 在脚本开头添加：
import sys
sys.path.append('third_party/Matcha-TTS')
```

## 2. 模型概览

| 模型 | 目录名 | 对应类 | 特点 |
|------|--------|--------|------|
| CosyVoice 1.0 (SFT) | `pretrained_models/CosyVoice-300M-SFT` | `CosyVoice` | 预训练音色，支持 SFT |
| CosyVoice 1.0 (Base) | `pretrained_models/CosyVoice-300M` | `CosyVoice` | 零样本、跨语种、VC |
| CosyVoice 1.0 (Instruct) | `pretrained_models/CosyVoice-300M-Instruct` | `CosyVoice` | 自然语言指令控制 |
| CosyVoice 2.0 | `pretrained_models/CosyVoice2-0.5B` | `CosyVoice2` | 零样本、instruct2、双向流式 |
| Fun-CosyVoice 3.0 | `pretrained_models/Fun-CosyVoice3-0.5B` | `CosyVoice3` | 最新模型，9 语种+18 方言 |

## 3. AutoModel 工厂函数

`AutoModel` 是工厂函数（非类），根据 `model_dir` 中的 YAML 配置文件自动选择对应模型类：

```python
from cosyvoice.cli.cosyvoice import AutoModel
import torchaudio

# 传入本地路径，自动识别模型版本
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

**构造参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_dir` | `str` | 必填 | 本地路径或 ModelScope 模型 ID。路径不存在时自动从 ModelScope 下载 |
| `load_jit` | `bool` | `False` | 加载 TorchScript 加速（需要 CUDA）。仅 v1/v2 支持，v3 无此参数 |
| `load_trt` | `bool` | `False` | 加载 TensorRT 加速（需要 CUDA） |
| `load_vllm` | `bool` | `False` | 使用 vLLM 加速 LLM 推理（仅 v2/v3，需要 CUDA） |
| `fp16` | `bool` | `False` | 半精度推理（CPU 自动禁用） |
| `trt_concurrent` | `int` | `1` | TensorRT 并发上下文数 |
| `device` | `str` | `None` | 推理设备（`'cuda'`/`'cpu'`），不指定时自动检测 |

**实例属性：**

- `cosyvoice.sample_rate` — 输出音频采样率（如 22050）
- `cosyvoice.frontend.spk2info` — 内置说话人字典

## 4. 通用返回格式

所有推理方法都是**生成器**，通过 `yield` 返回字典：

```python
{'tts_speech': torch.Tensor}  # shape: (1, samples), dtype: float
```

标准迭代方式：

```python
for i, j in enumerate(cosyvoice.inference_xxx(...)):
    torchaudio.save(f'output_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

`i` 是音频块索引，`j` 是包含 `tts_speech` 张量的字典。

---

## 5. 推理方法详解

### 5.1 SFT 预训练音色合成 (inference_sft)

使用模型内置的预训练说话人进行合成。仅 CosyVoice v1 SFT 模型支持。

```python
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-SFT')

# 查看可用的预训练说话人
print(cosyvoice.list_available_spks())
# 输出示例: ['中文女', '中文男', '英文女', '英文男', '日语男', '粤语女']

# 合成语音
for i, j in enumerate(cosyvoice.inference_sft(
        '你好，我是通义生成式语音大模型，请问有什么可以帮您的吗？',
        '中文女',
        stream=False)):
    torchaudio.save(f'sft_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tts_text` | `str` | 必填 | 要合成的文本 |
| `spk_id` | `str` | 必填 | 预训练说话人 ID（通过 `list_available_spks()` 查看） |
| `stream` | `bool` | `False` | 是否流式输出 |
| `speed` | `float` | `1.0` | 语速倍率（仅非流式模式有效） |
| `text_frontend` | `bool` | `True` | 是否进行文本规范化（数字、符号等） |

---

### 5.2 零样本语音克隆 (inference_zero_shot)

提供一段参考音频和对应文本，克隆该说话人音色合成新文本。

```python
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# CosyVoice3 的 prompt_text 需要以 "You are a helpful assistant.<|endofprompt|>" 开头
for i, j in enumerate(cosyvoice.inference_zero_shot(
        '八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。',
        'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
        './asset/zero_shot_prompt.wav',
        stream=False)):
    torchaudio.save(f'zero_shot_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

CosyVoice v1/v2 的 `prompt_text` 不需要前缀：

```python
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M')

for i, j in enumerate(cosyvoice.inference_zero_shot(
        '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。',
        '希望你以后能够做的比我还好呦。',
        './asset/zero_shot_prompt.wav')):
    torchaudio.save(f'zero_shot_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tts_text` | `str` 或 `Generator[str]` | 必填 | 要合成的文本。v2/v3 支持传入生成器实现双向流式 |
| `prompt_text` | `str` | 必填 | 参考音频的精确转录文本（使用 `zero_shot_spk_id` 时可传空字符串） |
| `prompt_wav` | `str` 路径或数组 | 必填 | 参考音频文件路径或波形数组，时长 <= 30 秒 |
| `zero_shot_spk_id` | `str` | `''` | 使用 `add_zero_shot_spk` 预保存的说话人 ID |
| `stream` | `bool` | `False` | 是否流式输出 |
| `speed` | `float` | `1.0` | 语速倍率（仅非流式模式有效） |
| `text_frontend` | `bool` | `True` | 是否进行文本规范化 |

> **注意：** `tts_text` 长度不应短于 `prompt_text` 的一半，否则可能影响效果。

---

### 5.3 跨语种复刻 (inference_cross_lingual)

克隆参考音频的音色，但用另一种语言说话。不需要 `prompt_text`。

```python
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M')

# CosyVoice v1: 使用语言标签前缀
for i, j in enumerate(cosyvoice.inference_cross_lingual(
        '<|en|>And then later on, fully acquiring that company. So keeping management in line.',
        './asset/cross_lingual_prompt.wav')):
    torchaudio.save(f'cross_lingual_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

```python
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# CosyVoice v3: 支持 [breath] 等细粒度控制标记
for i, j in enumerate(cosyvoice.inference_cross_lingual(
        'You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，都很熟悉。[breath]',
        './asset/zero_shot_prompt.wav',
        stream=False)):
    torchaudio.save(f'fine_grained_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

日语需转换为片假名：

```python
# 原文: 歴史的世界においては、過去は単に過ぎ去ったものではない
# 转换为片假名:
for i, j in enumerate(cosyvoice.inference_cross_lingual(
        'You are a helpful assistant.<|endofprompt|>レキシ テキ セカイ ニ オイ テ ワ、カコ ワ タンニ スギサッ タ モノ デ ワ ナイ。',
        './asset/zero_shot_prompt.wav',
        stream=False)):
    torchaudio.save(f'japanese_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tts_text` | `str` | 必填 | 合成文本。v1 支持语言标签前缀 `<|zh|>` `<|en|>` `<|ja|>` `<|yue|>` `<|ko|>` |
| `prompt_wav` | 路径或数组 | 必填 | 参考音频 |
| `zero_shot_spk_id` | `str` | `''` | 预保存说话人 ID |
| `stream` | `bool` | `False` | 是否流式输出 |
| `speed` | `float` | `1.0` | 语速倍率 |
| `text_frontend` | `bool` | `True` | 是否文本规范化 |

---

### 5.4 自然语言指令控制 (inference_instruct, v1 专用)

通过自然语言描述角色和风格，控制语音表现。仅 `CosyVoice-300M-Instruct` 模型支持。

```python
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M-Instruct')

for i, j in enumerate(cosyvoice.inference_instruct(
        '在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。',
        '中文男',
        "Theo 'Crimson', is a fiery, passionate rebel leader. Fights with fervor for justice, but struggles with impulsiveness.<|endofprompt|>")):
    torchaudio.save(f'instruct_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tts_text` | `str` | 必填 | 要合成的文本，支持嵌入 `<strong>` `[laughter]` 等标记 |
| `spk_id` | `str` | 必填 | 预训练说话人 ID |
| `instruct_text` | `str` | 必填 | 角色风格描述，以 `<|endofprompt|>` 结尾 |
| `stream` | `bool` | `False` | 是否流式输出 |
| `speed` | `float` | `1.0` | 语速倍率 |
| `text_frontend` | `bool` | `True` | 是否文本规范化 |

> **注意：** 在 CosyVoice2/3 上调用此方法会触发 `AssertionError`，请使用 `inference_instruct2`。

---

### 5.5 指令控制 2 (inference_instruct2, v2/v3)

v2/v3 的指令控制方法，使用参考音频音色 + 自然语言指令控制方言、情绪、语速等。

```python
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# 广东话
for i, j in enumerate(cosyvoice.inference_instruct2(
        '好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。',
        'You are a helpful assistant. 请用广东话表达。<|endofprompt|>',
        './asset/zero_shot_prompt.wav',
        stream=False)):
    torchaudio.save(f'instruct_cantonese_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)

# 快语速
for i, j in enumerate(cosyvoice.inference_instruct2(
        '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。',
        'You are a helpful assistant. 请用尽可能快地语速说一句话。<|endofprompt|>',
        './asset/zero_shot_prompt.wav',
        stream=False)):
    torchaudio.save(f'instruct_fast_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

CosyVoice2 同样支持：

```python
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')

# 四川话
for i, j in enumerate(cosyvoice.inference_instruct2(
        '收到好友从远方寄来的生日礼物，那份意外的惊喜与深深的祝福让我心中充满了甜蜜的快乐。',
        '用四川话说这句话<|endofprompt|>',
        './asset/zero_shot_prompt.wav')):
    torchaudio.save(f'instruct_sichuan_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `tts_text` | `str` | 必填 | 要合成的文本 |
| `instruct_text` | `str` | 必填 | 指令文本。v3 通常以 `You are a helpful assistant.` 开头，以 `<|endofprompt|>` 结尾 |
| `prompt_wav` | 路径或数组 | 必填 | 参考音频（目标音色） |
| `zero_shot_spk_id` | `str` | `''` | 预保存说话人 ID |
| `stream` | `bool` | `False` | 是否流式输出 |
| `speed` | `float` | `1.0` | 语速倍率 |
| `text_frontend` | `bool` | `True` | 是否文本规范化 |

---

### 5.6 语音转换 (inference_vc)

将源音频的内容转换为目标说话人的音色，不需要文本输入。

```python
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice-300M')

# source_wav 提供内容，prompt_wav 提供目标音色
for i, j in enumerate(cosyvoice.inference_vc(
        './asset/cross_lingual_prompt.wav',   # 源音频（内容）
        './asset/zero_shot_prompt.wav')):     # 目标音频（音色）
    torchaudio.save(f'vc_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

**参数说明：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `source_wav` | 路径或数组 | 必填 | 源语音（保留内容） |
| `prompt_wav` | 路径或数组 | 必填 | 目标语音（提供音色） |
| `stream` | `bool` | `False` | 是否流式输出 |
| `speed` | `float` | `1.0` | 语速倍率 |

---

## 6. 说话人保存与复用

通过 `add_zero_shot_spk` 预计算说话人特征，后续无需重复传入参考音频：

```python
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')

# 首次使用：正常零样本推理
for i, j in enumerate(cosyvoice.inference_zero_shot(
        '收到好友从远方寄来的生日礼物。',
        '希望你以后能够做的比我还好呦。',
        './asset/zero_shot_prompt.wav')):
    torchaudio.save(f'zero_shot_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)

# 保存说话人特征
assert cosyvoice.add_zero_shot_spk(
        '希望你以后能够做的比我还好呦。',
        './asset/zero_shot_prompt.wav',
        'my_zero_shot_spk') is True

# 后续使用：通过 zero_shot_spk_id 复用，无需再传 prompt_text 和 prompt_wav
for i, j in enumerate(cosyvoice.inference_zero_shot(
        '笑容如花儿般绽放。',
        '', '',
        zero_shot_spk_id='my_zero_shot_spk')):
    torchaudio.save(f'zero_shot_saved_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)

# 持久化到磁盘（保存到 {model_dir}/spk2info.pt）
cosyvoice.save_spkinfo()
```

| 方法 | 说明 |
|------|------|
| `add_zero_shot_spk(prompt_text, prompt_wav, zero_shot_spk_id)` | 预计算并存储说话人特征，返回 `True` 表示成功 |
| `save_spkinfo()` | 将内存中的说话人信息持久化到 `{model_dir}/spk2info.pt` |
| `list_available_spks()` | 返回所有可用说话人 ID 列表（含预训练和自定义） |

---

## 7. 流式推理与双向流式

### 流式输出 (stream=True)

```python
for i, j in enumerate(cosyvoice.inference_zero_shot(
        '收到好友从远方寄来的生日礼物。',
        '希望你以后能够做的比我还好呦。',
        './asset/zero_shot_prompt.wav',
        stream=True)):  # 逐块输出音频
    torchaudio.save(f'stream_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### 双向流式 (Bi-Streaming)

CosyVoice2/3 的 `inference_zero_shot` 支持将**文本生成器**作为 `tts_text`，实现 LLM 边生成文本、TTS 边合成音频：

```python
cosyvoice = AutoModel(model_dir='pretrained_models/CosyVoice2-0.5B')

def text_generator():
    yield '收到好友从远方寄来的生日礼物，'
    yield '那份意外的惊喜与深深的祝福'
    yield '让我心中充满了甜蜜的快乐，'
    yield '笑容如花儿般绽放。'

for i, j in enumerate(cosyvoice.inference_zero_shot(
        text_generator(),                         # 传入生成器
        '希望你以后能够做的比我还好呦。',
        './asset/zero_shot_prompt.wav',
        stream=False)):
    torchaudio.save(f'bistream_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

> **注意：** 双向流式不支持与 vLLM 同时使用。文本仍需基本的句子分割，LLM 无法处理任意长度的句子。

---

## 8. 细粒度控制标记

在 `tts_text` 中嵌入以下标记，控制语音细节：

### CosyVoice2 支持的标记

| 标记 | 说明 |
|------|------|
| `[laughter]` | 笑声 |
| `[breath]` | 呼吸声 |
| `[quick_breath]` | 快速呼吸 |
| `[cough]` | 咳嗽 |
| `[clucking]` | 咂嘴 |
| `[accent]` | 口音 |
| `[noise]` | 噪声 |
| `[hissing]` | 嘶嘶声 |
| `[sigh]` | 叹气 |
| `[lipsmack]` | 抿唇 |
| `[mn]` | 嗯 |
| `<strong>...</strong>` | 重读强调 |
| `<laughter>...</laughter>` | 带笑意说话 |
| `[vocalized-noise]` | 发声噪声 |

### CosyVoice3 额外支持

CosyVoice3 在上述基础上，还支持英语 CMU 音标标记（如 `[AA]`, `[AE]`, `[AH]` 等）用于发音控制。

**示例：**

```python
# 在文本中嵌入笑声和呼吸
text = '在他讲述那个荒诞故事的过程中，他突然[laughter]停下来，因为他自己也被逗笑了[laughter]。'

# CosyVoice3 的呼吸控制
text = 'You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人[breath]在乡里面住的要习惯一点，[breath]邻居都很活络。[breath]'
```

### 拼音纠错 (Hotfix)

CosyVoice3 支持在文本中直接标注拼音，纠正多音字发音：

```python
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

for i, j in enumerate(cosyvoice.inference_zero_shot(
        '高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。',  # [j][ǐ] 标注"给"的正确读音
        'You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。',
        './asset/zero_shot_prompt.wav',
        stream=False)):
    torchaudio.save(f'hotfix_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

---

## 9. 指令控制参考列表

CosyVoice3 的 `inference_instruct2` 支持以下指令（源自 `cosyvoice/utils/common.py`）：

### 方言控制

| 指令 | 方言 |
|------|------|
| `请用广东话表达。` | 广东话 |
| `请用东北话表达。` | 东北话 |
| `请用甘肃话表达。` | 甘肃话 |
| `请用贵州话表达。` | 贵州话 |
| `请用河南话表达。` | 河南话 |
| `请用湖北话表达。` | 湖北话 |
| `请用湖南话表达。` | 湖南话 |
| `请用江西话表达。` | 江西话 |
| `请用闽南话表达。` | 闽南话 |
| `请用宁夏话表达。` | 宁夏话 |
| `请用山西话表达。` | 山西话 |
| `请用陕西话表达。` | 陕西话 |
| `请用山东话表达。` | 山东话 |
| `请用上海话表达。` | 上海话 |
| `请用四川话表达。` | 四川话 |
| `请用天津话表达。` | 天津话 |
| `请用云南话表达。` | 云南话 |

### 音量与语速

| 指令 | 效果 |
|------|------|
| `Please say a sentence as loudly as possible.` | 最大音量 |
| `Please say a sentence in a very soft voice.` | 极轻柔声 |
| `请用尽可能慢地语速说一句话。` | 最慢语速 |
| `请用尽可能快地语速说一句话。` | 最快语速 |

### 情绪控制

| 指令 | 情绪 |
|------|------|
| `请非常开心地说一句话。` | 开心 |
| `请非常伤心地说一句话。` | 伤心 |
| `请非常生气地说一句话。` | 生气 |

### 风格控制

| 指令 | 效果 |
|------|------|
| `我想体验一下小猪佩奇风格，可以吗？` | 小猪佩奇风格 |
| `你可以尝试用机器人的方式解答吗？` | 机器人风格 |

**完整指令格式（v3）：**

```
You are a helpful assistant. <指令内容><|endofprompt|>
```

---

## 10. Web UI 启动

启动 Gradio Web 界面，方便交互测试：

```bash
python webui.py --port 50000
```

**参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--port` | `8000` | Web 服务端口 |
| `--model_dir` | `pretrained_models/Fun-CosyVoice3-0.5B` | 模型目录（本地路径或 ModelScope ID） |
| `--device` | `None` | 推理设备（`cuda`/`cpu`），不指定时自动检测 |
| `--fp16` | `False` | 启用 fp16 混合精度推理（降低显存占用） |

Web UI 支持以下模式：
- 预训练音色 (SFT)
- 3s 极速复刻 (Zero-shot)
- 跨语种复刻 (Cross-lingual)
- 自然语言控制 (Instruct)
- 语音转换 (Voice Conversion)

---

## 11. FastAPI 服务部署

### 启动服务

```bash
cd runtime/python/fastapi
python server.py --port 50000 --model_dir pretrained_models/Fun-CosyVoice3-0.5B
```

**参数：** `--port` 默认 `50000`；`--model_dir` 默认 `iic/CosyVoice2-0.5B`（ModelScope 模型 ID，本地不存在时自动下载）。

### API 端点

| 端点 | 方法 | 参数 | 说明 |
|------|------|------|------|
| `/inference_sft` | GET/POST | `tts_text`, `spk_id` | SFT 合成 |
| `/inference_zero_shot` | GET/POST | `tts_text`, `prompt_text`, `prompt_wav`(文件) | 零样本克隆 |
| `/inference_cross_lingual` | GET/POST | `tts_text`, `prompt_wav`(文件) | 跨语种 |
| `/inference_instruct` | GET/POST | `tts_text`, `spk_id`, `instruct_text` | 指令控制 (v1) |
| `/inference_instruct2` | GET/POST | `tts_text`, `instruct_text`, `prompt_wav`(文件) | 指令控制 (v2/v3) |

所有端点返回 `StreamingResponse`，内容为原始 PCM 音频字节（int16）。

### 客户端调用

```bash
cd runtime/python/fastapi
python client.py --port 50000 --mode zero_shot
```

`--mode` 支持：`sft` | `zero_shot` | `cross_lingual` | `instruct`

### Python 客户端示例

```python
import requests

# 零样本合成
response = requests.post(
    'http://localhost:50000/inference_zero_shot',
    data={
        'tts_text': '你好，世界！',
        'prompt_text': '希望你以后能够做的比我还好呦。',
    },
    files={
        'prompt_wav': open('./asset/zero_shot_prompt.wav', 'rb'),
    },
    stream=True
)

# 收集音频数据
audio_data = b''
for chunk in response.iter_content(chunk_size=4096):
    audio_data += chunk

# 保存为 wav（需自行添加 wav 头或使用 soundfile）
import soundfile as sf
import numpy as np
import io

samples = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
sf.write('output.wav', samples, 22050)
```

---

## 12. 注意事项

1. **PyTorch 版本**：macOS Intel 不支持 PyTorch 2.3+，使用 2.2.2 替代。Linux + CUDA 环境可使用指定版本。

2. **CPU 推理性能**：无 GPU 时 RTF 约 6-7（生成 1 秒语音需 6-7 秒），建议使用 GPU 部署。

3. **`inference_instruct` vs `inference_instruct2`**：
   - `inference_instruct` 仅限 CosyVoice v1，使用预训练说话人
   - `inference_instruct2` 限 CosyVoice v2/v3，使用参考音频

4. **`speed` 参数**仅在非流式模式（`stream=False`）下有效，流式模式会触发断言错误。

5. **`prompt_wav` 限制**：时长不超过 30 秒，采样率不低于 16 kHz。

6. **`text_frontend`**：如需复现 CosyVoice2 官方 demo 效果，设置 `text_frontend=False`。

7. **`save_spkinfo()`**：`add_zero_shot_spk` 仅保存在内存中，需调用 `save_spkinfo()` 才会持久化到磁盘。

8. **所有推理方法都是生成器**：需要迭代获取结果，不要只调用 `next()` 一次。

9. **CosyVoice3 的 `prompt_text` 前缀**：v3 模型的 `prompt_text` / `instruct_text` 通常需要以 `You are a helpful assistant.` 开头，以 `<|endofprompt|>` 结尾。

10. **日语合成**：必须将日文转换为片假名再输入。
