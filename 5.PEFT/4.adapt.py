# ============================ #
#   Qwen3-1.7B QLoRA 训练+推理
#   版本：trl=0.22.2 / tfm=4.55.2
#   特性：4bit(NF4)+LoRA、省显存；推理dtype统一
#   每行中文注释
# ============================ #

# —— 1) 在 transformers 导入前禁用 TF/JAX，避免 tf_keras 等无关依赖的干扰 —— #
import os                                              # 操作系统接口（设置环境变量等）
os.environ["TRANSFORMERS_NO_TF"] = "1"                # 禁用 TensorFlow 相关路径
os.environ["TRANSFORMERS_NO_JAX"] = "1"               # 禁用 JAX 相关路径

# —— 2) 常规依赖 —— #
import json                                            # 读取本地 JSONL 数据
import torch                                           # PyTorch 主库
from datasets import Dataset                           # 轻量构造 HuggingFace 数据集
from transformers import (                             # Transformers 组件
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig                  # TRL 监督微调训练器与配置
from peft import LoraConfig, TaskType, PeftModel       # PEFT：LoRA 配置与推理时加载适配器

# —— 3) 可调参数（按你的机器/数据调整） —— #
MODEL_ID   = "Qwen3-1.7B"                              # 本地基座模型目录/名称
OUTPUT_DIR = "outputs/qwen3_qLora_adapt"               # 训练输出目录（保存 LoRA 适配器）
MAX_LEN    = 1024                                      # 分词/训练的最大序列长度
EPOCHS     = 10                                         # 训练轮次（示例值；真实训练可加大）
BATCH_SIZE = 4                                         # 每设备 batch（显存不够就调小）
GRAD_ACC   = 8                                         # 梯度累积步数（等效放大 batch）
LR         = 2e-4                                      # 学习率（LoRA 常用 1e-4~5e-4）
WARMUP     = 0.03                                      # warmup 比例（3%）
WEIGHT_DECAY = 0.1                                     # 权重衰减（L2 正则）
MAX_GRAD_NORM = 1.0                                    # 梯度裁剪上限
LOAD_4BIT  = True                                      # ★ 是否启用 4bit 量化（QLoRA 的关键）
USE_FLASH  = False                                     # 是否启用 flash-attn（未安装保持 False）
REPEAT_K   = 200                                       # 小样本复制倍数（仅演示，真实训练请设为1）
MAX_NEW_TOKENS = 500                                    # 推理阶段新生成 token 上限（按你要求）

# —— 4) 数据读取：优先 JSONL，否则回退内置样本 —— #
JSONL_PATH = "data/train.jsonl"                        # JSONL 数据路径（每行一个样本）
def load_training_samples():
    """优先从 JSONL 读取；若不存在，返回少量内置样本以跑通流程"""
    if os.path.isfile(JSONL_PATH):                     # 若文件存在
        rows = []                                      # 用于收集样本
        with open(JSONL_PATH, "r", encoding="utf-8") as f:  # 打开文件
            for line in f:                             # 逐行读取
                line = line.strip()                    # 去掉首尾空白
                if not line:                           # 空行跳过
                    continue
                obj = json.loads(line)                 # 解析 JSON
                # 期望字段：instruction / output
                if "instruction" in obj and "output" in obj:
                    rows.append({"instruction": obj["instruction"], "output": obj["output"]})
        if rows:                                       # 若读取到样本
            return rows                                # 返回样本列表
    # —— 回退样本（少量，仅演示；真实训练请替换为你的语料） —— #
    return [
        {"instruction": "把下面句子翻成英文：我爱自然语言处理。", "output": "I love natural language processing."},
        {"instruction": "用一句话解释注意力机制。",           "output": "根据上下文给不同词分配权重，从而聚焦关键信息。"},
        {"instruction": "将这段话压缩成一句话：近年来，大语言模型在对话、翻译和编程等任务上表现突出，但也带来了安全与偏见问题，需要负责任地使用。",
         "output": "大语言模型能力强但伴随安全与偏见风险，需要负责任地应用。"},
        {"instruction": "纠正语法并保持原意：He go to school every day.", "output": "He goes to school every day."},
    ]

# —— 5) 加载分词器（自回归模型常见：若无 pad_token 则用 eos_token 代替） —— #
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,                                           # 基座模型名称/路径
    trust_remote_code=True,                             # 允许模型自定义代码（Qwen 需要）
    use_fast=True                                       # 使用 fast tokenizer（更快）
)
if tokenizer.pad_token is None:                         # 若分词器未定义 pad_token
    tokenizer.pad_token = tokenizer.eos_token           # 用 eos 作为 pad，训练/推理更稳

# —— 6) 样本 -> 单列 "text"（用 chat_template 组织 system/user/assistant） —— #
def to_text_row(example: dict) -> dict:
    """
    输入：{"instruction": "...", "output": "..."}
    输出：{"text": 用 chat_template 组织的完整对话文本}
    训练目标：assistant 段
    """
    messages = [
        {"role": "system",    "content": "你是一个乐于助人的助手。"},
        {"role": "user",      "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(               # 用模型内的 chat 模板拼接
        messages,
        tokenize=False,                                 # 返回字符串
        add_generation_prompt=False                     # 不加“assistant:”起始（训练已含答案）
    )
    return {"text": text}                               # SFTTrainer 读取这列文本

# —— 7) 构造训练集（少量样本可复制以便看到 loss 下降） —— #
raw_samples = load_training_samples()                   # 读取训练样本
if len(raw_samples) < 64:                               # 若样本较少（仅演示）
    raw_samples = raw_samples * REPEAT_K                # 复制多份（真实训练请用大语料并设为1）
train_ds = Dataset.from_list(raw_samples).map(          # 构建 HF Dataset 并将字段映射到 text
    to_text_row, remove_columns=list(raw_samples[0].keys())
)

# —— 8) QLoRA 的 4bit 量化配置（bitsandbytes），与训练/推理需保持一致 —— #
bnb_cfg = None                                          # 默认不量化
if LOAD_4BIT:                                           # 若启用 4bit
    bnb_cfg = BitsAndBytesConfig(                       # 创建 4bit 配置
        load_in_4bit=True,                              # 将基座权重加载为 4bit
        bnb_4bit_compute_dtype=torch.bfloat16,          # 前向计算用 bfloat16（稳定高效）
        bnb_4bit_use_double_quant=True,                 # 双重量化（进一步省显存）
        bnb_4bit_quant_type="nf4",                      # NF4 量化（QLoRA 推荐）
    )

# —— 9) LoRA 配置（QLoRA = 4bit 基座 + LoRA 小头） —— #
lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,                       # 任务：自回归语言建模
    r=16,                                               # LoRA 秩（8/16/32 常用）
    lora_alpha=32,                                      # 放大系数
    lora_dropout=0.05,                                  # LoRA dropout（防过拟合）
    bias="none",                                        # 不训练原始 bias
    target_modules=[                                    # 目标模块（Qwen/LLaMA/Mistral 通用命名）
        "q_proj","k_proj","v_proj","o_proj",            # 注意力四投影
        "gate_proj","up_proj","down_proj"               # FFN 三投影（SwiGLU）
    ],
    # modules_to_save=None,                             # 可选：额外保存某层（如 "lm_head"）为可训练
)

# —— 10) SFT 训练配置（trl=0.22.2 的参数名） —— #
sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,                              # 输出目录
    num_train_epochs=EPOCHS,                            # 训练轮数
    per_device_train_batch_size=BATCH_SIZE,             # 每设备 batch
    gradient_accumulation_steps=GRAD_ACC,               # 梯度累积（等效放大 batch）
    learning_rate=LR,                                   # 学习率
    lr_scheduler_type="cosine",                         # 余弦退火调度
    warmup_ratio=WARMUP,                                # warmup 比例
    weight_decay=WEIGHT_DECAY,                          # 权重衰减
    max_grad_norm=MAX_GRAD_NORM,                        # 梯度裁剪上限
    logging_steps=10,                                   # 日志打印频率
    save_steps=200,                                     # 保存步频（真实训练可调大）
    save_total_limit=3,                                 # 最多保留的 checkpoint 数
    bf16=torch.cuda.is_available(),                     # GPU 上用 bfloat16（与 bnb compute 类型一致）
    gradient_checkpointing=True,                        # 开启检查点（进一步省显存）
    dataset_text_field="text",                          # 数据列名
    max_length=MAX_LEN,                                 # 本版本使用 max_length（而非 max_seq_length）
    packing=False if not USE_FLASH else True,           # 未装 flash-attn 建议关闭 packing
    optim="paged_adamw_32bit" if LOAD_4BIT else "adamw_torch",  # 4bit 推荐 paged 优化器
    model_init_kwargs={                                 # 透传给 from_pretrained 的参数
        "quantization_config": bnb_cfg,                 # 4bit 配置（None 表示不量化）
        "device_map": "auto",                           # 自动放置到可用设备
        # 若已安装 flash-attn 并想开启 padding-free，可加入：
        # "attn_implementation": "flash_attention_2",
    },
)

# —— 11) 构建 SFTTrainer：内部加载（可选 4bit 的）基座，并注入 LoRA —— #
trainer = SFTTrainer(
    model=MODEL_ID,                                     # 直接传模型名/路径，内部 from_pretrained
    args=sft_args,                                      # 训练参数
    train_dataset=train_ds,                             # 训练数据集（只有 text 列）
    processing_class=tokenizer,                         # v0.22.2 用 processing_class 传 tokenizer
    peft_config=lora_cfg,                               # ★ 使用 LoRA 配置（QLoRA 的“小头”）
)

# —— 12) 打印可训练参数占比（应 ~1% 左右），确认 QLoRA 生效 —— #
trainer.model.print_trainable_parameters()              # 便于确认适配器是否生效

# —— 13) 开始训练 —— #
trainer.train()                                         # 正式训练（小数据会很快）

# —— 14) 保存 LoRA 适配器与分词器 —— #
trainer.model.save_pretrained(OUTPUT_DIR)               # 保存 LoRA 适配器（小）
tokenizer.save_pretrained(OUTPUT_DIR)                   # 保存 tokenizer（部署/复现实用）

# ===================== 推理/验证（20 条 messages，输出 30 tokens） ===================== #

# —— 15) 载入“4bit 基座 + LoRA 适配器”，并统一 dtype，避免 Half/Float 冲突 —— #
base = AutoModelForCausalLM.from_pretrained(            # 加载基座（保持与训练一致的量化配置）
    MODEL_ID, trust_remote_code=True, device_map="auto",
    quantization_config=bnb_cfg,                        # 4bit 配置保持一致
    torch_dtype=torch.bfloat16                          # ★ 统一为 bfloat16（也可改为 float16）
)
model = PeftModel.from_pretrained(                      # 将 LoRA 适配器挂到基座上
    base, OUTPUT_DIR
).eval()                                                # 切到 eval 模式（推理更稳）
try:
    # ★ 确保 lm_head 与前向 dtype 一致（统一为 bfloat16），避免 F.linear dtype 冲突
    model.base_model.lm_head = model.base_model.lm_head.to(dtype=torch.bfloat16)
except AttributeError:
    pass                                                # 若结构不同（无该属性），忽略即可
model.config.torch_dtype = torch.bfloat16               # 记录当前 dtype 设定

# —— 16) 工具：收集可能的 EOS（兼容不同模板的停止符） —— #
def collect_eos_ids(tok):                               # 定义一个收集 eos id 的小工具
    cands = []                                          # 候选 id 列表
    for t in ["<|im_end|>", "<|endoftext|>"]:           # 常见结束符
        try:
            tid = tok.convert_tokens_to_ids(t)          # 文本转 id
            if tid is not None and tid != tok.unk_token_id:
                cands.append(tid)                       # 合法则加入
        except Exception:
            pass                                        # 某些模型没有这些特殊符号
    if tok.eos_token_id is not None:                    # 把通用 eos 也加入
        cands.append(tok.eos_token_id)
    uniq = []                                           # 去重并保持顺序
    [uniq.append(x) for x in cands if x not in uniq]
    return uniq or None                                 # 若为空则返回 None

eos_ids = collect_eos_ids(tokenizer)                    # 供 generate 提前停止使用

# —— 17) 构造“20 条 messages”的对话上下文（1 system + 18 历史 + 1 当前 user） —— #
messages = []                                           # 初始化消息列表
messages.append({"role": "system", "content": "你是一个乐于助人的助手。请直接给出最终答案，不要展示思考过程。"})  # 第1条：system
for i in range(1, 10):                                  # 构造 9 轮历史（共 18 条），示例占位
    messages.append({"role": "user", "content": f"上下文问题{i}：请用一个短句说明自注意力的用途。"})
    messages.append({"role": "assistant", "content": f"历史回答{i}：它用于根据上下文为词分配权重以聚焦关键信息。"})
messages.append({"role": "user", "content": "判断情感（正面或负面）：这个产品太让人失望了。"})  # 第20条：当前 user 提问

# —— 18) 用 chat_template 生成 prompt，并让模型进入“assistant 生成”模式 —— #
prompt = tokenizer.apply_chat_template(                 # 根据模型的对话模板拼接 prompt
    messages,
    tokenize=False,                                     # 返回字符串
    add_generation_prompt=True                          # 添加“assistant:” 起始以便生成
)

# —— 19) 分词并移动到模型所在设备 —— #
inputs = tokenizer([prompt], return_tensors="pt").to(model.device)  # 文本→token id→张量→放到设备

# —— 20) 可选：屏蔽“思维标记”（若它们恰为单 token；多 token 主要靠 system 约束） —— #
bad_tokens = ["<think>", "</think>", "<|current_state|>"]           # 可能不希望输出的标记
bad_ids = []                                                         # 收集可屏蔽的 token id
for t in bad_tokens:                                                 # 遍历每个标记
    ids = tokenizer.encode(t, add_special_tokens=False)              # 编码为 id 序列
    if len(ids) == 1:                                                # 仅当为“单 token”时才能屏蔽
        bad_ids.append(ids)                                          # 形式需为 [[id1],[id2],...]
bad_words_ids = bad_ids if bad_ids else None                         # 若都不是单 token，则不传该参数

# —— 21) 生成（选用 Beam Search 更稳；不要传 temperature/top_p/top_k 以免被忽略告警） —— #
with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):  # ★ autocast 统一 dtype
    gen = model.generate(
        **inputs,                                       # 输入张量（包含 prompt）
        max_new_tokens=MAX_NEW_TOKENS,                  # ★ 新生成 token 上限
        do_sample=False,                                # 关闭采样（可控性更强）
        num_beams=5,                                    # Beam Search = 5
        length_penalty=1.0,                             # 长度惩罚（中性）
        repetition_penalty=1.05,                        # 轻微防重复
        eos_token_id=eos_ids,                           # 在任一 EOS 处停止
        bad_words_ids=bad_words_ids                     # 尝试屏蔽“思维标记”
        # 注意：此模式下 temperature / top_p / top_k 无效，不要传以免告警
    )

# —— 22) 只解码“新增 token”（避免回显 prompt），并打印结果 —— #
new_tokens = gen[0, inputs["input_ids"].shape[1]:]      # 取从 prompt 之后的新内容
result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()  # 解码为可读文本
print(result)                                           # 打印模型最终回复
