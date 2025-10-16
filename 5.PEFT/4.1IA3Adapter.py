# ============================ #
#   Qwen3-1.7B IA3 Adapter 微调
#   版本：trl=0.22.2 / tfm=4.55.2
#   可选4bit量化（QLoRA思路）
# ============================ #

# —— 1) 在 transformers 导入前禁用 TF/JAX，避免 tf_keras 等提示 —— #
import os                                              # 操作系统接口（设置环境变量）
os.environ["TRANSFORMERS_NO_TF"] = "1"                # 禁用 TensorFlow 路径（只用 PyTorch）
os.environ["TRANSFORMERS_NO_JAX"] = "1"               # 禁用 JAX 路径

# —— 2) 常规依赖 —— #
import json                                            # 读取本地 JSONL 数据
import torch                                           # PyTorch 主库
from datasets import Dataset                           # 轻量构造 HuggingFace 数据集
from transformers import (                             # Transformers 组件
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig                  # TRL 的 SFT 训练器与配置
from peft import IA3Config, TaskType, PeftModel        # ★ PEFT 的 IA3（Adapter）配置与推理挂载

# —— 3) 可调参数（根据你的机器/需求调整） —— #
MODEL_ID   = "../../model/Qwen3-1.7B"                  # 基座模型目录/名称（本地可用）
OUTPUT_DIR = "outputs/qwen3_ia3_adapter"               # 训练输出目录（保存 IA3 适配器）
MAX_LEN    = 1024                                      # 分词/训练的最大序列长度
EPOCHS     = 10                                        # 训练轮次（IA3 参数很少，一般无需太多）
BATCH_SIZE = 4                                         # 每设备 batch（显存不够就调小）
GRAD_ACC   = 8                                         # 梯度累积步数（等效放大 batch）
LR         = 1e-3                                      # 学习率（IA3 常用 5e-4~3e-3，可略大于 LoRA）
WARMUP     = 0.03                                      # warmup 比例（3%）
WEIGHT_DECAY = 0.0                                     # 权重衰减（IA3 一般可设 0）
MAX_GRAD_NORM = 1.0                                    # 梯度裁剪上限
LOAD_4BIT  = True                                      # 是否启用4bit量化（基座冻结+更省显存）
USE_FLASH  = False                                     # 是否启用 flash-attn（未安装保持 False）
REPEAT_K   = 200                                       # 小样本时复制倍数（仅演示，真实训练用大语料设为1）
MAX_NEW_TOKENS = 30                                    # 推理阶段新生成 token 上限

# —— 4) 数据读取：优先 JSONL，否则回退内置样本 —— #
JSONL_PATH = "data/train.jsonl"                        # 你的训练数据路径
def load_training_samples():
    """优先从 JSONL 读取；没有则回退到少量样本（仅用于跑通流程）"""
    if os.path.isfile(JSONL_PATH):                     # 若文件存在
        rows = []                                      # 存放样本
        with open(JSONL_PATH, "r", encoding="utf-8") as f:  # 打开文件
            for line in f:                             # 逐行读取
                line = line.strip()                    # 去空白
                if not line:                           # 空行跳过
                    continue
                obj = json.loads(line)                 # 解析 JSON
                if "instruction" in obj and "output" in obj:  # 检查键
                    rows.append({"instruction": obj["instruction"], "output": obj["output"]})
        if rows:                                       # 若读到内容
            return rows                                # 返回样本
    # —— 回退样本（仅演示，真实训练请换成你的大语料） —— #
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
if len(raw_samples) < 64:                               # 若样本很少（仅演示）
    raw_samples = raw_samples * REPEAT_K                # 复制多份（真实训练请换大语料并设为1）
train_ds = Dataset.from_list(raw_samples).map(          # 构建 HF Dataset 并将字段映射到 text
    to_text_row, remove_columns=list(raw_samples[0].keys())
)

# —— 8) （可选）4bit 量化配置（QLoRA 思路：基座冻结 + 4bit 省显存） —— #
bnb_cfg = None                                          # 默认不量化
if LOAD_4BIT:                                           # 若选择开启 4bit
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,                              # 将基座权重加载为 4bit
        bnb_4bit_compute_dtype=torch.bfloat16,          # 计算用 bfloat16（更稳更快）
        bnb_4bit_use_double_quant=True,                 # 双重量化进一步省显存
        bnb_4bit_quant_type="nf4",                      # NF4 量化类型（QLoRA 推荐）
    )

# —— 9) IA3（Adapter）配置 —— #
#    思路：在目标模块上学习“乘性缩放向量”（不改原权重），极省参、切任务快。
ia3_cfg = IA3Config(
    task_type=TaskType.CAUSAL_LM,                       # 任务：自回归语言建模
    target_modules=[                                    # 目标模块命名（Qwen/LLaMA/Mistral 通用）
        "q_proj","k_proj","v_proj","o_proj",            # 注意力四投影（常见命名）
        "gate_proj","up_proj","down_proj"               # FFN 三投影（SwiGLU）
    ],
    feedforward_modules=None,                           # 也可指定仅对 FFN 生效；None=按 target_modules
    # modules_to_save=None,                             # 可选：额外保留某些层（如 "lm_head"）为可训练
)

# —— 10) SFT 训练配置（trl=0.22.2 的参数名） —— #
sft_args = SFTConfig(
    output_dir=OUTPUT_DIR,                              # 输出目录
    num_train_epochs=EPOCHS,                            # 训练轮次
    per_device_train_batch_size=BATCH_SIZE,             # 每设备 batch
    gradient_accumulation_steps=GRAD_ACC,               # 梯度累积（等效放大 batch）
    learning_rate=LR,                                   # 学习率（IA3 可略大）
    lr_scheduler_type="cosine",                         # 余弦退火
    warmup_ratio=WARMUP,                                # warmup 比例
    weight_decay=WEIGHT_DECAY,                          # 权重衰减
    max_grad_norm=MAX_GRAD_NORM,                        # 梯度裁剪
    logging_steps=10,                                   # 日志打印频率
    save_steps=200,                                     # 保存步频（真实训练可调大）
    save_total_limit=3,                                 # 最多保留 checkpoint 数
    bf16=torch.cuda.is_available(),                     # GPU 上用 bfloat16
    gradient_checkpointing=False,                       # IA3 参数很少，通常不必开
    dataset_text_field="text",                          # 数据列名
    max_length=MAX_LEN,                                 # 本版本使用 max_length
    packing=False if not USE_FLASH else True,           # 未装 flash-attn 建议 False
    optim="paged_adamw_32bit" if LOAD_4BIT else "adamw_torch",  # 4bit 时 paged 优化器更省资源
    model_init_kwargs={                                 # 透传给 from_pretrained 的参数
        "quantization_config": bnb_cfg,                 # 若启 4bit 则传入；否则为 None
        "device_map": "auto",                           # 自动将模型放置到可用设备
        # 若已安装 flash-attn 并想开启 padding-free，可加入：
        # "attn_implementation": "flash_attention_2",
    },
)

# —— 11) 构建 SFTTrainer：内部加载（可选 4bit 的）基座，并注入 IA3（Adapter） —— #
trainer = SFTTrainer(
    model=MODEL_ID,                                     # 直接传模型名/路径，内部 from_pretrained
    args=sft_args,                                      # 训练参数
    train_dataset=train_ds,                             # 训练数据集（只有 text 列）
    processing_class=tokenizer,                         # v0.22.2 用 processing_class 传 tokenizer
    peft_config=ia3_cfg,                                # ★ 使用 IA3（Adapter）配置
)

# —— 12) 打印可训练参数占比（应极小，仅为 IA3 的通道缩放参数） —— #
trainer.model.print_trainable_parameters()              # 验证只训练 IA3 参数

# —— 13) 开始训练 —— #
trainer.train()                                         # 进行监督微调（只更新 IA3 参数）

# —— 14) 保存 IA3 适配器与分词器 —— #
trainer.model.save_pretrained(OUTPUT_DIR)               # 保存 IA3 适配器（小）
tokenizer.save_pretrained(OUTPUT_DIR)                   # 保存 tokenizer（部署/复现实用）

# ===================== 推理/验证（messages + 30 tokens） ===================== #

# —— 15) 载入“基座 + IA3 适配器”进行推理 —— #
base = AutoModelForCausalLM.from_pretrained(            # 加载基座（与训练保持一致的量化配置）
    MODEL_ID, trust_remote_code=True, device_map="auto",
    quantization_config=bnb_cfg                         # 若训练时用了 4bit，这里也保持一致
)
model = PeftModel.from_pretrained(                      # 将 IA3 适配器挂到基座上
    base, OUTPUT_DIR
).eval()                                                # 切到 eval 模式（推理更稳）

# —— 16) 构造对话消息（与训练同风格，使用 chat_template） —— #
messages = [
    {"role": "system", "content": "你是一个乐于助人的助手。请直接给出最终答案，不要展示思考过程。"},
    {"role": "user",   "content": "把下面句子翻成英文：我爱自然语言处理。"},
]

# —— 17) 用 chat_template 组装 prompt，并让模型进入“assistant 生成”模式 —— #
prompt_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,                                     # 返回字符串
    add_generation_prompt=True                          # 添加“assistant:”起始以便生成
)

# —— 18) 将文本分词成张量并移动到模型所在设备 —— #
inputs = tokenizer([prompt_text], return_tensors="pt").to(model.device)

# —— 19) 收集可能的 EOS（兼容 <|im_end|> / <|endoftext|> / eos_token_id） —— #
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

# —— 20) 生成参数：输出 30 个新 token，用 Beam Search 保持可控 —— #
with torch.no_grad():                                   # 推理不需要梯度
    outputs = model.generate(
        **inputs,                                       # 输入张量（包含 prompt）
        max_new_tokens=MAX_NEW_TOKENS,                  # ★ 新生成 token 上限（30）
        do_sample=False,                                # 关闭采样（更稳定可控）
        num_beams=5,                                    # Beam Search = 5
        length_penalty=1.0,                             # 长度惩罚（中性）
        repetition_penalty=1.05,                        # 轻微防重复
        eos_token_id=eos_ids                            # 在任一 EOS 处停止
    )

# —— 21) 只解码“新增 token”（避免回显 prompt），并打印结果 —— #
new_tokens = outputs[0, inputs["input_ids"].shape[1]:]  # 取从 prompt 之后的新内容
result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()  # 解码为文本
print(result)                                           # 打印最终答案
