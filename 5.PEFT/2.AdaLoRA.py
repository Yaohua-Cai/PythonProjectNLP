# ============================ #
#   Qwen3-1.7B AdaLoRA 训练+推理
#   版本：trl=0.22.2 / tfm=4.55.2
#   特性：自动计算 total_step，QLoRA 可选
# ============================ #

# —— 1) 在 transformers 导入前禁用 TF/JAX，避免 tf_keras 警告 —— #
import os                                              # 操作系统模块，用来设置环境变量
os.environ["TRANSFORMERS_NO_TF"] = "1"                # 告诉 transformers 不要探测/使用 TensorFlow
os.environ["TRANSFORMERS_NO_JAX"] = "1"               # 告诉 transformers 不要探测/使用 JAX

# —— 2) 常规依赖 —— #
import json                                            # 用于读取 JSONL 数据
import math                                            # 用于步数计算时的向上取整
import torch                                           # PyTorch 主库
from datasets import Dataset                           # 轻量构造内存数据集
from transformers import (                             # 从 transformers 导入所需模块
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from trl import SFTTrainer, SFTConfig                  # TRL 的监督微调 Trainer 与配置
from peft import AdaLoraConfig, TaskType, PeftModel    # 使用 AdaLoRA 配置 + 推理时加载适配器

# —— 3) 可调超参数（根据你的机器与需求调整） —— #
MODEL_ID   = "../../model/Qwen3-1.7B"                              # 基座模型的本地目录或模型名（与你一致）
OUTPUT_DIR = "outputs/qwen3_adalora"                   # 训练输出目录（保存 AdaLoRA 适配器）
MAX_LEN    = 1024                                      # 训练/分词时的最大序列长度
EPOCHS     = 500                                        # 训练轮数（你设为 30，这里保持）
BATCH_SIZE = 4                                         # 每设备的 batch size（显存不够就调小）
GRAD_ACC   = 8                                         # 梯度累积步数（等效增大 batch）
LR         = 2e-4                                      # 学习率（与 LoRA 相近范围）
WARMUP     = 0.03                                      # warmup 比例（3%）
WEIGHT_DECAY = 0.1                                     # 权重衰减（L2 正则）
MAX_GRAD_NORM = 1.0                                    # 梯度裁剪上限
LOAD_4BIT  = True                                      # 是否启用 4bit 量化（QLoRA）
USE_FLASH  = False                                     # 是否使用 flash-attn（未安装时保持 False）
REPEAT_K   = 200                                       # 小样本时重复次数（仅为演示，真实训练用大语料设为 1）
MAX_NEW_TOKENS = 2000                               # 推理阶段的最大新 token 数（按你要求）

# —— 4) 数据读取（优先 JSONL，否则回退到内置样例） —— #
JSONL_PATH = "data/train.jsonl"                        # 你的训练数据文件路径（每行一个 JSON）
def load_training_samples():
    """优先从 JSONL 读取数据；若不存在，则返回若干内置样本以跑通流程。"""
    if os.path.isfile(JSONL_PATH):                     # 若 JSONL 文件存在
        rows = []                                      # 用于收集样本
        with open(JSONL_PATH, "r", encoding="utf-8") as f:  # 打开文件
            for line in f:                             # 逐行读取
                line = line.strip()                    # 去掉首尾空白
                if not line:                           # 空行则跳过
                    continue
                obj = json.loads(line)                 # 解析 JSON 行
                # 期望字段为 instruction / output
                if "instruction" in obj and "output" in obj:
                    rows.append({"instruction": obj["instruction"], "output": obj["output"]})
        if len(rows) > 0:                              # 若成功读取到样本
            return rows                                # 返回样本列表
    # —— 回退：若没有 JSONL，返回若干内置样本（仅用于跑通流程）—— #
    return [
        {"instruction": "把下面句子翻成英文：我爱自然语言处理。", "output": "I love natural language processing."},
        {"instruction": "用一句话解释注意力机制。",           "output": "根据上下文给不同词分配权重，从而聚焦关键信息。"},
        {"instruction": "将这段话压缩成一句话：近年来，大语言模型在对话、翻译和编程等任务上表现突出，但也带来了安全与偏见问题，需要负责任地使用。",
         "output": "大语言模型能力强但伴随安全与偏见风险，需要负责任地应用。"},
        {"instruction": "判断情感（正面或负面）：这个产品太让人失望了。", "output": "负面"},
        {"instruction": "提取关键词：深度学习模型需要大量数据和计算资源来训练。", "output": "深度学习, 模型, 数据, 计算资源, 训练"},
        {"instruction": "把这句话改写得更礼貌：把报告立刻给我。", "output": "请尽快把报告发给我，谢谢。"},
        {"instruction": "纠正语法并保持原意：He go to school every day.", "output": "He goes to school every day."},
        {"instruction": "把这句话翻成中文：Transformers are powerful sequence models.", "output": "Transformer 是一种强大的序列模型。"},
        {"instruction": "为这段文字生成一个简短标题：人工智能正被用于医疗影像分析、客服自动化和交通优化等领域。", "output": "人工智能应用概览"},
        {"instruction": "用一句话解释 Transformer 的核心思想。", "output": "通过自注意力机制建模长程依赖，并行处理序列。"},
        {"instruction": "给出三条学习自注意力的建议。", "output": "从点积注意力公式入门；用小例子手算一遍；可视化权重理解关注点。"},
        {"instruction": "识别这句话的语言：Bonjour, comment ça va ?", "output": "法语"},
        {"instruction": "续写一句，使语义合理：春天到了，花园里", "output": "开满了五彩的花，空气里带着清新的泥土香。"},
        {"instruction": "将主动语态改为被动语态：The committee approved the proposal.", "output": "The proposal was approved by the committee."},
        {"instruction": "从句子中抽取时间和地点：我们将于2024年6月18日在北京召开发布会。", "output": "时间: 2024年6月18日; 地点: 北京"},
        {"instruction": "写一个匹配电子邮箱的正则表达式。", "output": "^[\\w\\.-]+@[\\w\\.-]+\\.[a-zA-Z]{2,}$"},
        {"instruction": "把下面三点合并为条目式摘要：1) 提升数据质量；2) 调整学习率；3) 增大批量大小。", "output": "- 提升数据质量\\n- 调整学习率\\n- 增大批量大小"},
        {"instruction": "给出单词 fast 的三个反义词。", "output": "slow, sluggish, leisurely"},
        {"instruction": "把这句话改写得更幽默：今天加班到很晚。", "output": "今天和公司谈了场持久的恋爱，直到深夜才分手。"},
        {"instruction": "用通俗的话解释 attention mask 的作用。", "output": "用来指定哪些位置可见、哪些位置要遮住，避免模型看见不该看的信息。"}
    ]

# —— 5) 加载分词器（自回归 LM 常用：pad_token = eos_token） —— #
tokenizer = AutoTokenizer.from_pretrained(              # 从基座目录/名称加载分词器
    MODEL_ID,                                           # 基座模型标识
    trust_remote_code=True,                             # 允许加载仓库自定义代码
    use_fast=True                                       # 使用更快的 fast tokenizer
)
if tokenizer.pad_token is None:                         # 若分词器未定义 pad_token
    tokenizer.pad_token = tokenizer.eos_token           # 将 pad_token 指向 eos，训练/推理更稳

# —— 6) 样本 → 训练文本（单列 "text"），使用 chat_template 组织对话 —— #
def to_text_row(example: dict) -> dict:
    """
    将 {instruction, output} 转换为可训练文本：
    用 chat_template 组织为 [system, user, assistant]，训练目标就是 assistant 段。
    """
    messages = [                                        # 构造标准消息列表
        {"role": "system",    "content": "你是一个乐于助人的助手。"},
        {"role": "user",      "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]},
    ]
    text = tokenizer.apply_chat_template(               # 使用模型内置聊天模板拼接文本
        messages,
        tokenize=False,                                 # 返回字符串（不直接分词）
        add_generation_prompt=False                     # 不添加“等待生成”的 assistant 前缀
    )
    return {"text": text}                               # 返回给 SFTTrainer 使用

# —— 7) 构建训练数据集（小样本时可重复以便看到 loss 下降） —— #
raw_samples = load_training_samples()                   # 读取样本（JSONL 或内置）
if len(raw_samples) <= 5:                               # 若样本太少（仅为演示）
    raw_samples = raw_samples * REPEAT_K                # 复制多份样本以拉长训练
train_ds = Dataset.from_list(raw_samples).map(          # 构造 HF Dataset 并映射到 "text"
    to_text_row, remove_columns=list(raw_samples[0].keys())
)

# —— 8) QLoRA：4bit 量化配置（未安装 bitsandbytes 则设 LOAD_4BIT=False） —— #
bnb_cfg = None                                          # 默认不量化
if LOAD_4BIT:                                           # 若开启 QLoRA（4bit）
    bnb_cfg = BitsAndBytesConfig(                       # 构造 4bit 配置
        load_in_4bit=True,                              # 将权重加载为 4bit
        bnb_4bit_compute_dtype=torch.bfloat16,          # 计算用 bfloat16（稳定高效）
        bnb_4bit_use_double_quant=True,                 # 双重量化进一步省显存
        bnb_4bit_quant_type="nf4",                      # NF4 量化类型（QLoRA 推荐）
    )

# —— 9) ★ 关键：计算 AdaLoRA 需要的 total_step（优化器更新总步数） —— #
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))     # 世界大小（多卡时由启动器注入；单卡=1）
steps_per_epoch = math.ceil(                            # 每个 epoch 的前向步数（按 batch 计）
    len(train_ds) / (BATCH_SIZE * WORLD_SIZE)           # 样本数 / (每设备 batch * 设备数)
)
opt_steps_per_epoch = math.ceil(                        # 每个 epoch 的 optimizer.step() 次数
    steps_per_epoch / GRAD_ACC                          # 梯度累积 GRAD_ACC 次前向 → 1 次更新
)
TOTAL_STEPS = opt_steps_per_epoch * EPOCHS              # ★ AdaLoRA 必须知道的总更新步数
print(f"[AdaLoRA] len(train_ds)={len(train_ds)}, WORLD_SIZE={WORLD_SIZE}, "
      f"steps/ep={steps_per_epoch}, opt_steps/ep={opt_steps_per_epoch}, total={TOTAL_STEPS}")

# —— 10) 为 AdaLoRA 设定 rank 调度时间表（按总步数的比例推导） —— #
TINIT   = max(10, int(0.05 * TOTAL_STEPS))             # 约 5% 步开始评估重要性（至少 10）
TFINAL  = max(TINIT + 1, int(0.60 * TOTAL_STEPS))      # 约 60% 步完成 rank 衰减
DELTA_T = max(1, int(0.01 * TOTAL_STEPS))              # 每隔 ~1% 步重估/重分配一次

# —— 11) AdaLoRA 配置（替代 LoRAConfig） —— #
adalora_cfg = AdaLoraConfig(                            # 构造 AdaLoRA 配置
    task_type=TaskType.CAUSAL_LM,                       # 任务：自回归语言建模
    # 自适应 rank 超参（可按需微调）
    init_r=12,                                          # 初期 rank（稍大便于探索）
    target_r=8,                                         # 收敛目标 rank（更省参）
    total_step=TOTAL_STEPS,                             # 必填：总优化步数（避免报错）
    tinit=TINIT,                                        # 何时开始评估重要性
    tfinal=TFINAL,                                      # 何时完成 rank 衰减
    deltaT=DELTA_T,                                     # 评估/重分配间隔
    beta1=0.85, beta2=0.85,                             # 重要性估计的 EMA 平滑系数
    # orth_reg_weight=0.0,                              # 可选：正交正则（不稳定时可设 0.1~0.5）
    # 与 LoRA 相同的公共超参
    lora_alpha=32,                                      # 放大系数
    lora_dropout=0.05,                                  # Dropout 抑制过拟合
    target_modules=[                                    # 目标模块（Qwen/LLaMA/Mistral 通用命名）
        "q_proj","k_proj","v_proj","o_proj",            # 注意力四投影
        "gate_proj","up_proj","down_proj"               # FFN 三投影（SwiGLU）
    ],
)

# —— 12) SFT 训练配置（trl=0.22.2 写法） —— #
sft_args = SFTConfig(                                   # 构造 SFT 配置
    output_dir=OUTPUT_DIR,                              # 输出目录
    num_train_epochs=EPOCHS,                            # 训练轮数
    per_device_train_batch_size=BATCH_SIZE,             # 每设备 batch
    gradient_accumulation_steps=GRAD_ACC,               # 梯度累积
    learning_rate=LR,                                   # 学习率
    lr_scheduler_type="cosine",                         # 余弦退火调度
    warmup_ratio=WARMUP,                                # warmup 比例
    weight_decay=WEIGHT_DECAY,                          # 权重衰减
    max_grad_norm=MAX_GRAD_NORM,                        # 梯度裁剪
    logging_steps=10,                                   # 日志步频
    save_steps=200,                                     # 保存步频（实际训练可调大）
    save_total_limit=3,                                 # 最多保留的检查点数
    bf16=torch.cuda.is_available(),                     # GPU 下使用 bf16（更稳更快）
    gradient_checkpointing=True,                        # 开启检查点以省显存
    dataset_text_field="text",                          # 告诉 SFT 哪一列是文本
    max_length=MAX_LEN,                                 # 本版本使用 max_length 参数
    packing=False if not USE_FLASH else True,           # 未装 flash-attn 前请保持 False
    optim="paged_adamw_32bit" if LOAD_4BIT else "adamw_torch",  # QLoRA 推荐 paged 优化器
    model_init_kwargs={                                 # 透传给 from_pretrained 的参数
        "quantization_config": bnb_cfg,                 # 4bit 配置（None 表示不量化）
        "device_map": "auto",                           # 自动放置到可用设备
        # 若已安装 flash-attn，且想开启 padding-free packing，可加：
        # "attn_implementation": "flash_attention_2",
    },
)

# —— 13) 构建 SFTTrainer：内部加载 4bit 基座并注入 AdaLoRA —— #
trainer = SFTTrainer(                                   # 创建监督微调 Trainer
    model=MODEL_ID,                                     # 直接给模型名/路径，内部会 from_pretrained
    args=sft_args,                                      # 训练配置
    train_dataset=train_ds,                             # 训练数据集（只有 text 一列）
    processing_class=tokenizer,                         # v0.22.2 用 processing_class 传 tokenizer
    peft_config=adalora_cfg,                            # ★ 关键：使用 AdaLoRA 配置
)

# —— 14) 打印可训练参数占比（约 1% 左右；AdaLoRA 会动态分配各层 rank） —— #
trainer.model.print_trainable_parameters()              # 便于确认适配器是否生效

# —— 15) 开始训练 —— #
trainer.train()                                         # 正式训练（小数据会很快）

# —— 16) 保存 AdaLoRA 适配器与分词器 —— #
trainer.model.save_pretrained(OUTPUT_DIR)               # 将 AdaLoRA 适配器权重保存到 OUTPUT_DIR
tokenizer.save_pretrained(OUTPUT_DIR)                   # 保存分词器（部署/复现实用）

# ===================== 推理 / 验证（构造 20 条 messages，输出 30 tokens） ===================== #

# —— 17) 载入“4bit 基座 + AdaLoRA 适配器” —— #
base = AutoModelForCausalLM.from_pretrained(            # 载入基座模型（保持与训练一致的量化配置）
    MODEL_ID, trust_remote_code=True, device_map="auto",
    quantization_config=bnb_cfg
)
model = PeftModel.from_pretrained(base, OUTPUT_DIR).eval()  # 将 AdaLoRA 挂载到基座上并设为 eval

# —— 18) 收集可能的 EOS（兼容不同模板的停止符） —— #
def collect_eos_ids(tok):                               # 定义工具函数收集多个可能的 EOS id
    cands = []                                          # 候选 id 列表
    for t in ["<|im_end|>", "<|endoftext|>"]:           # 常见特殊结束符
        try:
            tid = tok.convert_tokens_to_ids(t)          # 将特殊 token 文本转为 id
            if tid is not None and tid != tok.unk_token_id:
                cands.append(tid)                       # 合法则加入候选
        except Exception:
            pass                                        # 某些模型可能没有这些特殊符号
    if tok.eos_token_id is not None:                    # 把通用 eos_token_id 也加入
        cands.append(tok.eos_token_id)
    uniq = []                                           # 去重并保持顺序
    [uniq.append(x) for x in cands if x not in uniq]
    return uniq or None                                 # 若为空则返回 None

eos_ids = collect_eos_ids(tokenizer)                    # 在 generate() 时使用

# —— 19) 构造“20 条 messages”的对话上下文（1 system + 18 历史 + 1 当前 user） —— #
messages = []                                           # 初始化消息列表
messages.append({"role": "system", "content": "你是一个乐于助人的助手。请直接给出最终答案，不要展示思考过程。"})  # 第1条：system
# 构造 9 轮历史（共 18 条），占位示意；真实使用可替换成你的历史上下文
# for i in range(1, 10):                                  # i=1..9
#     messages.append({"role": "user", "content": f"上下文问题{i}：请用一个短句说明自注意力的用途。"})
#     messages.append({"role": "assistant", "content": f"历史回答{i}：它用于根据上下文为词分配权重以聚焦关键信息。"})
# 第20条：当前用户提问
messages.append({"role": "user", "content": "把这句话改写得更幽默：今天加班到很晚。"})

# —— 20) 用 chat_template 生成提示（让模型进入“生成助手回复”状态） —— #
prompt = tokenizer.apply_chat_template(                 # 根据模型的对话模板拼接 prompt
    messages,
    tokenize=False,                                     # 返回字符串
    add_generation_prompt=True                          # 添加“assistant:” 起始以便生成
)

# —— 21) 分词并移动到模型同一设备 —— #
inputs = tokenizer([prompt], return_tensors="pt").to(model.device)  # 文本→token id→张量→放到设备

# —— 22) 可选：屏蔽“思维标记”（若它们恰为单 token；多 token 主要靠 system 约束） —— #
bad_tokens = ["<think>", "</think>", "<|current_state|>"]           # 可能不希望输出的标记
bad_ids = []                                                         # 收集可屏蔽的 token id
for t in bad_tokens:                                                 # 遍历每个标记
    ids = tokenizer.encode(t, add_special_tokens=False)              # 编码为 id 序列
    if len(ids) == 1:                                                # 仅当为“单 token”时才能屏蔽
        bad_ids.append(ids)                                          # 形式需为 [[id1],[id2],...]
bad_words_ids = bad_ids if bad_ids else None                         # 若都不是单 token，则不传该参数

# —— 23) 生成：输出上限 30 个新 token（按你的要求），用 beam search 更稳 —— #
with torch.no_grad():                                               # 推理阶段关闭梯度
    gen = model.generate(
        **inputs,                                                   # 输入张量
        max_new_tokens=MAX_NEW_TOKENS,                              # ★ 输出上限 = 30
        do_sample=False,                                            # 关闭采样（可控性更强）
        num_beams=5,                                                # beam search = 5（可调）
        length_penalty=1.0,                                         # 长度惩罚（中性）
        repetition_penalty=1.05,                                    # 轻微防重复
        eos_token_id=eos_ids,                                       # 在任一 EOS 上停止
        bad_words_ids=bad_words_ids                                 # 尝试屏蔽“思维标记”
    )

# —— 24) 只解码“新增 token”（避免回显提示），并打印结果 —— #
new_tokens = gen[0, inputs["input_ids"].shape[1]:]                  # 从 prompt 末尾开始的新内容
result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()  # 解码为可读文本
print(result)                                                       # 打印模型最终回复
