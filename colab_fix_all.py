#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPT-SoVITS Colab 一键修复 + 全自动训练（Tesla T4 硬校验，可断点续跑）。

用法（Colab 新建 cell 粘贴运行，一次性跑完 修复→ASR→格式化→s2→s1→验证）:
    !wget -q -O /content/colab_fix_all.py https://raw.githubusercontent.com/cccbbot/voice-clone-colab/master/colab_fix_all.py && python /content/colab_fix_all.py

幂等：已完成阶段自动跳过，断线/失败后重跑本行即可继续。
"""
import os
import re
import subprocess
import sys
import threading
import time

WS = "/content/drive/MyDrive/GPT-SoVITS"
REPO = "/content/GPT-SoVITS"
PM = f"{REPO}/GPT_SoVITS/pretrained_models"
EXP_NAME = "tingshu_club"
S1_DIR = f"{REPO}/logs/{EXP_NAME}"
ASR_LIST_TXT = "/content/asr_list_path.txt"


def sh(cmd, timeout=3600, cwd=None):
    full = f"cd {cwd} && {cmd}" if cwd else cmd
    r = subprocess.run(["bash", "-lc", full], capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "") + (r.stderr or ""), r.returncode


def esh(cmd, timeout=3600, cwd=None):
    return sh(f"source /content/env_path.sh && {cmd}", timeout, cwd)


def banner(t):
    print("\n" + "#" * 70 + f"\n## {t}\n" + "#" * 70, flush=True)


def ok(msg=""):
    print(f"  [OK] {msg}", flush=True)


def warn(msg):
    print(f"  [WARN] {msg}", flush=True)


def fail(msg):
    print(f"  [FAIL] {msg}", flush=True)


# 心跳线程：每 120s 打印一次，防 Colab 空闲断连
def heartbeat():
    n = 0
    while True:
        time.sleep(120)
        n += 1
        print(f"\n  [心跳] 运行中 {n*2} 分钟...", flush=True)


threading.Thread(target=heartbeat, daemon=True).start()

# ============================================================
banner("0) GPU 硬校验：必须 Tesla T4")
out, rc = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null")
print("  nvidia-smi:", out.strip() or "(无输出)", flush=True)
if "Tesla T4" not in out:
    print("  ❌ 不是 Tesla T4！请：菜单 → 代码执行程序 → 更改运行时类型 → T4 GPU → 重新连接后重跑本行", flush=True)
    sys.exit(1)
ok("GPU = Tesla T4")

# ============================================================
banner("1) 代码仓库 → 最新 main")
out, rc = sh("cd " + REPO + " && git log -1 --oneline 2>/dev/null")
print("  更新前:", out.strip() or "(无 repo)", flush=True)
out, rc = sh(
    "cd " + REPO + " && git fetch origin 2>&1 | tail -1; "
    "git checkout -- . 2>/dev/null; git reset --hard origin/master 2>&1 | tail -1; git log -1 --oneline"
)
line = out.strip().splitlines()[-1] if out.strip() else "FAIL"
print("  更新后:", line, flush=True)
ok("repo = " + line) if rc == 0 else fail("repo 更新失败: " + out[-300:])

# ============================================================
banner("2) env_path.sh / PYTHONPATH")
if os.path.exists("/content/env_path.sh"):
    content = open("/content/env_path.sh").read()
    if "PYTHONPATH=/content/GPT-SoVITS" not in content:
        with open("/content/env_path.sh", "a") as f:
            f.write("\nexport PYTHONPATH=/content/GPT-SoVITS\n")
        warn("PYTHONPATH 已追加")
    ok("env_path.sh 就绪（含 PYTHONPATH）")
else:
    fail("env_path.sh 不存在 — 先跑第 2 步 setup.sh")
    sys.exit(1)

# ============================================================
banner("3) Drive 软链")
for d in ["logs", "output", "SoVITS_weights", "GPT_weights",
          "SoVITS_weights_v2Pro", "GPT_weights_v2Pro"]:
    p = f"{REPO}/{d}"
    if not os.path.islink(p):
        os.makedirs(f"{WS}/{d}", exist_ok=True)
        sh(f"rm -rf {p} && ln -s {WS}/{d} {p}")
        warn(f"{d} 软链已修复")
    else:
        ok(f"{d} 软链正常")
os.makedirs(f"{WS}/pretrained_models", exist_ok=True)
if not os.path.islink(PM):
    sh(f"rm -rf {PM} && ln -s {WS}/pretrained_models {PM}")
    warn("pretrained_models 软链已修复")
else:
    ok("pretrained_models 软链正常")

# ============================================================
banner("4) 预训练模型（缺则下载，约 2GB）")
if os.path.exists(f"{PM}/v2Pro/s2Gv2Pro.pth"):
    ok("模型已存在，跳过下载")
else:
    print("  下载 pretrained_models.zip（HF 主源，失败自动换 ModelScope）...", flush=True)
    rc, out = sh("cd /content && wget -q --tries=3 -O pretrained_models.zip "
                 "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/pretrained_models.zip")
    if rc != 0:
        print("  HF 失败，改用 ModelScope", flush=True)
        rc, out = sh("cd /content && wget -q --tries=3 -O pretrained_models.zip "
                     "https://www.modelscope.cn/models/XXXXRT/GPT-SoVITS-Pretrained/resolve/master/pretrained_models.zip")
    if rc == 0:
        print("  解压（经软链落 Drive）...", flush=True)
        sh("unzip -q -o /content/pretrained_models.zip -d " + REPO + " && rm -f /content/pretrained_models.zip", 900)
    else:
        fail("模型下载失败：" + out[-200:])
        sys.exit(1)
ok("s2Gv2Pro.pth 等模型就绪" if os.path.exists(f"{PM}/v2Pro/s2Gv2Pro.pth") else "模型仍缺失！")

if not os.path.exists(f"{REPO}/GPT_SoVITS/text/G2PWModel"):
    print("  下载 G2PWModel.zip...", flush=True)
    rc, out = sh("cd /content && wget -q --tries=3 -O G2PWModel.zip "
                 "https://huggingface.co/XXXXRT/GPT-SoVITS-Pretrained/resolve/main/G2PWModel.zip && "
                 "unzip -q -o G2PWModel.zip -d " + REPO + "/GPT_SoVITS/text && rm -f /content/G2PWModel.zip")
    ok("G2PWModel 就绪") if rc == 0 else fail("G2PWModel 下载失败")
else:
    ok("G2PWModel 已存在")

# ============================================================
banner("5) torchcodec（对齐 install.sh，可选）")
_, rc = esh("python -c 'import torchcodec' 2>/dev/null")
if rc != 0:
    print("  安装 torchcodec...", flush=True)
    esh("pip install --no-deps torchcodec --index-url https://download.pytorch.org/whl/cu126 2>&1 | tail -2", 600)
    _, rc = esh("python -c 'import torchcodec' 2>/dev/null")
    ok("torchcodec 就绪") if rc == 0 else warn("torchcodec 未装上（非必需，先继续）")
else:
    ok("torchcodec 已存在")

# ============================================================
banner("6) Step2 ASR 标注（342 切片，large-v3 中文，约 10~20min）")
SLICE_OUT = f"{WS}/output/slicer_opt"
ASR_OUT = f"{WS}/output/asr_opt"
os.makedirs(ASR_OUT, exist_ok=True)
lists = [f for f in os.listdir(ASR_OUT) if f.endswith(".list")] if os.path.isdir(ASR_OUT) else []
if lists:
    list_path = os.path.join(ASR_OUT, lists[0])
    ok(f"ASR .list 已存在: {lists[0]}，跳过")
else:
    out, rc = esh(
        f"export PYTHONPATH={REPO} && cd {REPO} && "
        f"python tools/asr/fasterwhisper_asr.py -i {SLICE_OUT} -o {ASR_OUT} -s large-v3 -l zh -p int8 2>&1 | tail -8",
        3600,
    )
    print(out[-1200:], flush=True)
    lists = [f for f in os.listdir(ASR_OUT) if f.endswith(".list")] if os.path.isdir(ASR_OUT) else []
    if lists:
        list_path = os.path.join(ASR_OUT, lists[0])
        ok(f"ASR 完成: {lists[0]}")
    else:
        fail("ASR 未生成 .list，中止（把上面日志贴回）")
        sys.exit(1)

with open(ASR_LIST_TXT, "w") as f:
    f.write(list_path)
ok("list_path 已记录: " + list_path)

# ============================================================
banner("7) Step3 格式化（v2Pro 全量：文本→HuBERT→声纹→语义 + 分片合并）")
os.makedirs(S1_DIR, exist_ok=True)


def missing(name):
    return not os.path.exists(f"{S1_DIR}/{name}")


envs = " ".join(f'{k}="{v}"' for k, v in {
    "inp_text": list_path,
    "inp_wav_dir": SLICE_OUT,
    "exp_name": EXP_NAME,
    "opt_dir": S1_DIR,
    "bert_pretrained_dir": f"{PM}/chinese-roberta-wwm-ext-large",
    "cnhubert_base_dir": f"{PM}/chinese-hubert-base",
    "pretrained_s2G": f"{PM}/v2Pro/s2Gv2Pro.pth",
    "s2config_path": f"{REPO}/GPT_SoVITS/configs/s2v2Pro.json",
    "sv_path": f"{PM}/sv/pretrained_eres2netv2w24s4ep4.ckpt",
    "i_part": "0", "all_parts": "1", "is_half": "True",
}.items())

steps = []
if missing("2-name2text.txt") or missing("3-bert"):
    steps.append(("1) 文本分词", "1-get-text.py"))
if missing("4-cnhubert") or missing("5-wav32k"):
    steps.append(("2) HuBERT 特征", "2-get-hubert-wav32k.py"))
if missing("7-sv_cn"):
    steps.append(("3) 声纹特征", "2-get-sv.py"))
if missing("6-name2semantic.tsv"):
    steps.append(("4) 语义 token", "3-get-semantic.py"))

for name, script in steps:
    print(f"  === {name} ===", flush=True)
    out, rc = esh(
        f"export PYTHONPATH={REPO} && cd {REPO} && "
        f"env {envs} python -s GPT_SoVITS/prepare_datasets/{script} 2>&1 | tail -8",
        3600,
    )
    print(out[-900:], flush=True)
    if rc != 0:
        fail(f"{name} 失败，中止")
        sys.exit(1)

# 合并文本分片
if missing("2-name2text.txt"):
    parts = sorted(__import__("glob").glob(f"{S1_DIR}/2-name2text-*.txt"))
    if parts:
        opt = []
        for p in parts:
            with open(p, encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(p)
        with open(f"{S1_DIR}/2-name2text.txt", "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ok(f"合并 2-name2text.txt：{len(opt)} 行")
    else:
        fail("没有 2-name2text-*.txt 分片"); sys.exit(1)

# 合并语义分片
if missing("6-name2semantic.tsv"):
    parts = sorted(__import__("glob").glob(f"{S1_DIR}/6-name2semantic-*.tsv"))
    if parts:
        opt = ["item_name\tsemantic_audio"]
        for p in parts:
            with open(p, encoding="utf8") as f:
                opt += f.read().strip("\n").split("\n")
            os.remove(p)
        with open(f"{S1_DIR}/6-name2semantic.tsv", "w", encoding="utf8") as f:
            f.write("\n".join(opt) + "\n")
        ok(f"合并 6-name2semantic.tsv：{len(opt)-1} 行")
    else:
        fail("没有 6-name2semantic-*.tsv 分片"); sys.exit(1)

print("  == 数据集最终检查 ==", flush=True)
all_ok = True
for f in ["2-name2text.txt", "4-cnhubert", "5-wav32k", "7-sv_cn", "6-name2semantic.tsv"]:
    e = os.path.exists(f"{S1_DIR}/{f}")
    all_ok &= e
    print(f"    {'OK ' if e else 'MISS'} {f}", flush=True)
sz = os.path.getsize(f"{S1_DIR}/6-name2semantic.tsv") if os.path.exists(f"{S1_DIR}/6-name2semantic.tsv") else 0
if not all_ok or sz < 31:
    fail("数据集不完整，中止"); sys.exit(1)
ok("数据集就绪（语义文件 " + str(sz) + "B）")

# ============================================================
banner("8) Step4 s2 训练（SoVITS 声学模型，10 epoch 约 30~60min）")
import json
import yaml

VERSION = "v2Pro"
TOTAL_EPOCH = 10
BATCH_SIZE = 4
TMP = f"{REPO}/tmp"
os.makedirs(TMP, exist_ok=True)

sovits_dir = f"{WS}/SoVITS_weights_v2Pro"
if os.path.isdir(sovits_dir) and any(f.endswith(".pth") for f in os.listdir(sovits_dir)):
    ok("SoVITS 权重已存在，跳过 s2")
else:
    with open(f"{REPO}/GPT_SoVITS/configs/s2v2Pro.json") as f:
        data = json.load(f)
    os.makedirs(f"{S1_DIR}/logs_s2_{VERSION}", exist_ok=True)
    data["train"]["batch_size"] = BATCH_SIZE
    data["train"]["epochs"] = TOTAL_EPOCH
    data["train"]["text_low_lr_rate"] = 1.0
    data["train"]["pretrained_s2G"] = f"GPT_SoVITS/pretrained_models/v2Pro/s2G{VERSION}.pth"
    data["train"]["pretrained_s2D"] = f"GPT_SoVITS/pretrained_models/v2Pro/s2D{VERSION}.pth"
    data["train"]["if_save_latest"] = True
    data["train"]["if_save_every_weights"] = False
    data["train"]["save_every_epoch"] = 5
    data["train"]["gpu_numbers"] = "0"
    data["train"]["grad_ckpt"] = False
    data["train"]["lora_rank"] = 32
    data["model"]["version"] = VERSION
    data["data"]["exp_dir"] = data["s2_ckpt_dir"] = S1_DIR
    data["save_weight_dir"] = "SoVITS_weights_v2Pro"
    data["name"] = EXP_NAME
    data["version"] = VERSION
    with open(f"{TMP}/tmp_s2.json", "w") as f:
        json.dump(data, f)
    out, rc = esh(
        f"export PYTHONPATH={REPO} && cd {REPO} && "
        f"python -s GPT_SoVITS/s2_train.py --config {TMP}/tmp_s2.json 2>&1 | tail -12",
        10800,
    )
    print(out[-1500:], flush=True)
    if rc != 0:
        fail("s2 训练失败（继续尝试 s1，最后统一看权重）")
    else:
        ok("s2 训练完成")

# ============================================================
banner("9) Step4 s1 训练（GPT 语义模型，10 epoch 约 30~60min）")
gpt_dir = f"{WS}/GPT_weights_v2Pro"
if os.path.isdir(gpt_dir) and any(f.endswith(".ckpt") for f in os.listdir(gpt_dir)):
    ok("GPT 权重已存在，跳过 s1")
else:
    with open(f"{REPO}/GPT_SoVITS/configs/s1longer-v2.yaml") as f:
        data = yaml.safe_load(f)
    os.makedirs(f"{S1_DIR}/logs_s1_{VERSION}", exist_ok=True)
    data["train"]["batch_size"] = BATCH_SIZE
    data["train"]["epochs"] = TOTAL_EPOCH
    data["train"]["save_every_n_epoch"] = 1
    data["train"]["if_save_every_weights"] = True
    data["train"]["if_save_latest"] = True
    data["train"]["half_weights_save_dir"] = "GPT_weights_v2Pro"
    data["train"]["exp_name"] = EXP_NAME
    data["pretrained_s1"] = "GPT_SoVITS/pretrained_models/s1v3.ckpt"
    data["train_semantic_path"] = f"{S1_DIR}/6-name2semantic.tsv"
    data["train_phoneme_path"] = f"{S1_DIR}/2-name2text.txt"
    data["output_dir"] = f"{S1_DIR}/logs_s1_{VERSION}"
    with open(f"{TMP}/tmp_s1.yaml", "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    out, rc = esh(
        f"export PYTHONPATH={REPO} && cd {REPO} && "
        f"python -s GPT_SoVITS/s1_train.py --config_file {TMP}/tmp_s1.yaml 2>&1 | tail -12",
        10800,
    )
    print(out[-1500:], flush=True)
    if rc != 0:
        fail("s1 训练失败")
    else:
        ok("s1 训练完成")

# ============================================================
banner("10) 最终验证")
for name, d in [("SoVITS_weights_v2Pro", sovits_dir), ("GPT_weights_v2Pro", gpt_dir)]:
    files = sorted(os.listdir(d)) if os.path.isdir(d) else []
    print(f"  {name}: {files}", flush=True)
if os.path.isdir(sovits_dir) and any(f.endswith(".pth") for f in os.listdir(sovits_dir)) and \
   os.path.isdir(gpt_dir) and any(f.endswith(".ckpt") for f in os.listdir(gpt_dir)):
    print("\n  ✅✅ 训练全部完成！权重在 Drive（重启不丢）。", flush=True)
    print("     下一步：跑 notebook 第 6 步 API 合成，或 WebUI 1C 推理。", flush=True)
else:
    print("\n  ⚠️ 有阶段未完成，把上面输出贴回给 Hermes。", flush=True)
print("\n=== 全流程结束 ===", flush=True)
