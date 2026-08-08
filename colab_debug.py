#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GPT-SoVITS Colab 一键诊断 + 自动修复（Tesla T4 硬性校验）。

用法（Colab 新建 cell 粘贴运行）:
    !wget -q -O /content/colab_debug.py https://raw.githubusercontent.com/cccbbot/voice-clone-colab/master/colab_debug.py && python /content/colab_debug.py

自动执行：
    1. GPU 必须是 Tesla T4（不是则明确提示切换运行时）
    2. 代码仓库 git reset --hard 到最新 main（修复旧快照 import torchaudio 问题）
    3. env_path.sh 补 PYTHONPATH
    4. 检查 Drive 软链 / 依赖 / 预训练模型 / 数据集现状
输出结构化报告，跑完贴回给 Hermes 继续下一步。
"""
import os
import subprocess

OK, WARN, FAIL = "OK", "WARN", "FAIL"
results = []


def sh(cmd, timeout=300):
    r = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "") + (r.stderr or ""), r.returncode


def esh(cmd, timeout=300):
    return sh("source /content/env_path.sh && " + cmd, timeout)


def sec(title):
    print("\n" + "=" * 60 + "\n" + title + "\n" + "=" * 60)


def report(name, status, detail=""):
    icon = {"OK": "\u2705", "WARN": "\u26a0\ufe0f", "FAIL": "\u274c"}[status]
    print(f"  {icon} [{status}] {name}" + (f" — {detail}" if detail else ""))
    results.append((name, status, detail))


sec("0) 基础环境")
out, rc = sh("python3 --version 2>/dev/null")
report("Python(脚本自身)", "OK" if rc == 0 else "FAIL", out.strip())

sec("1) GPU 检查（必须 Tesla T4）")
out, rc = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null")
print("  nvidia-smi:", out.strip() or "(无输出)")
if "Tesla T4" in out:
    report("GPU", "OK", out.strip().splitlines()[0].strip())
elif "Tesla" in out:
    report("GPU", "WARN", f"不是 T4: {out.strip()[:60]}，菜单 代码执行程序→更改运行时类型→T4 GPU，然后重新连接")
else:
    report("GPU", "FAIL", "没有 NVIDIA GPU（CPU 运行时），必须切到 T4 GPU")

sec("2) 代码仓库 → 更新到最新 main")
out, _ = sh("cd /content/GPT-SoVITS 2>/dev/null && git log -1 --oneline 2>/dev/null")
print("  更新前:", out.strip() or "(无 repo)")
out, rc = sh(
    "cd /content/GPT-SoVITS && git fetch origin 2>&1 | tail -1; "
    "git checkout -- . 2>/dev/null; git reset --hard origin/master 2>&1 | tail -1; "
    "git log -1 --oneline"
)
print("  更新后:", out.strip().splitlines()[-1] if out.strip() else "FAIL")
report("Repo", "OK" if rc == 0 else "FAIL", out.strip().splitlines()[-1] if out.strip() else "")

sec("3) env_path.sh / PYTHONPATH")
if os.path.exists("/content/env_path.sh"):
    content = open("/content/env_path.sh").read().strip()
    print("  内容:", content)
    if "PYTHONPATH=/content/GPT-SoVITS" in content:
        report("PYTHONPATH", "OK")
    else:
        sh('echo "export PYTHONPATH=/content/GPT-SoVITS" >> /content/env_path.sh')
        report("PYTHONPATH", "WARN", "缺失，已自动追加到 env_path.sh")
else:
    report("env_path.sh", "FAIL", "不存在 —— 先跑第 2 步 setup.sh")

sec("4) Drive 软链")
out, _ = sh("ls -la /content/GPT-SoVITS/ 2>/dev/null | grep -E 'logs|output|SoVITS_weights|GPT_weights|pretrained'")
print(out.strip() or "(无)")
for d in ["logs", "output", "SoVITS_weights_v2Pro", "GPT_weights_v2Pro", "GPT_SoVITS/pretrained_models"]:
    p = f"/content/GPT-SoVITS/{d}"
    if os.path.islink(p):
        report(f"软链 {d}", "OK", "-> Drive")
    elif os.path.isdir(p):
        report(f"软链 {d}", "WARN", "存在但不是软链（本地目录重启丢），建议重跑 setup.sh")
    else:
        report(f"软链 {d}", "FAIL", "缺失")

sec("5) 依赖自检（env python）")
out, rc = esh("python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'")
print("  torch:", out.strip()[-120:] or "(import 失败)")
report("torch", "OK" if rc == 0 else "FAIL")
for mod in ["faster_whisper", "torchcodec"]:
    _, rc = esh(f"python -c 'import {mod}'")
    report(mod, "OK" if rc == 0 else "FAIL", "" if rc == 0 else "import 失败，可能需要 pip 安装")

sec("6) 预训练模型（Drive 软链下）")
pm = "/content/GPT-SoVITS/GPT_SoVITS/pretrained_models"
for f in [
    "chinese-roberta-wwm-ext-large",
    "chinese-hubert-base",
    "v2Pro/s2Gv2Pro.pth",
    "v2Pro/s2Dv2Pro.pth",
    "s1v3.ckpt",
    "sv/pretrained_eres2netv2w24s4ep4.ckpt",
]:
    report(f"模型 {f}", "OK" if os.path.exists(f"{pm}/{f}") else "FAIL")

sec("7) 数据集现状")
WS = "/content/drive/MyDrive/GPT-SoVITS"
slices = (
    [f for f in os.listdir(f"{WS}/output/slicer_opt") if f.endswith(".wav")]
    if os.path.isdir(f"{WS}/output/slicer_opt")
    else []
)
report("切片 output/slicer_opt", "OK" if len(slices) >= 100 else ("WARN" if slices else "FAIL"), f"{len(slices)} 个")
lists = (
    [f for f in os.listdir(f"{WS}/output/asr_opt") if f.endswith(".list")]
    if os.path.isdir(f"{WS}/output/asr_opt")
    else []
)
report("ASR .list", "OK" if lists else "WARN(未生成，需跑 Step2)", str(lists))
s1 = "/content/GPT-SoVITS/logs/tingshu_club"
for f in ["2-name2text.txt", "4-cnhubert", "5-wav32k", "7-sv_cn", "6-name2semantic.tsv"]:
    report(f"数据集 {f}", "OK" if os.path.exists(f"{s1}/{f}") else "WARN(未生成，Step3 会补)")

sec("8) 总结")
fails = [r for r in results if r[1] == FAIL]
warns = [r for r in results if r[1] == WARN]
print(f"  FAIL {len(fails)} 项 / WARN {len(warns)} 项 / OK {len(results) - len(fails) - len(warns)} 项")
for name, status, detail in fails:
    print(f"    ❌ {name}: {detail}")
for name, status, detail in warns:
    print(f"    ⚠️ {name}: {detail}")
if not fails:
    print("  无致命问题 → 下一步：重跑 Step2(ASR) → Step3(格式化) → Step4(训练)")
