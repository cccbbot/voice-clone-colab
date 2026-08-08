#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""修复 GPT-SoVITS Colab 环境 torchaudio 加载失败（版本匹配重装）。
用法（Colab cell）:
    !wget -q -O /content/fix_torchaudio.py https://raw.githubusercontent.com/cccbbot/voice-clone-colab/master/fix_torchaudio.py && python /content/fix_torchaudio.py
"""
import re
import subprocess


def sh(cmd, timeout=900):
    r = subprocess.run(
        ["bash", "-lc", "source /content/env_path.sh && " + cmd],
        capture_output=True, text=True, timeout=timeout,
    )
    out = (r.stdout or "") + (r.stderr or "")
    print(out[-1000:])
    return r.returncode


def main():
    print("=== 1) 当前 torch 版本 ===")
    out = subprocess.run(
        ["bash", "-lc", "source /content/env_path.sh && python -c 'import torch; print(torch.__version__)'"],
        capture_output=True, text=True,
    ).stdout
    print(out.strip())
    m = re.search(r"\d+\.\d+\.\d+", out)
    ver = m.group(0) if m else None
    if not ver:
        print("无法获取 torch 版本，中止")
        return 1

    print("=== 2) 重装 torchaudio==" + ver + " (cu126) ===")
    rc = sh(
        "pip install --force-reinstall --no-deps torchaudio==" + ver
        + " --index-url https://download.pytorch.org/whl/cu126"
    )
    if rc != 0:
        print("cu126 源没有该版本，改用 PyPI CPU 版（ASR 加载音频足够用）")
        sh("pip install --force-reinstall --no-deps torchaudio==" + ver)

    print("=== 3) 验证 ===")
    rc = sh("python -c 'import torchaudio; print(torchaudio.__version__)'")
    if rc == 0:
        print("OK 修复完成，直接重跑 Step2")
        return 0
    print("仍失败，请把上面输出贴回给 Hermes")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
