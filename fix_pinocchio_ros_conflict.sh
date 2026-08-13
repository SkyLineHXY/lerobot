#!/usr/bin/env bash
#
# 修复 pinocchio（pip 的 `pin` / cmeel 轮子）与 ROS Noetic 的 libeigenpy 冲突。
#
# ============================== 症状 ==============================
#
#   import pinocchio
#   ImportError: /usr/lib/x86_64-linux-gnu/libboost_python38.so.1.71.0:
#                undefined symbol: _Py_tracemalloc_config
#
# 主臂重力补偿（lerobot/teleoperators/piper_leader/gravity_compensation.py）
# 是在 PiperGravityCompensationLoop.__init__ 里才 `import pinocchio` 的，所以这个
# 错误通常在 leader.connect() -> configure() -> set_manual_control(True) 时才炸出来。
#
# ============================== 根因 ==============================
#
# 系统上同时存在两份 libeigenpy.so，且 SONAME 完全相同、都不带版本号：
#
#   cmeel 那份  SONAME=libeigenpy.so   为 Python 3.12 编译，配 libboost_python312
#   ROS  那份   SONAME=libeigenpy.so   为 Python 3.8  编译，配 libboost_python38
#               （/opt/ros/noetic/lib/x86_64-linux-gnu/libeigenpy.so）
#
# 名字一模一样，于是纯粹由搜索顺序决胜负。一旦选中 ROS 那份，就会拖进
# libboost_python38.so.1.71.0，而它引用的 `_Py_tracemalloc_config` 是 Python 3.8
# 的内部符号，3.12 里已经不存在 —— 于是 undefined symbol。
#
# 动态链接器的搜索顺序是：
#
#   1. DT_RPATH     本对象的，并且沿加载链**向上继承**；本对象存在 RUNPATH 时被忽略
#   2. LD_LIBRARY_PATH                                    <-- ROS 在这里
#   3. DT_RUNPATH   只作用于本对象的直接依赖，**不继承**   <-- pinocchio 原本在这里
#   4. ld.so.cache / 默认目录
#
# pinocchio 的 .so 用的是 RUNPATH，排在 LD_LIBRARY_PATH 之后，必输。
#
# ============================== 做法 ==============================
#
# 把这几个 .so 的 RUNPATH 原样改成 RPATH（路径值不变，只改搜索优先级）。RPATH 排在
# LD_LIBRARY_PATH 之前且沿依赖链传递，因此**与启动方式无关**：
#
#   - `conda activate` 之后跑                              ✓
#   - PyCharm / VSCode 调试器直接调解释器（不走 activate）  ✓
#   - systemd、cron、裸 python                             ✓
#
# 这一点很关键：把 cmeel 目录前置到 LD_LIBRARY_PATH（比如写 conda 的 activate.d 钩子）
# 只对「先 conda activate 再运行」有效，IDE 调试器是直接调用解释器的，钩子根本不执行。
#
# ---- patchelf 的坑 ----
# `patchelf --force-rpath --set-rpath <原值>` 在新值与旧值相同时会**报成功但什么都不做**，
# readelf 一看还是 RUNPATH。必须先 --remove-rpath 再 --set-rpath，本脚本就是这么做的。
#
# ============================== 用法 ==============================
#
#   bash fix_pinocchio_ros_conflict.sh              # 诊断 -> 修复 -> 验证（默认）
#   bash fix_pinocchio_ros_conflict.sh --check      # 只诊断，不改任何文件
#   bash fix_pinocchio_ros_conflict.sh --verify     # 只跑验证
#   PYTHON=/path/to/python bash fix_pinocchio_ros_conflict.sh   # 指定解释器
#
# 依赖：patchelf（sudo apt install patchelf）、binutils 的 readelf
#
# 注意：重装或升级 pinocchio（例如 `pip install --force-reinstall pin`）会覆盖这些
# .so，补丁随之失效 —— 重装后重新跑一遍本脚本即可。脚本是幂等的。
#

set -uo pipefail

MODE="fix"
case "${1:-}" in
    --check)  MODE="check" ;;
    --verify) MODE="verify" ;;
    -h|--help)
        sed -n '2,66p' "$0" | sed 's/^# \{0,1\}//'
        exit 0
        ;;
    "") ;;
    *)
        echo "未知参数: $1（可用: --check / --verify / --help）" >&2
        exit 2
        ;;
esac

PYTHON="${PYTHON:-python}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "错误: 找不到解释器 '$PYTHON'。可用 PYTHON=/path/to/python 指定。" >&2
    exit 1
fi
PYTHON="$(command -v "$PYTHON")"
echo "解释器: $PYTHON"

# ---------------------------------------------------------------- 定位 cmeel
CMEEL_PREFIX="$("$PYTHON" - <<'PY'
import os
import sysconfig

path = os.path.join(sysconfig.get_paths()["purelib"], "cmeel.prefix")
print(path if os.path.isdir(path) else "")
PY
)"

if [ -z "$CMEEL_PREFIX" ]; then
    echo "错误: 该环境里没有 cmeel.prefix 目录。" >&2
    echo "      本脚本只针对 pip 安装的 pinocchio（pip install pin，基于 cmeel 轮子）。" >&2
    echo "      若 pinocchio 是 conda-forge 或 apt 装的，不会有这个冲突，也不该用本脚本。" >&2
    exit 1
fi
echo "cmeel 前缀: $CMEEL_PREFIX"

# ------------------------------------------------------------- 冲突方诊断
# 找出 cmeel 之外的 libeigenpy.so —— 它们就是会抢占加载的那些。
find_conflicts() {
    {
        # LD_LIBRARY_PATH 里的目录（优先级高于 RUNPATH，是主要肇事者）
        IFS=':' read -r -a dirs <<< "${LD_LIBRARY_PATH:-}"
        for d in "${dirs[@]}"; do
            [ -n "$d" ] && [ -e "$d/libeigenpy.so" ] && echo "$d/libeigenpy.so"
        done
        # ldconfig 缓存里的
        ldconfig -p 2>/dev/null | grep -oP '=> \K.*libeigenpy\.so.*' || true
        # ROS 的常见位置，即使当前 shell 没 source 也要报出来
        for d in /opt/ros/*/lib /opt/ros/*/lib/x86_64-linux-gnu; do
            [ -e "$d/libeigenpy.so" ] && echo "$d/libeigenpy.so"
        done
    } 2>/dev/null | grep -v "^$CMEEL_PREFIX" | sort -u
}

CONFLICTS="$(find_conflicts)"
echo
echo "--- 冲突诊断 ---"
if [ -z "$CONFLICTS" ]; then
    echo "没有发现 cmeel 之外的 libeigenpy.so；当前机器上大概率不存在这个冲突。"
else
    echo "发现会抢占加载的 libeigenpy.so："
    while IFS= read -r lib; do
        boost="$(readelf -d "$lib" 2>/dev/null | grep -oP 'NEEDED.*\[\Klibboost_python[^\]]*' | head -1)"
        printf "  %-56s -> %s\n" "$lib" "${boost:-未链接 boost_python}"
    done <<< "$CONFLICTS"
fi

# --------------------------------------------------------------- 目标文件
# Python 扩展模块是 import 的入口；RPATH 沿依赖链继承，理论上只 patch 入口即可，
# 但把 libeigenpy / libcoal / libpinocchio_* 一并处理更稳，代价也只是几个文件。
collect_targets() {
    find "$CMEEL_PREFIX" -name "*.cpython-*.so" -type f 2>/dev/null
    find "$CMEEL_PREFIX/lib" -maxdepth 1 -type f \
        \( -name "libeigenpy.so*" -o -name "libcoal.so*" -o -name "libpinocchio*.so*" \) 2>/dev/null
}

rpath_kind() {  # 打印 RPATH / RUNPATH / 空
    readelf -d "$1" 2>/dev/null | grep -oE "\(RPATH\)|\(RUNPATH\)" | head -1 | tr -d '()'
}

rpath_value() {
    readelf -d "$1" 2>/dev/null | grep -oP '(RUNPATH|RPATH).*\[\K[^\]]+' | head -1
}

echo
echo "--- 目标 .so 当前状态 ---"
TARGETS="$(collect_targets)"
if [ -z "$TARGETS" ]; then
    echo "错误: 在 $CMEEL_PREFIX 下没找到任何目标 .so。" >&2
    exit 1
fi
while IFS= read -r f; do
    printf "  %-10s %-58s %s\n" "$(rpath_kind "$f")" "$(basename "$f")" "$(rpath_value "$f")"
done <<< "$TARGETS"

# ------------------------------------------------------------------- 验证
verify() {
    # 构造一个「最恶劣」环境来验证：清空所有环境变量，只把冲突方所在目录放进
    # LD_LIBRARY_PATH 的最前面，并且不走 conda activate —— 这正是 IDE 调试器的情形。
    local hostile=""
    if [ -n "$CONFLICTS" ]; then
        while IFS= read -r lib; do
            hostile="${hostile:+$hostile:}$(dirname "$lib")"
        done <<< "$CONFLICTS"
    fi
    echo
    echo "--- 验证（模拟 IDE 调试器：不走 activate，冲突目录置于 LD_LIBRARY_PATH 最前）---"
    echo "LD_LIBRARY_PATH=${hostile:-<空>}"

    env -i HOME="$HOME" PATH=/usr/bin:/bin LD_LIBRARY_PATH="$hostile" "$PYTHON" - <<'PY'
import sys

try:
    import pinocchio
except ImportError as exc:
    print(f"✗ 导入失败: {exc}")
    sys.exit(1)

print(f"✓ pinocchio {pinocchio.__version__} 导入成功")

# 光能 import 还不够：必须确认实际映射进来的是 cmeel 那份，而不是碰巧没触发符号。
with open("/proc/self/maps") as fh:
    loaded = {line.split()[-1] for line in fh if "/" in line}
ok = True
for name in ("libeigenpy", "libboost_python"):
    hits = sorted({p for p in loaded if name in p})
    for path in hits:
        good = "cmeel.prefix" in path
        ok = ok and good
        print(f"  {'✓' if good else '✗ 仍在用系统/ROS 的那份'} {path}")
    if not hits:
        print(f"  ? 未映射 {name}")
sys.exit(0 if ok else 1)
PY
    return $?
}

if [ "$MODE" = "check" ]; then
    echo
    echo "（--check 模式：未修改任何文件）"
    exit 0
fi

if [ "$MODE" = "verify" ]; then
    verify
    exit $?
fi

# ------------------------------------------------------------------- 修复
if ! command -v patchelf >/dev/null 2>&1; then
    echo "错误: 未找到 patchelf。请先安装: sudo apt install patchelf" >&2
    exit 1
fi

echo
echo "--- 把 RUNPATH 改成 RPATH ---"
N_PATCHED=0
while IFS= read -r f; do
    value="$(rpath_value "$f")"
    if [ -z "$value" ]; then
        printf "  跳过（无 RPATH/RUNPATH）  %s\n" "$(basename "$f")"
        continue
    fi
    if [ "$(rpath_kind "$f")" = "RPATH" ]; then
        printf "  已是 RPATH                %s\n" "$(basename "$f")"
        continue
    fi
    # 两步走：patchelf 在新旧值相同时会跳过重写，直接 --force-rpath --set-rpath 是无效操作。
    if ! patchelf --remove-rpath "$f"; then
        echo "错误: patchelf --remove-rpath 失败: $f" >&2
        exit 1
    fi
    if ! patchelf --force-rpath --set-rpath "$value" "$f"; then
        echo "错误: patchelf --set-rpath 失败: $f" >&2
        exit 1
    fi
    if [ "$(rpath_kind "$f")" != "RPATH" ]; then
        echo "错误: $f 改写后仍不是 RPATH（patchelf 版本可能不支持 --force-rpath）。" >&2
        exit 1
    fi
    printf "  RUNPATH -> RPATH          %-46s %s\n" "$(basename "$f")" "$value"
    N_PATCHED=$((N_PATCHED + 1))
done <<< "$TARGETS"
echo "本次改写 $N_PATCHED 个文件。"

verify
RC=$?
echo
if [ $RC -eq 0 ]; then
    echo "完成。重装/升级 pinocchio 后请重新运行本脚本。"
else
    echo "验证未通过，请把上面的输出贴出来排查。" >&2
fi
exit $RC
