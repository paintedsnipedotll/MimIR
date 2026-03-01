"""Host test for MAC probe kernel — isolates lane mapping of aie2p.mac_i16_i64.

Runs three input patterns through a single MAC + SRS(shift=0, round=0)
to determine how the 32 input lanes of A and B map to the 16 output lanes.

Pattern 1: A = all ones, B = all ones
  -> Shows the reduction factor per output lane (expect 32 if each lane sums all pairs)

Pattern 2: A[0] = [0,1,2,...,31], B[0] = all ones
  -> Shows which A input positions contribute to each output lane

Pattern 3: A[0] = all ones, B[0] = [0,1,2,...,31]
  -> Shows which B input positions contribute to each output lane

Usage:
  python3 test_mac_probe.py -x build/final.xclbin -i build/insts.txt

Requires: pyxrt (from /opt/xilinx/xrt/python/)
"""

import argparse
import os
import struct
import sys

import numpy as np

# pyxrt lives at /opt/xilinx/xrt/python/ — ensure it's importable
XRT_PYTHON = "/opt/xilinx/xrt/python"
if XRT_PYTHON not in sys.path:
    sys.path.insert(0, XRT_PYTHON)

import pyxrt as xrt

A_SIZE = 4 * 32  # 128 i16 values (256 bytes)
B_SIZE = 4 * 32  # 128 i16 values (256 bytes)
C_SIZE = 16      # 16 i32 values  (64 bytes)


def load_instructions(filepath):
    """Auto-detect format and load instruction sequence."""
    try:
        with open(filepath, "r") as f:
            return [int(line, 16) for line in f if line.strip()]
    except (ValueError, UnicodeDecodeError):
        size = os.path.getsize(filepath)
        with open(filepath, "rb") as f:
            return list(struct.unpack(f"{size // 4}I", f.read()))


def run_one(kernel, bo_instr, bo_a, bo_b, bo_c, instr_v, buf_a, buf_b, label):
    """Run a single MAC probe with given A and B buffers, print results."""
    bo_a.write(buf_a, 0)
    bo_b.write(buf_b, 0)
    # Zero out C
    bo_c.write(np.zeros(C_SIZE, dtype=np.int32), 0)
    bo_instr.write(np.array(instr_v, dtype=np.uint32), 0)

    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_a.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_b.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_c.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    run = kernel(3, bo_instr, len(instr_v), bo_a, bo_b, bo_c)
    r = run.wait()
    assert r == xrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED, f"Kernel failed: {r}"

    bo_c.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    buf_c = bo_c.read(C_SIZE * 4, 0)
    result = np.frombuffer(buf_c, dtype=np.int32)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  A[0] (first 32 i16): {buf_a[:32]}")
    print(f"  B[0] (first 32 i16): {buf_b[:32]}")
    print(f"  Output C[0:16] (i32):")
    for i in range(16):
        print(f"    C[{i:2d}] = {result[i]}")
    print()
    return result


def main():
    parser = argparse.ArgumentParser(description="MAC probe lane mapping test")
    parser.add_argument("-x", "--xclbin", required=True, help="Path to .xclbin")
    parser.add_argument("-i", "--instr", required=True, help="Path to instruction file")
    parser.add_argument("-k", "--kernel", default="MLIR_AIE", help="Kernel name")
    args = parser.parse_args()

    instr_v = load_instructions(args.instr)
    print(f"Loaded {len(instr_v)} instructions from {args.instr}")

    device = xrt.device(0)
    xclbin = xrt.xclbin(args.xclbin)
    device.register_xclbin(xclbin)
    context = xrt.hw_context(device, xclbin.get_uuid())
    xkernels = xclbin.get_kernels()
    xkernel = [k for k in xkernels if args.kernel in k.get_name()][0]
    kernel = xrt.kernel(context, xkernel.get_name())

    bo_instr = xrt.bo(device, len(instr_v) * 4, xrt.bo.cacheable, kernel.group_id(1))
    bo_a = xrt.bo(device, A_SIZE * 2, xrt.bo.host_only, kernel.group_id(3))
    bo_b = xrt.bo(device, B_SIZE * 2, xrt.bo.host_only, kernel.group_id(4))
    bo_c = xrt.bo(device, C_SIZE * 4, xrt.bo.host_only, kernel.group_id(5))

    # ---- Pattern 1: all ones ----
    a1 = np.ones(A_SIZE, dtype=np.int16)
    b1 = np.ones(A_SIZE, dtype=np.int16)
    # Only row 0 has data; rows 1-3 are zero
    a1[32:] = 0
    b1[32:] = 0
    run_one(kernel, bo_instr, bo_a, bo_b, bo_c, instr_v,
            a1, b1, "Pattern 1: A=ones(32), B=ones(32) [rows 1-3 zero]")

    # ---- Pattern 2: A[0] = iota, B[0] = ones ----
    a2 = np.zeros(A_SIZE, dtype=np.int16)
    a2[:32] = np.arange(32, dtype=np.int16)
    b2 = np.zeros(B_SIZE, dtype=np.int16)
    b2[:32] = 1
    run_one(kernel, bo_instr, bo_a, bo_b, bo_c, instr_v,
            a2, b2, "Pattern 2: A[0]=[0..31], B[0]=ones")

    # ---- Pattern 3: A[0] = ones, B[0] = iota ----
    a3 = np.zeros(A_SIZE, dtype=np.int16)
    a3[:32] = 1
    b3 = np.zeros(B_SIZE, dtype=np.int16)
    b3[:32] = np.arange(32, dtype=np.int16)
    run_one(kernel, bo_instr, bo_a, bo_b, bo_c, instr_v,
            a3, b3, "Pattern 3: A[0]=ones, B[0]=[0..31]")

    # ---- Pattern 4: A[16]=1 only, B=ones → does upper half of A contribute? ----
    a4 = np.zeros(A_SIZE, dtype=np.int16)
    a4[16] = 1  # only A[16] is nonzero
    b4 = np.zeros(B_SIZE, dtype=np.int16)
    b4[:32] = 1
    run_one(kernel, bo_instr, bo_a, bo_b, bo_c, instr_v,
            a4, b4, "Pattern 4: A[16]=1 only, B=ones → upper-half A test")

    # ---- Pattern 5: A=ones, B[16]=1 only → does upper half of B contribute? ----
    a5 = np.zeros(A_SIZE, dtype=np.int16)
    a5[:32] = 1
    b5 = np.zeros(B_SIZE, dtype=np.int16)
    b5[16] = 1  # only B[16] is nonzero
    run_one(kernel, bo_instr, bo_a, bo_b, bo_c, instr_v,
            a5, b5, "Pattern 5: A=ones, B[16]=1 only → upper-half B test")

    # ---- Pattern 6: A[0..31]=iota, B[0..31]=iota → cross-check full dot product ----
    a6 = np.zeros(A_SIZE, dtype=np.int16)
    a6[:32] = np.arange(32, dtype=np.int16)
    b6 = np.zeros(B_SIZE, dtype=np.int16)
    b6[:32] = np.arange(32, dtype=np.int16)
    run_one(kernel, bo_instr, bo_a, bo_b, bo_c, instr_v,
            a6, b6, "Pattern 6: A=iota, B=iota → C[i]=i*i or i*(i+16)?")

    print("All patterns completed successfully.")


if __name__ == "__main__":
    main()
