#!/usr/bin/env python3

import shutil
import r2pipe
import sys

def patch_250k(device_offset):
    fn = "INSPECT" if device_offset == 0 else f"INSPEC{device_offset}"
    o_fn = fn[0:4]+'2'+fn[5:]

    shutil.copy(f"{fn}.MAP",f"{o_fn}.MAP")
    shutil.copy(f"{fn}.EXE",f"{o_fn}.EXE")

    print("Copied 500 K version")

    with open(f"{fn}.MAP",'r') as f:
        lines = f.readlines()

    found = 0
    for l in lines:
        if "INITCAN" in l:
            addr_INITCAN = int(l[1:5],16)*16+int(l[6:10],16)
            found += 1
        if "CANINIT" in l:
            addr_CANINIT = int(l[1:5],16)*16+int(l[6:10],16)
            found += 1

    if found != 2:
        sys.exit(1)
    print("Identified addresses of INITCAN and CANINIT")

    r = r2pipe.open(f"{o_fn}.EXE", flags=['-w'])
    r.cmd(f"af INITCAN {addr_INITCAN}")
    r.cmd(f"af CANINIT {addr_CANINIT}")

    for x in r.cmdj("axtj CANINIT"):
        if x["fcn_name"] != "INITCAN":
            sys.exit(2)
        r.cmd(f"s {x['from']}")
        push_addr = int(r.cmd(f"/ba push 2 | head -n 1 | cut -f 1 -d ' '"),16)
        r.cmd(f"s {push_addr}")
        r.cmd("wa push 1")
        print("Applied patch")
    r.quit()

with open("INSPECT.EXE", 'rb') as f:
    file = f.read()

can_id_positions = [0x4948,0x494A,0x494E,0x4950,0x4954,0x4956,0x495a,0x495c,0x4960,0x4962,0x4966,0x4968]
can_id_positions = [x+0x49d8-0x4948 for x in can_id_positions]

for device_offset in range(1,16):
    file_copy = bytearray(file)

    file_copy[0x30fd+5] = file_copy[0x30fd+5]+device_offset

    for i,offset in enumerate(can_id_positions):
        canid = 0x101+i+16*device_offset
        file_copy[offset] = canid & 0xff
        file_copy[offset+1] = (canid & 0xff00) >> 8

    file_copy[0x4f1+12:0x4f1+14] = [ord(x) for x in f"{device_offset:02d}"]

    with open(f"INSPEC{device_offset}.EXE",'wb') as f:
        f.write(file_copy)


    shutil.copy("INSPECT.MAP", f"INSPEC{device_offset}.MAP")


# now create EISmeter compatible version

for device_offset in range(0,16):
    patch_250k(device_offset)
