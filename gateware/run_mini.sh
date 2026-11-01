#!/bin/sh
# Build, verify and run the FPGA backprop engine on the Mac mini (Icepi Zero on its USB port).
# Usage (from the repo): rsync -a gateware/ mini.local:anneq-runs/fpga/src/ && ssh mini.local 'sh anneq-runs/fpga/src/run_mini.sh'
# Outputs stay in ~/anneq-runs/fpga/work on the mini; the measurements kept in git are in results/fpga/.
# SRAM load only (openFPGALoader without --write-flash): the board's flash image is untouched.
set -e
export PATH=/opt/homebrew/bin:$PATH
N=${NENG:-10}
mkdir -p ~/anneq-runs/fpga/work && cd ~/anneq-runs/fpga/work
P() { uv run -q --no-project --with numpy --with pyserial python ../src/py/anneq_fpga.py "$@"; }
P lut tanh_lut.hex
P job small.bin --trials 3 --epochs 50
cc -O3 -mcpu=native -o bpfix ../src/ref/bpfix.c -lm
./bpfix fix tanh_lut.hex small.bin ref.bin
verilator --cc --exe --build -O3 -Wno-fatal -Wno-WIDTH --top-module bp_engine \
    ../src/rtl/bp_engine.v ../src/sim/tb_engine.cpp -Mdir obj > build.log 2>&1
./obj/Vbp_engine small.bin sim.bin && cmp ref.bin sim.bin && echo "sim: bit-exact"
yosys -q -l syn$N.log -p "chparam -set NENG $N top; synth_ecp5 -top top -json top$N.json" ../src/rtl/top.v ../src/rtl/bp_engine.v
nextpnr-ecp5 --25k --package CABGA256 --lpf ../src/rtl/icepi-zero.lpf --json top$N.json --textcfg top$N.config --seed 1 -l pnr$N.log > /dev/null 2>&1
grep "Max frequency" pnr$N.log | tail -1
ecppack --compress top$N.config top$N.bit
openFPGALoader -b icepi-zero top$N.bit
for s in 6 10 14 21; do
    P job ch1_$s.bin --trials 20 --snr $s
    ./bpfix fix tanh_lut.hex ch1_$s.bin fix_$s.bin
    P board ch1_$s.bin brd_$s.bin --engines $N
    cmp fix_$s.bin brd_$s.bin && echo "board: bit-exact"
    P ber ch1_$s.bin brd_$s.bin
    P numpy ch1_$s.bin
done
./bpfix bench tanh_lut.hex ch1_10.bin
