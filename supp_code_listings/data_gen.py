import os
import subprocess
import numpy as np
import pandas as pd
import skrf as rf
from scipy.stats import qmc

# --- 1. SETUP ---
NUM_SIMULATIONS = 20000
TEMPLATE_FILE   = "circuit_template.net"
TEMP_NETLIST    = "temp_run.net"
OUTPUT_S2P      = "sim_output.s2p"   # 2-port Touchstone
ADS_SIM_EXE     = r"C:\Program Files\Keysight\ADS202X\bin\hpeesofsim.exe"

# Variable parameters: [Rb(Ohm), Cb(fF), Rd(Ohm), Ld(nH), Cd(fF), Ra(Ohm)]
lower_bounds = [1000,  50,  2000, 1.0, 0.5,  10]
upper_bounds  = [20000, 200, 25000, 10.0, 5.0, 100]

# --- 2. LHS SAMPLING ---
sampler    = qmc.LatinHypercube(d=len(lower_bounds))
sample     = sampler.random(n=NUM_SIMULATIONS)
parameters = qmc.scale(sample, lower_bounds, upper_bounds)

with open(TEMPLATE_FILE, 'r') as f:
    template_text = f.read()

dataset = []

# --- 3. SIMULATION LOOP ---
for i in range(NUM_SIMULATIONS):
    rb, cb, rd, ld, cd, ra = parameters[i]

    current_netlist = template_text.format(
        Rb_val=rb, Cb_val=cb, Rd_val=rd,
        Ld_val=ld, Cd_val=cd, Ra_val=ra
    )
    with open(TEMP_NETLIST, 'w') as f:
        f.write(current_netlist)

    subprocess.run([ADS_SIM_EXE, TEMP_NETLIST],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # --- 4. PARSE 2-PORT RESULTS ---
    try:
        network = rf.Network(OUTPUT_S2P)
        freqs   = network.f

        # Retain all three S-parameters: S11, S21, S22
        s11_re = network.s_re[:, 0, 0];  s11_im = network.s_im[:, 0, 0]
        s21_re = network.s_re[:, 1, 0];  s21_im = network.s_im[:, 1, 0]
        s22_re = network.s_re[:, 1, 1];  s22_im = network.s_im[:, 1, 1]

        # Row: [6 labels] + [Re(S11)|Im(S11)|Re(S21)|Im(S21)|Re(S22)|Im(S22)]
        # Total raw features: 6 x 435 = 2610
        row = [rb, cb, rd, ld, cd, ra]
        row.extend(s11_re.tolist()); row.extend(s11_im.tolist())
        row.extend(s21_re.tolist()); row.extend(s21_im.tolist())
        row.extend(s22_re.tolist()); row.extend(s22_im.tolist())
        dataset.append(row)

    except Exception:
        pass

# --- 5. SAVE TO CSV ---
cols = ['Rb', 'Cb', 'Rd', 'Ld', 'Cd', 'Ra']
for tag, mat in [('S11_Re',freqs),('S11_Im',freqs),
                 ('S21_Re',freqs),('S21_Im',freqs),
                 ('S22_Re',freqs),('S22_Im',freqs)]:
    cols.extend([f'{tag}_{f/1e9:.2f}GHz' for f in mat])

df = pd.DataFrame(dataset, columns=cols)
df.to_csv("ssec_training_data.csv", index=False)