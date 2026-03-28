# Q1 commands: 
## Processing data:  

Navigate to hw3 directory.
```
python3 rob831/scripts/read_results.py --logdir data/q1_ddqn_1_LunarLander-v3_07-03-2026_22-54-42-20260307T231643Z-3-001/q1_ddqn_1_LunarLander-v3_07-03-2026_22-54-42 --csv_out data/q1_ddqn_1_LunarLander-v3_07-03-2026_22-54-42-20260307T231643Z-3-001/q1_ddqn_1_LunarLander-v3_07-03-2026_22-54-42/q1_ddqn1.csv

python3 rob831/scripts/read_results.py --logdir data/q1_ddqn_2_LunarLander-v3_07-03-2026_23-16-19-20260307T233627Z-3-001/q1_ddqn_2_LunarLander-v3_07-03-2026_23-16-19 0 --csv_out data/q1_ddqn_2_LunarLander-v3_07-03-2026_23-16-19-20260307T233627Z-3-001/q1_ddqn_2_LunarLander-v3_07-03-2026_23-16-19/q1_ddqn2.csv

python3 rob831/scripts/read_results.py --logdir data/q1_ddqn_3_LunarLander-v3_07-03-2026_23-36-04-20260307T235506Z-3-001/q1_ddqn_3_LunarLander-v3_07-03-2026_23-36-04 --csv_out data/q1_ddqn_3_LunarLander-v3_07-03-2026_23-36-04-20260307T235506Z-3-001/q1_ddqn_3_LunarLander-v3_07-03-2026_23-36-04/ddqn3.csv

python3 rob831/scripts/read_results.py --logdir data/q1_dqn_1_LunarLander-v3_07-03-2026_21-50-11-20260307T221623Z-3-001/q1_dqn_1_LunarLander-v3_07-03-2026_21-50-11 --csv_out data/q1_dqn_1_LunarLander-v3_07-03-2026_21-50-11-20260307T221623Z-3-001/q1_dqn_1_LunarLander-v3_07-03-2026_21-50-11/q1_dqn1.csv

python3 rob831/scripts/read_results.py --logdir data/q1_dqn_2_LunarLander-v3_07-03-2026_22-12-52-20260307T223340Z-3-001/q1_dqn_2_LunarLander-v3_07-03-2026_22-12-52 --csv_out data/q1_dqn_2_LunarLander-v3_07-03-2026_22-12-52-20260307T223340Z-3-001/q1_dqn_2_LunarLander-v3_07-03-2026_22-12-52/q1_dqn2.csv

python3 rob831/scripts/read_results.py --logdir data/q1_dqn_3_LunarLander-v3_07-03-2026_22-33-16-20260307T225504Z-3-001/q1_dqn_3_LunarLander-v3_07-03-2026_22-33-16
--csv_out data/q1_dqn_3_LunarLander-v3_07-03-2026_22-33-16-20260307T225504Z-3-001/q1_dqn_3_LunarLander-v3_07-03-2026_22-33-16/q1_dqn3.csv
```
 

### Plotting Q1  

```
python3 rob831/scripts/plot_q1.py --dqn_csvs data/q1_dqn_3_LunarLander-v3_07-03-2026_22-33-16-20260307T225504Z-3-001/q1_dqn_3_LunarLander-v3_07-03-2026_22-33-16/q1_dqn3.csv data/q1_dqn_2_LunarLander-v3_07-03-2026_22-12-52-20260307T223340Z-3-001/q1_dqn_2_LunarLander-v3_07-03-2026_22-12-52/q1_dqn2.csv data/q1_dqn_1_LunarLander-v3_07-03-2026_21-50-11-20260307T221623Z-3-001/q1_dqn_1_LunarLander-v3_07-03-2026_21-50-11/q1_dqn1.csv --ddqn_csvs data/q1_ddqn_3_LunarLander-v3_07-03-2026_23-36-04-20260307T235506Z-3-001/q1_ddqn_3_LunarLander-v3_07-03-2026_23-36-04/ddqn3.csv data/q1_ddqn_2_LunarLander-v3_07-03-2026_23-16-19-20260307T233627Z-3-001/q1_ddqn_2_LunarLander-v3_07-03-2026_23-16-19/q1_ddqn2.csv data/q1_ddqn_1_LunarLander-v3_07-03-2026_22-54-42-20260307T231643Z-3-001/q1_ddqn_1_LunarLander-v3_07-03-2026_22-54-42/q1_ddqn1.csv --errorbar_style band --show 
```

### Output
```
Saved plot to: dqn_vs_ddqn_plot.png 
```
 
# Q2 commands 
## Plotting
```
python3 rob831/scripts/plot_csv.py data/q2_10_10_CartPole-v0_08-03-2026_18-35-11-20260308T185853Z-3-001/q2_10_10_CartPole-v0_08-03-2026_18-35-11/run1.csv --xlabel "Iteration" --ylabel "Eval. Ave. Return" --title "Q2: Cartpole-V0 (ntu: 10, ngsptu: 10)" 
```

 
# Q3 commands
## Plotting
```
python3 rob831/scripts/plot_csv.py data/q3_10_10_InvertedPendulum-v4_08-03-2026_18-50-39-20260308T190355Z-3-001/q3_10_10_InvertedPendulum-v4_08-03-2026_18-50-39/run1.csv --xlabel "Iteration" --ylabel "Eval Ave. Return" --title "Q3: Inverted Pendulum-V4 (ntu: 10, ngsptu: 10)" 
```