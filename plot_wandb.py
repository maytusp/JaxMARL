# Quick snippet for multi-seed local aggregation
import seaborn as sns

all_data = []
# Match folders for a specific layout/experiment
for run_folder in glob.glob('wandb/run-*-your_experiment_tag*'):
    h_path = os.path.join(run_folder, 'files', 'wandb-history.jsonl')
    if os.path.exists(h_path):
        with open(h_path, 'r') as f:
            for line in f:
                d = json.loads(line)
                d['run_id'] = run_folder # Identify the seed
                all_data.append(d)

df_all = pd.DataFrame(all_data)
sns.lineplot(data=df_all, x='_step', y='returns', ci='sd')
plt.show()