import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
save_dir = "logs"
os.makedirs(save_dir, exists_ok=True)

# Initialize the API
api = wandb.Api()

# Access your run
# Replace 'your_entity' and 'your_project' with the names in your config
# The ID is the suffix of your folder: lmtom_overcooked_v2_wide_cramped_room_v2
run_id = "lmtom_overcooked_v2_wide_cramped_room_v2"
run = api.run(f"maytusp/like-me-tom-overcookedv2/6mfvt3it")
filename = f"{run_id}.png"

save_path = os.path.join(save_dir, filename)


# Download the history
# scan_history() is better than history() because it avoids the 500-point sampling limit
history = run.scan_history()
df = pd.DataFrame([row for row in history])

# Check column names (W&B sometimes prefixes them)
print("Columns found:", df.columns.tolist())

# Plot Mean and Standard Deviation
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")

# Seaborn lineplot automatically detects multiple entries for the same 'update_step'
# and calculates the mean (line) and standard deviation (shaded area).
sns.lineplot(
    data=df, 
    x='update_step', 
    y='returned_episode_returns', # Change this to your desired metric
    errorbar='sd',       # This replaces the older 'ci="sd"'
    label='Like-Me ToM (5 Seeds)'
)

plt.title(f"Performance on {run_id}")
plt.xlabel("Update Step")
plt.ylabel("Combined Reward")
plt.legend()

# Save the figure
# bbox_inches='tight' ensures labels aren't cut off
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Successfully saved figure to: {save_path}")

# Optional: Close the plot to free up memory
plt.close()
