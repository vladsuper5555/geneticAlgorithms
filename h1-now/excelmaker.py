import pandas as pd

# read data from files
hill_climbing_data_method_best = pd.read_csv('hill_climbing_data_method_best.txt', header=None)
hill_climbing_data_method_first = pd.read_csv('hill_climbing_data_method_first.txt', header=None)
hill_climbing_data_method_worst = pd.read_csv('hill_climbing_data_method_worst.txt', header=None)
# simulated_annealing_data = pd.read_csv('simulated_annealing_data.txt', header=None)

# create dataframes
df_hill_climbing_best = pd.DataFrame(hill_climbing_data_method_best)
df_hill_climbing_first = pd.DataFrame(hill_climbing_data_method_first)
df_hill_climbing_worst = pd.DataFrame(hill_climbing_data_method_worst)
# df_simulated_annealing = pd.DataFrame(simulated_annealing_data)

# write data to Excel file
with pd.ExcelWriter('results.xlsx') as writer:
    df_hill_climbing_best.to_excel(writer, sheet_name='Hill_Climbing_Data_Best', index=False)
    df_hill_climbing_first.to_excel(writer, sheet_name='Hill_Climbing_Data_First', index=False)
    df_hill_climbing_worst.to_excel(writer, sheet_name='Hill_Climbing_Data_Worst', index=False)
    # df_simulated_annealing.to_excel(writer, sheet_name='Simulated_Annealing', index=False)