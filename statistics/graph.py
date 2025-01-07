import matplotlib.pyplot as plt
from Battery.secrets import volt_column, curr_column
import pandas as pd
from Battery.secrets import show_graphFile
result_type = 'volt'
sequence_length = 4

test_data = pd.read_csv(f"../PostProcess/11column/{show_graphFile}")
column = volt_column if result_type == 'volt' else curr_column
y_true = test_data[column]

# draw graph
plt.figure(figsize=(10, 6))
plt.plot(y_true, label='True values', color='blue', marker='o')

plt.title('True vs Predicted Values')
plt.xlabel('Samples')
plt.ylabel('Values')

plt.legend()
plt.show()
