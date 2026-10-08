import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # so it saves without needing a display

# Chart 1: simple red vs green bars
fig, ax = plt.subplots()
ax.bar(['Category A', 'Category B', 'Category C'], [40, 65, 50],
       color=['red', 'green', 'red'])
ax.set_title('Red-Green Bar Chart Test')
plt.savefig('../data/sample_images/chart_bars.png', dpi=150)
plt.close()

# Chart 2: red vs green line chart
fig, ax = plt.subplots()
ax.plot([1,2,3,4,5], [10,20,15,25,30], color='red', label='Series 1', linewidth=3)
ax.plot([1,2,3,4,5], [12,18,20,22,28], color='green', label='Series 2', linewidth=3)
ax.legend()
ax.set_title('Red-Green Line Chart Test')
plt.savefig('../data/sample_images/chart_lines.png', dpi=150)
plt.close()

# Chart 3: pie chart with red/green slices
fig, ax = plt.subplots()
ax.pie([30, 30, 20, 20], labels=['A','B','C','D'],
       colors=['red','green','red','green'])
ax.set_title('Red-Green Pie Chart Test')
plt.savefig('../data/sample_images/chart_pie.png', dpi=150)
plt.close()

print("Charts generated successfully.")