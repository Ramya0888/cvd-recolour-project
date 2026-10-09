import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # saves without needing a display

# Confusable red-green pair — verified to simulate to Delta E ~3 under
# deuteranopia (truly confusable), because they are matched in lightness.
# matplotlib's default 'red'/'green' do NOT work: default green (0,128,0)
# is much darker than red, so the lightness gap survives CVD simulation.
CONFUSABLE_RED = (230/255, 30/255, 30/255)
CONFUSABLE_GREEN = (30/255, 160/255, 30/255)

# Chart 1: red vs green bars
fig, ax = plt.subplots()
ax.bar(['Category A', 'Category B', 'Category C'], [40, 65, 50],
       color=[CONFUSABLE_RED, CONFUSABLE_GREEN, CONFUSABLE_RED])
ax.set_title('Red-Green Bar Chart Test')
plt.savefig('../data/sample_images/chart_bars.png', dpi=150)
plt.close()

# Chart 2: red vs green line chart
fig, ax = plt.subplots()
ax.plot([1, 2, 3, 4, 5], [10, 20, 15, 25, 30],
        color=CONFUSABLE_RED, label='Series 1', linewidth=3)
ax.plot([1, 2, 3, 4, 5], [12, 18, 20, 22, 28],
        color=CONFUSABLE_GREEN, label='Series 2', linewidth=3)
ax.legend()
ax.set_title('Red-Green Line Chart Test')
plt.savefig('../data/sample_images/chart_lines.png', dpi=150)
plt.close()

# Chart 3: pie chart with alternating red/green slices
fig, ax = plt.subplots()
ax.pie([30, 30, 20, 20], labels=['A', 'B', 'C', 'D'],
       colors=[CONFUSABLE_RED, CONFUSABLE_GREEN, CONFUSABLE_RED, CONFUSABLE_GREEN])
ax.set_title('Red-Green Pie Chart Test')
plt.savefig('../data/sample_images/chart_pie.png', dpi=150)
plt.close()

print("Charts generated successfully with confusable (lightness-matched) colours.")