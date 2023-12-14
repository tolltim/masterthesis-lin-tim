import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# Define the data points and project start dates for inner and outer areas
data_points = {
    'Südliche Au': {
        'before_inner': 0.82, 'after_inner': 0.84,
        'before_outer': 0.83, 'after_outer': 0.83,
        'project_start': '2023-06-12'
    },
    'Walchenseeplatz': {
        'before_inner': 0.85, 'after_inner': 0.87,
        'before_outer': 0.84, 'after_outer': 0.86,
        'project_start': '2023-07-05'
    }
}

# Colors
tum_blue = '#3070B3'
tum_dark_blue = '#072140'
tum_lighter_blue = '#5E94D4'
tum_orange = '#F7811E'
tum_green = '#9FBA36'
black = '#000000'
tum_gray = '#6A757E'
tum_light_gray = '#dad7cb'

# Create a date range from May 2023 to September 2023
dates = pd.date_range(start="2023-04-30", end="2023-09-20", freq='MS')

# Set up the plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot each area's before and after project start date for inner and outer areas
for i, (area, values) in enumerate(data_points.items()):
    # Inner area
    axs[i].plot([dates[0], pd.to_datetime(values['project_start'])], [values['before_inner'], values['before_inner']],
                label=f'{area} inner before', color=tum_dark_blue, linestyle = '--')
    axs[i].plot([pd.to_datetime(values['project_start']), dates[-1]], [values['after_inner'], values['after_inner']],
                label=f'{area} inner after', color=tum_dark_blue)

    # Outer area
    axs[i].plot([dates[0], pd.to_datetime(values['project_start'])], [values['before_outer'], values['before_outer']],
                label=f'{area} outer before', color=tum_lighter_blue, linestyle = '--')
    axs[i].plot([pd.to_datetime(values['project_start']), dates[-1]], [values['after_outer'], values['after_outer']],
                label=f'{area} outer after', color=tum_lighter_blue)

    # Project start line
    axs[i].axvline(pd.to_datetime(values['project_start']), color=tum_orange, linestyle='--', label=f'Project start {area}')
    axs[i].set_title(area)
    axs[i].legend()

# Format the dates on the x-axis to show month and year only
date_form = mdates.DateFormatter("%b-%Y")
axs[0].xaxis.set_major_formatter(date_form)
axs[1].xaxis.set_major_formatter(date_form)
axs[0].xaxis.set_major_locator(mdates.MonthLocator())
axs[1].xaxis.set_major_locator(mdates.MonthLocator())

# Set the y-axis label and limit
axs[0].set_ylabel('Average Relative Speed')
axs[0].set_ylim(0, 1)
axs[1].set_ylim(0, 1)

# Rotate the date labels for clarity
plt.setp(axs[0].xaxis.get_majorticklabels(), rotation=45)
plt.setp(axs[1].xaxis.get_majorticklabels(), rotation=45)

# Improve the spacing between subplots
plt.tight_layout()

# Show the plot
plt.show()
