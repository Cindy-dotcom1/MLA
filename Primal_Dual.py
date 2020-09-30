import pygal
xy_chart = pygal.XY()
xy_chart.title = 'Primal and Dual problems'
xy_chart.add('Primal problem (f_0)', [(x, x**2) for x in range(-5, 6, 1)])
xy_chart.add('Dual problem (g)', [(x, -x**2) for x in range(-5, 6, 1)])
chart.render_to_file('C:/Users/cindy/OneDrive/Bureau/Master 2/Untitled-1.svg')