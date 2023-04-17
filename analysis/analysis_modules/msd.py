# Mean Squared Displacement
pxDimension = 1 # has to be fixed 
fps = 10 # fps of the video
maxLagtime = 1000 # maximum lagtime to be considered
#x = np.array(imsd[1:].index)
x = np.arange(1., 100.1, .1)


## IMSD
print("Global IMSD")
imsd, fit, pw_exp = get_imsd(rawTrajs, pxDimension, fps, maxLagtime, nDrops)
imsd_smooth, fit_smooth, pw_exp_smooth = get_imsd(smoothTrajs, pxDimension, fps, maxLagtime, nDrops)
fig, (ax, ax1) = plt.subplots(1, 2, figsize = (10, 4), sharey=True)
ax.plot(imsd[1:].index, fit[0])
ax.plot(imsd.index, imsd[0], label = "Raw Trajs")
ax.set_title(f"k: {round(pw_exp[0, 0, 1], 2)} $\pm$ {round(pw_exp[0, 1, 1], 2)}   A: {round(pw_exp[0, 0, 0], 2)} $\pm$ {round(pw_exp[0, 1, 0], 2)} ")
ax.set(xscale = 'log', yscale = 'log', xlabel = "Time Lag [s]", ylabel = "MSD [$px^2$]")
ax.grid()
ax.legend()

ax1.plot(imsd_smooth[1:].index, fit_smooth[0])
ax1.plot(imsd_smooth.index, imsd_smooth[0], label = "Smooth Trajs")
ax1.set_title(f"k: {round(pw_exp_smooth[0, 0, 1], 2)} $\pm$ {round(pw_exp_smooth[0, 1, 1], 2)}  A: {round(pw_exp_smooth[0, 0, 0], 2)} $\pm$ {round(pw_exp_smooth[0, 1, 0], 2)} ")
ax1.set(xscale = 'log', yscale = 'log', xlabel = "Time Lag [s]", ylabel = "MSD [$px^2$]")
ax1.grid()
ax1.legend()
if save_verb: plt.savefig("./results/mean_squared_displacement/raw_smooth_confront.png")
if show_verb: 
    plt.show()
else: 
    plt.close()
fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
ax.plot(imsd.index, imsd)
ax.set(xscale = 'log', yscale = 'log', xlabel = "Time Lag [s]", ylabel = "MSD [px^2]")
ax.grid()
ax1.errorbar(np.arange(nDrops), pw_exp[:, 0, 1], yerr=pw_exp[:, 1, 1], fmt = 'o', capsize = 3)
ax1.set(xlabel = "Particle ID", ylabel = "Powerlaw Exponent")
ax1.grid()
plt.suptitle("Mean Squared Displacement - Raw Trajectories")
plt.tight_layout()
if save_verb: plt.savefig("./results/mean_squared_displacement/IMSD_raw.png")
if show_verb: 
    plt.show()
else:
    plt.close()

fig, (ax, ax1) = plt.subplots(1, 2, figsize=(10, 4))
ax.plot(imsd_smooth.index, imsd_smooth)
ax.set(xscale = 'log', yscale = 'log', xlabel = "Time Lag [s]", ylabel = "MSD [px^2]")
ax.grid()
ax1.errorbar(np.arange(nDrops), pw_exp_smooth[:, 0, 1], yerr=pw_exp_smooth[:, 1, 1], fmt = 'o', capsize = 3)
ax1.set(xlabel = "Particle ID", ylabel = "Powerlaw Exponent")
ax1.grid()
plt.suptitle("Mean Squared Displacement - Smooth Trajectories")
plt.tight_layout()
if save_verb: plt.savefig("./results/mean_squared_displacement/IMSD_smooth.png")
if show_verb: 
    plt.show()
else:
    plt.close()

## EMSD
print("Global EMSD")
MSD_b, MSD_r, fit = get_emsd(imsd, x, red_particle_idx, nDrops)
MSD_b_smooth, MSD_r_smooth, fit_smooth = get_emsd(imsd_smooth, x, red_particle_idx, nDrops)
a = [round(fit["pw_exp_b"][0, 1], 3), round(fit["pw_exp_b"][1, 1], 3)]
b = [round(fit["pw_exp_r"][0, 1], 3), round(fit["pw_exp_r"][1, 1], 3)]

print(f"Raw trajs - Blue Particles: {a[0]} ± {a[1]}, Red Particle: {b[0]} ± {b[1]}")

fig, ax = plt.subplots(1, 1, figsize = (10, 4))
ax.plot(imsd.index, MSD_b[0], 'b-', label = "Blue particles") 
ax.plot(imsd[1:].index, fit["fit_b"], 'b--')
ax.fill_between(imsd.index, MSD_b[0] - MSD_b[1], MSD_b[0] + MSD_b[1], alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
ax.plot(imsd.index, MSD_r, 'r-', label = "Red particle")
ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$px^2$]',   
        xlabel = 'lag time $t$ [s]', title = "EMSD - Raw Trajectories")
ax.legend()
ax.grid()
if save_verb: plt.savefig("./results/mean_squared_displacement/EMSD_raw.png")
if show_verb:
    plt.show()
else:
    plt.close()
    

a = [round(fit_smooth["pw_exp_b"][0, 1], 3), round(fit_smooth["pw_exp_b"][1, 1], 3)]
b = [round(fit_smooth["pw_exp_r"][0, 1], 3), round(fit_smooth["pw_exp_r"][1, 1], 3)]
print(f"Smooth trajs - Blue Particles: {a[0]} ± {a[1]}, Red Particle: {b[0]} ± {b[1]}")
fig, ax = plt.subplots(1, 1, figsize = (10, 4))
ax.plot(imsd_smooth.index, MSD_b_smooth[0], 'b-', label = "Blue particles") 
ax.plot(imsd_smooth[1:].index, fit_smooth["fit_b"], 'b--')
ax.fill_between(imsd_smooth.index, MSD_b_smooth[0] - MSD_b_smooth[1], MSD_b_smooth[0] + MSD_b_smooth[1], 
                    alpha=0.5, edgecolor='#00FFFF', facecolor='#F0FFFF')
ax.plot(imsd_smooth.index, MSD_r_smooth, 'r-', label = "Red particle")
ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$px^2$]',   
         xlabel = 'lag time $t$ [s]', title = "EMSD - Smooth Trajectories")
ax.legend()
ax.grid()
if save_verb: plt.savefig("./results/mean_squared_displacement/EMSD_smooth.png")
if show_verb:
    plt.show()
else:
    plt.close()
## Windowed Analysis
### IMSD 
print("Windowed IMSD")
if run_windowed_analysis:
    MSD_wind, fit_wind, pw_exp_wind = get_imsd_windowed(nSteps, startFrames, endFrames, rawTrajs, pxDimension, fps, maxLagtime, nDrops)

    MSD_wind_smooth, fit_wind_smooth, pw_exp_wind_smooth = get_imsd_windowed(nSteps, startFrames, endFrames, smoothTrajs, 
                                                                            pxDimension, fps, maxLagtime, nDrops)
    if 0:
        for k in range(nSteps):
            MSD_wind[k].columns = [str(i) for i in range(nDrops)]
            MSD_wind[k].to_parquet(f"./analysis_data/MSD/raw/MSD_wind_raw{k}.parquet")
            MSD_wind_smooth[k].columns = [str(i) for i in range(nDrops)]
            MSD_wind_smooth[k].to_parquet(f"./analysis_data/MSD/smooth/MSD_wind_smooth{k}.parquet")
else:
    MSD_wind = []
    MSD_wind_smooth = []
    for k in range(nSteps):
        temp = pd.read_parquet(f"./analysis_data/MSD/raw/MSD_wind_raw{k}.parquet")
        temp.columns = [i for i in range(nDrops)]
        MSD_wind.append(temp)
        temp = pd.read_parquet(f"./analysis_data/MSD/smooth/MSD_wind_smooth{k}.parquet") 
        temp.columns = [i for i in range(nDrops)]
        MSD_wind_smooth.append(temp)
if animated_plot_verb:
    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5))
    anim_running = True

    def onClick(event):
        global anim_running
        if anim_running:
            ani.event_source.stop()
            anim_running = False
        else:
            ani.event_source.start()
            anim_running = True

    def update_graph(step):
        for i in range(nDrops):
            graphic_data[i].set_ydata(np.array(MSD_wind[step].iloc[:, i]))
        title.set_text(f"Mean Squared Displacement - Raw Trajectories - window [{startFrames[step]/10} - {endFrames[step]/10}] s")
        graph1.set_ydata(pw_exp_wind[step, :, 0, 1])
        return graphic_data, graph1,


    title = ax.set_title(f"Mean Squared Displacement - Raw Trajectories - window [{startFrames[0]/10} - {endFrames[0]/10}] s")
    graphic_data = []
    for i in range(nDrops):
        graphic_data.append(ax.plot(MSD_wind[i].index, np.array(MSD_wind[0].iloc[:, i]))[0])
    ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$px^2$]', xlabel = 'lag time $t$ [s]', ylim=(0.5, 10**5))
    ax.grid()

    ax1.set(xlabel = "Particle ID", ylabel = r"$\beta$", ylim = (0, 2))
    graph1, = ax1.plot(np.arange(nDrops), pw_exp_wind[0, :, 0, 1], '.', markersize = 10)
    ax1.axvline(x = red_particle_idx, color = 'r', linestyle = '--')

    plt.tight_layout()
    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, nSteps, interval = 5, blit=False)
    if save_verb: ani.save('./results/mean_squared_displacement/windowed_analysis/IMSD_wind_raw.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    if anim_show_verb:
        plt.show()
    else:
        plt.close()
# version 2
if animated_plot_verb:
    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5))
    anim_running = True

    def onClick(event):
        global anim_running
        if anim_running:
            ani.event_source.stop()
            anim_running = False
        else:
            ani.event_source.start()
            anim_running = True

    def update_graph(step):
        for i in range(nDrops):
            graphic_data[i].set_ydata(np.array(MSD_wind[step].iloc[:, i]))
            graphic_data2[i].set_data(startFrames[:step]/10, pw_exp_wind[:step, i, 0, 1])
        title.set_text(f"Mean Squared Displacement - Raw Trajectories - window [{startFrames[step]/10} - {endFrames[step]/10}] s")
        ax1.set_xlim(0, startFrames[step]/10 + 0.0001)
        return graphic_data, graphic_data2,


    title = ax.set_title(f"Mean Squared Displacement - Raw Trajectories - window [{startFrames[0]/10} - {endFrames[0]/10}] s")
    graphic_data = []
    for i in range(nDrops):
        if i == red_particle_idx:
            graphic_data.append(ax.plot(MSD_wind[i].index, np.array(MSD_wind[0].iloc[:, i]), color=colors[i], alpha = 1)[0])
        else:
            graphic_data.append(ax.plot(MSD_wind[i].index, np.array(MSD_wind[0].iloc[:, i]), color=colors[i], alpha = 0.3)[0])
    ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$px^2$]', xlabel = 'lag time $t$ [s]', ylim = (0.5, 10**5))
    ax.grid()

    graphic_data2 = []
    for i in range(nDrops):
        if i == red_particle_idx:
            graphic_data2.append(ax1.plot(startFrames[0]/10, pw_exp_wind[0, i, 0, 1], color=colors[i], alpha = 1)[0])
        else:
            graphic_data2.append(ax1.plot(startFrames[0]/10, pw_exp_wind[0, i, 0, 1], color=colors[i], alpha = 0.3)[0])
    ax1.set(xlabel = 'Window time [s]', ylabel = r'$\beta$', ylim = (0, 2))
    ax1.grid()

    plt.tight_layout()
    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, nSteps, interval = 5, blit=False)
    if save_verb: ani.save('./results/mean_squared_displacement/windowed_analysis/IMSD_wind_raw_v2.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    if anim_show_verb:
        plt.show()
    else:
        plt.close()
if animated_plot_verb:
    fig, (ax, ax1) = plt.subplots(2, 1, figsize = (8, 5))
    anim_running = True

    def onClick(event):
        global anim_running
        if anim_running:
            ani.event_source.stop()
            anim_running = False
        else:
            ani.event_source.start()
            anim_running = True

    def update_graph(step):
        for i in range(nDrops):
            graphic_data[i].set_ydata(np.array(MSD_wind_smooth[step].iloc[:, i]))
        title.set_text(f"Mean Squared Displacement - Smooth Trajectories - window [{startFrames[step]/10} - {endFrames[step]/10}] s")
        graph1.set_ydata(pw_exp_wind_smooth[step, :, 0, 1])
        return graphic_data, graph1,


    title = ax.set_title(f"Mean Squared Displacement - Smooth Trajectories - window [{startFrames[0]/10} - {endFrames[0]/10}] s")
    graphic_data = []
    for i in range(nDrops):
        graphic_data.append(ax.plot(MSD_wind_smooth[i].index, np.array(MSD_wind_smooth[0].iloc[:, i]))[0])
    ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$px^2$]', xlabel = 'lag time $t$ [s]', ylim=(0.05, 10**5))
    ax.grid()

    ax1.set(xlabel = "Particle ID", ylabel = r"$\beta$", ylim = (0, 2))
    graph1, = ax1.plot(np.arange(nDrops), pw_exp_wind_smooth[0, :, 0, 1], '.', markersize = 10)
    ax1.axvline(x = red_particle_idx, color = 'r', linestyle = '--')

    plt.tight_layout()
    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, nSteps, interval = 5, blit=False)
    if save_verb: ani.save('./results/mean_squared_displacement/windowed_analysis/IMSD_wind_smooth.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    if anim_show_verb:
        plt.show()
    else:
        plt.close()
if animated_plot_verb:
    # version 2
    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5))
    anim_running = True

    def onClick(event):
        global anim_running
        if anim_running:
            ani.event_source.stop()
            anim_running = False
        else:
            ani.event_source.start()
            anim_running = True

    def update_graph(step):
        for i in range(nDrops):
            graphic_data[i].set_ydata(np.array(MSD_wind_smooth[step].iloc[:, i]))
            graphic_data2[i].set_data(startFrames[:step]/10, pw_exp_wind_smooth[:step, i, 0, 1])
        title.set_text(f"Mean Squared Displacement - Smooth Trajectories - window [{startFrames[step]/10} - {endFrames[step]/10}] s")
        ax1.set_xlim(0, startFrames[step]/10 + 0.0001)
        return graphic_data, graphic_data2,


    title = ax.set_title(f"Mean Squared Displacement - Smooth Trajectories - window [{startFrames[0]/10} - {endFrames[0]/10}] s")
    graphic_data = []
    for i in range(nDrops):
        if i == red_particle_idx:
            graphic_data.append(ax.plot(MSD_wind_smooth[i].index, np.array(MSD_wind_smooth[0].iloc[:, i]), color=colors[i], alpha = 1)[0])
        else:
            graphic_data.append(ax.plot(MSD_wind_smooth[i].index, np.array(MSD_wind_smooth[0].iloc[:, i]), color=colors[i], alpha = 0.3)[0])
    ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$px^2$]', xlabel = 'lag time $t$ [s]', ylim = (0.1, 10**5))
    ax.grid()

    graphic_data2 = []
    for i in range(nDrops):
        if i == red_particle_idx:
            graphic_data2.append(ax1.plot(startFrames[0]/10, pw_exp_wind_smooth[0, i, 0, 1], color=colors[i], alpha = 1)[0])
        else:
            graphic_data2.append(ax1.plot(startFrames[0]/10, pw_exp_wind_smooth[0, i, 0, 1], color=colors[i], alpha = 0.3)[0])
    ax1.set(xlabel = 'Window time [s]', ylabel = r'$\beta$', ylim = (0, 2))
    ax1.grid()

    plt.tight_layout()
    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, nSteps, interval = 5, blit=False)
    if save_verb: ani.save('./results/mean_squared_displacement/windowed_analysis/IMSD_wind_smooth_v2.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    if anim_show_verb:
        plt.show()
    else:
        plt.close()
### EMSD 
print("Windowed EMSD")
EMSD_wind_b, EMSD_wind_r, fit_dict = get_emsd_windowed(MSD_wind, x, nDrops, red_particle_idx, nSteps, maxLagtime)
EMSD_wind_b_smooth, EMSD_wind_r_smooth, fit_dict_smooth = get_emsd_windowed(MSD_wind_smooth, x, nDrops, red_particle_idx, nSteps, maxLagtime)
if animated_plot_verb:
    Y1_msd = EMSD_wind_b[0] - EMSD_wind_b[1]
    Y2_msd = EMSD_wind_b[0] + EMSD_wind_b[1]

    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5))
    anim_running = True

    def onClick(event):
        global anim_running
        if anim_running:
            ani.event_source.stop()
            anim_running = False
        else:
            ani.event_source.start()
            anim_running = True

    def update_graph(step):
        # update title
        title.set_text(f"Mean Squared Displacement - Raw Trajectories - window {startFrames[step]/10} - {endFrames[step]/10} seconds")
        # update MSD
        graphic_data[0].set_ydata(EMSD_wind_b[0][step])
        graphic_data[1].set_ydata(EMSD_wind_r[step])
        # update fill between
        path = fill_graph.get_paths()[0]
        verts = path.vertices
        verts[1:1000+1, 1] = Y1_msd[step, :]
        verts[1000+2:-1, 1] = Y2_msd[step, :][::-1]
        # update powerlaw exponents
        line.set_data(startFrames[:step]/10, fit_dict["pw_exp_wind_b"][:step, 0, 1])
        line1.set_data(startFrames[:step]/10, fit_dict["pw_exp_wind_r"][:step, 0, 1]) 
        line2.set_data(startFrames[:step]/10, np.ones(step)) 
        ax1.set_xlim(0, (startFrames[step]+10)/10)
        return graphic_data, fill_graph, line, line1, 


    title = ax.set_title(f"Mean Squared Displacement - Raw Trajectories - window {startFrames[0]/10} - {endFrames[0]/10} seconds")
    graphic_data = []
    graphic_data.append(ax.plot(np.arange(0.1, 100.1, 0.1), EMSD_wind_b[0][0], 'b-', alpha=0.5, label = "Blue particles")[0] )
    graphic_data.append(ax.plot(np.arange(0.1, 100.1, 0.1), EMSD_wind_r[0], 'r-' , label = "Red particle")[0] )
    fill_graph = ax.fill_between(np.arange(0.1, 100.1, 0.1), Y1_msd[0], Y2_msd[0], alpha=0.5, edgecolor='#F0FFFF', facecolor='#00FFFF')
    ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$px^2$]', xlabel = 'lag time $t$ [s]', ylim=(0.5, 10**5))
    ax.legend()
    ax.grid()

    line, = ax1.plot(startFrames[0]/10, fit_dict["pw_exp_wind_b"][0, 0, 1], 'b-', alpha = 0.5, label = 'Blue particles')
    line1, = ax1.plot(startFrames[0]/10, fit_dict["pw_exp_wind_r"][0, 0, 1], 'r-', alpha = 0.5, label = 'Red particle')
    line2, = ax1.plot(startFrames[0]/10, 1, 'k-')
    ax1.legend()
    ax1.grid()
    ax1.set(xlabel = 'Window time [s]', ylabel = r'$\beta$', ylim = (0, 2))

    plt.tight_layout()
    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, nSteps, interval = 5, blit=False)
    if save_verb: ani.save('./results/mean_squared_displacement/windowed_analysis/EMSD_wind_raw.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    if anim_show_verb:
        plt.show()
    else:
        plt.close()
if animated_plot_verb:
    Y1_msd_smooth = EMSD_wind_b_smooth[0] - EMSD_wind_b_smooth[1]
    Y2_msd_smooth = EMSD_wind_b_smooth[0] + EMSD_wind_b_smooth[1]

    fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 5))
    anim_running = True

    def onClick(event):
        global anim_running
        if anim_running:
            ani.event_source.stop()
            anim_running = False
        else:
            ani.event_source.start()
            anim_running = True

    def update_graph(step):
        # update title
        title.set_text(f"Mean Squared Displacement - Smooth Trajectories - window [{startFrames[step]/10} - {endFrames[step]/10}] s")
        # update MSD
        graphic_data[0].set_ydata(EMSD_wind_b_smooth[0][step])
        graphic_data[1].set_ydata(EMSD_wind_r_smooth[step])
        # update fill between
        path = fill_graph.get_paths()[0]
        verts = path.vertices
        verts[1:1000+1, 1] = Y1_msd_smooth[step, :]
        verts[1000+2:-1, 1] = Y2_msd_smooth[step, :][::-1]
        # update powerlaw exponents
        line.set_data(startFrames[:step]/10, fit_dict_smooth["pw_exp_wind_b"][:step, 0, 1])
        line1.set_data(startFrames[:step]/10, fit_dict_smooth["pw_exp_wind_r"][:step, 0, 1])
        line2.set_data(startFrames[:step]/10, np.ones(step)) 
        ax1.set_xlim(0, (startFrames[step]+10)/10)
        return graphic_data, fill_graph, line, line1, 


    title = ax.set_title(f"Mean Squared Displacement - Smooth Trajectories - window [{startFrames[0]/10} - {endFrames[0]/10}] s")
    graphic_data = []
    graphic_data.append(ax.plot(np.arange(0.1, 100.1, 0.1), EMSD_wind_b_smooth[0][0], 'b-', alpha=0.5, label = "Blue particles")[0] )
    graphic_data.append(ax.plot(np.arange(0.1, 100.1, 0.1), EMSD_wind_r_smooth[0], 'r-' , label = "Red particle")[0] )
    fill_graph = ax.fill_between(np.arange(0.1, 100.1, 0.1), Y1_msd_smooth[0], Y2_msd_smooth[0], alpha=0.5, edgecolor='#F0FFFF', facecolor='#00FFFF')
    ax.set(xscale = 'log', yscale = 'log', ylabel = r'$\langle \Delta r^2 \rangle$ [$px^2$]', xlabel = 'lag time $t$ [s]', ylim=(0.1, 10**5))
    ax.legend()
    ax.grid()

    line, = ax1.plot(startFrames[0]/10, fit_dict_smooth["pw_exp_wind_b"][0, 0, 1], 'b-', alpha = 0.5, label = 'Blue particles')
    line1, = ax1.plot(startFrames[0]/10, fit_dict_smooth["pw_exp_wind_r"][0, 0, 1], 'r-', alpha = 0.5, label = 'Red particles')
    line2, = ax1.plot(startFrames[0]/10, 1, 'k-')
    ax1.legend()
    ax1.grid()
    ax1.set(xlabel = 'Window time [s]', ylabel = r'$\beta$', ylim = (0, 2))

    plt.tight_layout()
    fig.canvas.mpl_connect('button_press_event', onClick)
    ani = matplotlib.animation.FuncAnimation(fig, update_graph, nSteps, interval = 5, blit=False)
    if save_verb: ani.save('./results/mean_squared_displacement/windowed_analysis/EMSD_wind_smooth.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
    if anim_show_verb:
        plt.show()
    else:
        plt.close()
fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.set_title(f"Power Law Exponents - Raw Trajectories")
ax.plot(startFrames/10, fit_dict["pw_exp_wind_b"][:, 0, 1], 'b-', alpha = 0.5, label = 'blue particles')
ax.fill_between(startFrames/10, fit_dict["pw_exp_wind_b"][:, 0, 1] - fit_dict["pw_exp_wind_b"][:, 1, 1],     
                    fit_dict["pw_exp_wind_b"][:, 0, 1] + fit_dict["pw_exp_wind_b"][:, 1, 1],
                    alpha=0.5, edgecolor='#F0FFFF', facecolor='#00FFFF')
ax.plot(startFrames/10, fit_dict["pw_exp_wind_r"][:, 0, 1], 'r-', alpha = 0.5, label = 'red particle ')
ax.fill_between(startFrames/10, fit_dict["pw_exp_wind_r"][:, 0, 1] - fit_dict["pw_exp_wind_r"][:, 1, 1],
                    fit_dict["pw_exp_wind_r"][:, 0, 1] + fit_dict["pw_exp_wind_r"][:, 1, 1],
                    alpha=0.5, edgecolor='#F0FFFF', facecolor='#FF0000')
ax.plot(startFrames/10, np.ones(nSteps), 'k-')
ax.legend()
ax.grid()
ax.set(xlabel = 'Window time [s]', ylabel = r'$\beta$', ylim = (0, 2))
plt.tight_layout()
if save_verb: plt.savefig('./results/mean_squared_displacement/EMSD_beta_raw.png', )
if show_verb:
    plt.show()
else:
    plt.close()

fig, ax = plt.subplots(1, 1, figsize=(10, 4))
ax.set_title(f"Power Law Exponents - Smooth Trajectories")
ax.plot(startFrames/10, fit_dict_smooth["pw_exp_wind_b"][:, 0, 1], 'b-', alpha = 0.5, label = 'blue particles')
ax.fill_between(startFrames/10,  fit_dict_smooth["pw_exp_wind_b"][:, 0, 1] -  fit_dict_smooth["pw_exp_wind_b"][:, 1, 1],     
                     fit_dict_smooth["pw_exp_wind_b"][:, 0, 1] +  fit_dict_smooth["pw_exp_wind_b"][:, 1, 1],
                    alpha=0.5, edgecolor='#F0FFFF', facecolor='#00FFFF')
ax.plot(startFrames/10, fit_dict_smooth["pw_exp_wind_r"][:, 0, 1], 'r-', alpha = 0.5, label = 'red particle ')
ax.fill_between(startFrames/10,  fit_dict_smooth["pw_exp_wind_r"][:, 0, 1] -  fit_dict_smooth["pw_exp_wind_r"][:, 1, 1],
                        fit_dict_smooth["pw_exp_wind_r"][:, 0, 1] +  fit_dict_smooth["pw_exp_wind_r"][:, 1, 1],
                        alpha=0.5, edgecolor='#F0FFFF', facecolor='#FF0000')
ax.plot(startFrames/10, np.ones(nSteps), 'k-')
ax.legend()
ax.grid()
plt.tight_layout()
ax.set(xlabel = 'Window time [s]', ylabel = r'$\beta$', ylim = (0, 2))
if save_verb: plt.savefig('./results/mean_squared_displacement/EMSD_beta_smooth.png', )
if show_verb:
    plt.show()
else:
    plt.close()
fig, (ax,ax1) = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle(f"Generalized Diffusion Coefficients - Raw Trajectories")
ax.plot(startFrames/10, fit_dict["pw_exp_wind_b"][:, 0, 0], 'b-', alpha = 0.5, label = 'blue particles')
ax.set(xlabel = 'Window time [s]', ylabel = 'K')
ax.legend()
ax.grid()
ax1.plot(startFrames/10, fit_dict["pw_exp_wind_r"][:, 0, 0], 'r-', alpha = 0.5, label = 'red particle ')
ax1.legend()
ax1.grid()
ax1.set(xlabel = 'Window time [s]')

plt.tight_layout()
if save_verb: plt.savefig('./results/mean_squared_displacement/EMSD_D_raw.png', )
if show_verb:
    plt.show()
else:
    plt.close()

fig, (ax,ax1) = plt.subplots(1, 2, figsize=(10, 4))
plt.suptitle(f"Generalized Diffusion Coefficients - Smooth Trajectories")
ax.plot(startFrames/10, fit_dict_smooth["pw_exp_wind_b"][:, 0, 0], 'b-', alpha = 0.5, label = 'blue particles')
ax.set(xlabel = 'Window time [s]', ylabel = 'K')
ax.legend()
ax.grid()
ax1.plot(startFrames/10, fit_dict_smooth["pw_exp_wind_r"][:, 0, 0], 'r-', alpha = 0.5, label = 'red particle ')
ax1.set(xlabel = 'Window time [s]')
ax1.legend()
ax1.grid()
plt.tight_layout()

if save_verb: plt.savefig('./results/mean_squared_displacement/EMSD_D_smooth.png', )
if show_verb:
    plt.show()
else:
    plt.close()
