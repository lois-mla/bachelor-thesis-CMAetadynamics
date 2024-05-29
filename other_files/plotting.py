import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns



def plot_free_energy2d():
    # plot free energy as a function of simulation time

    for i in range(7):

        # import fes file into pandas dataset
        data = np.loadtxt(f"fes_{i}.dat")

        phi = data[:, 0]
        # psi = data[:, 1]
        free_energy = data[:, 1]  

        # plot fes
        plt.plot(phi, free_energy) 

    # labels
    plt.xlabel("phi [rad]")
    plt.ylabel("free energy [kJ/mol]")
    # plt.legend(ncol=3)

    plt.savefig(f'free_energy2d_compare.png', bbox_inches='tight')


def plot_free_energy3d():
      # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # import fes file into pandas dataset
    data = np.loadtxt("fes_compare.dat")

    phi = data[:, 0]
    psi = data[:, 1]
    free_energy = data[:, 2]  

    # Plot the data points
    ax.scatter(phi, psi, free_energy, c=free_energy, cmap='viridis', marker='o')

    # Set labels and title
    ax.set_xlabel('Phi')
    ax.set_ylabel('Psi')
    ax.set_zlabel('Free Energy')
    ax.set_title('3D Plot of Phi, Psi, and Free Energy')

    # Function to update the view angle
    def update_view_angle(elev, azim):
        ax.view_init(elev=elev, azim=azim)

    # Create interactive sliders for elevation and azimuth
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax_elev = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_azim = plt.axes([0.25, 0.15, 0.65, 0.03])
    s_elev = plt.Slider(ax_elev, 'Elevation', -180, 180, valinit=30)
    s_azim = plt.Slider(ax_azim, 'Azimuth', -180, 180, valinit=30)

    # Define function to update plot when sliders change
    def update(val):
        elev = s_elev.val
        azim = s_azim.val
        update_view_angle(elev, azim)

    # Register the update function with the sliders
    s_elev.on_changed(update)
    s_azim.on_changed(update)

    plt.savefig(f'free_energy_3d_compare.png', bbox_inches='tight')

    plt.show()


    
def plot_cvs_and_heights():

    plt.figure()  # Create a new figure

    colvar_data = np.loadtxt(f"COLVAR_compare2")
    hills_data = np.loadtxt("HILLS_compare2")

    # for i, cv_label in enumerate(cvs):
    phi = colvar_data[:, 1]
    psi = colvar_data[:, 2]
    # height = hills_data[:, -2]
    phi_hills = hills_data[:, 1]
    psi_hills = hills_data[:, 2]

    # plot locations of the Gaussians
    plt.scatter(phi_hills, psi_hills, s=1, marker='x', color="k", alpha=0.25)

    # plt.scatter(phi, psi, s=1, marker='x', color="b")

    # Adding labels and legend
    plt.xlabel("phi")
    plt.ylabel("psi")

    plt.savefig(f'tingz.png', bbox_inches='tight')

    plt.close()  # Close the current figure to prevent accumulation


def plot_bias(colvar_file, save_path):

    colvar_data = np.loadtxt(colvar_file)
    # hills_data = np.loadtxt("HILLS_compare2")
    begin_index = 0
    end_index = 25000000

    # for i, cv_label in enumerate(cvs):
    phi = colvar_data[:, 1][begin_index: end_index]
    psi = colvar_data[:, 2][begin_index: end_index]
    bias = colvar_data[:, 3][begin_index: end_index]

    # Create a 2D histogram
    bins = 50
    heatmap, xedges, yedges = np.histogram2d(phi, psi, bins=bins, weights=bias)
    print(heatmap)

    # ax.figure(figsize=(10, 8))
    im = plt.imshow(heatmap.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')
    plt.xlabel('Phi')
    plt.ylabel('Psi')
        # Plot colorbar only for the first cycle
    plt.colorbar(im, label='bias')
    plt.savefig(save_path)


def plot_fes_heatmap(fes_file, save_path):

    fes_data = np.loadtxt(fes_file)

    # phi = fes_data[:, 0].unique
    # psi = fes_data[:, 1].unique
    # fes_grid = fes_data.pivot(index='psi', columns='phi', values='fes')


    # # phi_values = data['phi'].unique()
    # # psi_values = data['psi'].unique()
    # # fes_grid = data.pivot(index='psi', columns='phi', values='fes')

    # # Step 3: Create the heatmap
    # plt.figure(figsize=(10, 8))
    # sns.heatmap(fes_grid, cmap='viridis', xticklabels=phi, yticklabels=psi)


    # Step 2: Extract phi, psi, and fes
    phi = fes_data[:, 0]
    psi = fes_data[:, 1]
    fes = fes_data[:, 2]
    # Step 3: Determine the grid size and reshape
    # Assuming the grid is 50x50 and phi and psi values are sorted appropriately
    grid_size = 51

    # Reshape fes into a 50x50 grid
    fes_grid = fes.reshape((grid_size, grid_size))

    # Generate phi and psi values for labeling
    phi_values = np.unique(phi).reshape((grid_size,))
    psi_values = np.unique(psi).reshape((grid_size,))

    # Step 4: Plot the heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(fes_grid, cmap='viridis', extent=[phi_values.min(), phi_values.max(), psi_values.min(), psi_values.max()],
            origin='lower', aspect='auto')
    plt.colorbar(label='FES')


    # # Create a 2D histogram
    # bins = 50
    # heatmap, xedges, yedges = np.histogram2d(phi, psi, bins=bins, weights=fes)
    # print(heatmap)

    # # ax.figure(figsize=(10, 8))
    # im = plt.imshow(heatmap.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], cmap='hot')
    plt.xlabel('Phi')
    plt.ylabel('Psi')
        # Plot colorbar only for the first cycle
    # plt.colorbar(im, label='fes')
    plt.savefig(save_path)



if __name__ == "__main__":
    plot_fes_heatmap("fes_compare.dat", "fes")