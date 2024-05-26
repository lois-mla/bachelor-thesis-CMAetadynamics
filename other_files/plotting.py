import matplotlib.pyplot as plt
import numpy as np



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



if __name__ == "__main__":
    plot_cvs_and_heights()