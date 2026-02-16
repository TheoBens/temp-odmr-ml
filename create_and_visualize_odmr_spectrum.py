import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from Ensemble_Ellipticity_ESR_Simulator import Ensemble_Spectrum_HF
from Ensemble_Ellipticity_ESR_Simulator_V2 import Ensemble_Spectrum_HF as Ensemble_Spectrum_HF_V2

def main():
    '''
    Function to visualize ODMR spectrum from different configurations of magnetic field, MW, and frequency which are parametrable through sliders.
    The user can choose between the original simulator (Ensemble_Spectrum_HF) and the new version (Ensemble_Spectrum_HF_V2) which includes additional features.
    '''

    # ========== CONFIGURATION ==========
    # Default simulation parameters
    B_field = (0.005, 0.005, 0.005)  # Tesla
    mw_config = ((1.0, 0.2, 0.0),
                    np.deg2rad(180))  # (Ox, Oy, Oz), phase in radians
    freq_list = np.linspace(2700, 3050, 500)  # MHz
    tilt_x = 0.0  # degrees
    tilt_y = 0.0  # degrees
    use_v2 = True  # Track which simulator to use

    # ========== SIMULATION ==========
    simulator = Ensemble_Spectrum_HF_V2 if use_v2 else Ensemble_Spectrum_HF
    odmr_spectrum = simulator(
        B_field, 
        mw_config[0], 
        mw_config[1], 
        freq_list, 
        tilt_x=tilt_x, 
        tilt_y=tilt_y, 
    )

    # ========== VISUALIZATION ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(left=0.25, bottom=0.35)
    line, = ax.plot(freq_list, odmr_spectrum, label='ODMR Spectrum')
    ax.set_xlabel('Fréquence (MHz)', fontsize=12)
    ax.set_ylabel('Signal ODMR (u.a.)', fontsize=12)
    ax.set_title('ODMR Spectrum Visualization', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Checkbox for simulator selection
    ax_checkbox = plt.axes([0.05, 0.7, 0.15, 0.15])
    check = CheckButtons(ax_checkbox, ['Use V2'])
    check.set_active(0)

    # Sliders for B field components
    axcolor = 'lightgoldenrodyellow'
    ax_bx = plt.axes([0.25, 0.25, 0.65, 0.03], facecolor=axcolor)
    ax_by = plt.axes([0.25, 0.20, 0.65, 0.03], facecolor=axcolor)
    ax_bz = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
    slider_bx = Slider(ax_bx, 'Bx (T)', 0.0, 0.01, valinit=B_field[0], valstep=0.0001)
    slider_by = Slider(ax_by, 'By (T)', 0.0, 0.01, valinit=B_field[1], valstep=0.0001)
    slider_bz = Slider(ax_bz, 'Bz (T)', 0.0, 0.01, valinit=B_field[2], valstep=0.0001)

    # Slider for MW components and phase
    ax_phase = plt.axes([0.25, 0.10, 0.65, 0.03], facecolor=axcolor)
    slider_phase = Slider(ax_phase, 'MW Phase (°)', 0, 360, valinit=np.rad2deg(mw_config[1]), valstep=1)

    # Update function for sliders
    def update(val):
        nonlocal use_v2
        simulator = Ensemble_Spectrum_HF_V2 if use_v2 else Ensemble_Spectrum_HF
        Bx = slider_bx.val
        By = slider_by.val
        Bz = slider_bz.val
        phase_rad = np.deg2rad(slider_phase.val)
        new_spectrum = simulator(
            (Bx, By, Bz), 
            mw_config[0], 
            phase_rad, 
            freq_list, 
            tilt_x=tilt_x, 
            tilt_y=tilt_y, 
        )
        line.set_ydata(new_spectrum)
        # Automatically adjust both x and y axes
        ax.relim()  # Recalculate limits based on current data
        ax.autoscale_view()  # Rescale the view
        fig.canvas.draw_idle()

    # Checkbox update function
    def on_check(label):
        nonlocal use_v2
        use_v2 = not use_v2
        update(None)

    slider_bx.on_changed(update)
    slider_by.on_changed(update)
    slider_bz.on_changed(update)
    slider_phase.on_changed(update)
    check.on_clicked(on_check)

    plt.show()

if __name__ == "__main__":
    main()