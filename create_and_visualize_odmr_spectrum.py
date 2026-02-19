import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, CheckButtons
from Ensemble_Ellipticity_ESR_Simulator_Optimized import Ensemble_Spectrum_HF
from Ensemble_Ellipticity_ESR_Simulator_V2_Optimized import Ensemble_Spectrum_HF as Ensemble_Spectrum_HF_V2

def main():
    '''
    Function to visualize ODMR spectrum from different configurations of magnetic field, MW, and frequency which are parametrable through sliders.
    The user can choose between the original simulator (Ensemble_Spectrum_HF) and the new version (Ensemble_Spectrum_HF_V2) which includes additional features.
    '''

    # ========== CONFIGURATION ==========
    # Default simulation parameters
    # B_field = (0.005, 0.005, 0.005)  # Tesla
    B_field = np.load("B_configs.npy").tolist()[0]  # Load the first B configuration as default
    mw_config = ((1.0, 0.2, 0.0), np.deg2rad(180))  # (Ox, Oy, Oz), phase in radians
    freq_list = np.linspace(2300, 3400, 300)  # MHz - Large plage pour capturer tous les pics
    tilt_x = 0.0   # degrees
    tilt_y = 0.0   # degrees
    use_v2 = False # Track which simulator to use

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
    fig, ax = plt.subplots(figsize=(14, 8))
    plt.subplots_adjust(left=0.12, bottom=0.30, right=0.95, top=0.95)
    line, = ax.plot(freq_list, odmr_spectrum, label='ODMR Spectrum')
    ax.set_xlabel('Fréquence (MHz)')
    ax.set_ylabel('Signal ODMR (u.a.)')
    ax.set_title('ODMR Spectrum Visualization')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')

    # Checkbox for simulator selection
    ax_checkbox = plt.axes([0.01, 0.80, 0.06, 0.06])
    check = CheckButtons(ax_checkbox, ['Use V2'])
    check.set_active(0)

    # Sliders for B field components
    axcolor = 'lightgoldenrodyellow'
    ax_bx = plt.axes([0.12, 0.18, 0.80, 0.02], facecolor=axcolor)
    ax_by = plt.axes([0.12, 0.15, 0.80, 0.02], facecolor=axcolor)
    ax_bz = plt.axes([0.12, 0.12, 0.80, 0.02], facecolor=axcolor)
    slider_bx = Slider(ax_bx, 'Bx (T)', 0.0, 0.01, valinit=B_field[0], valstep=0.0001)
    slider_by = Slider(ax_by, 'By (T)', 0.0, 0.01, valinit=B_field[1], valstep=0.0001)
    slider_bz = Slider(ax_bz, 'Bz (T)', 0.0, 0.01, valinit=B_field[2], valstep=0.0001)

    # Sliders for MW components
    ax_ox = plt.axes([0.12, 0.08, 0.80, 0.02], facecolor=axcolor)
    ax_oy = plt.axes([0.12, 0.05, 0.80, 0.02], facecolor=axcolor)
    slider_ox = Slider(ax_ox, 'MW Ox', 0.0, 2.0, valinit=mw_config[0][0], valstep=0.1)
    slider_oy = Slider(ax_oy, 'MW Oy', 0.0, 2.0, valinit=mw_config[0][1], valstep=0.1)

    # Slider for MW phase
    ax_phase = plt.axes([0.12, 0.01, 0.80, 0.02], facecolor=axcolor)
    slider_phase = Slider(ax_phase, 'MW Phase (°)', 0, 360, valinit=np.rad2deg(mw_config[1]), valstep=1)

    # Update function for sliders avec optimisation
    def update(val):
        nonlocal use_v2
        simulator = Ensemble_Spectrum_HF_V2 if use_v2 else Ensemble_Spectrum_HF
        Bx = slider_bx.val
        By = slider_by.val
        Bz = slider_bz.val
        Ox = slider_ox.val
        Oy = slider_oy.val
        phase_rad = np.deg2rad(slider_phase.val)
        
        # Calculer le nouveau spectre
        new_spectrum = simulator(
            (Bx, By, Bz), 
            (Ox, Oy, 0.0), 
            phase_rad, 
            freq_list, 
            tilt_x=tilt_x, 
            tilt_y=tilt_y, 
        )
        
        # Mettre à jour les données
        line.set_ydata(new_spectrum)
        
        # Détecter automatiquement la zone avec des pics significatifs
        threshold = 0.05 * new_spectrum.max()  # 5% du max
        significant_indices = np.where(new_spectrum > threshold)[0]
        
        if len(significant_indices) > 0:
            # Ajuster l'axe X pour montrer la zone avec activité + marge
            freq_min = freq_list[significant_indices[0]]
            freq_max = freq_list[significant_indices[-1]]
            freq_range = freq_max - freq_min
            margin = max(50, 0.1 * freq_range)  # Marge de 10% ou 50 MHz min
            ax.set_xlim(freq_min - margin, freq_max + margin)
        else:
            # Si pas de pics significatifs, afficher toute la plage
            ax.set_xlim(freq_list[0], freq_list[-1])
        
        # Ajuster seulement l'axe Y automatiquement
        ax.relim()
        ax.autoscale_view(scalex=False, scaley=True)
        
        fig.canvas.draw_idle()

    # Checkbox update function
    def on_check(label):
        nonlocal use_v2
        use_v2 = not use_v2
        update(None)

    slider_bx.on_changed(update)
    slider_by.on_changed(update)
    slider_bz.on_changed(update)
    slider_ox.on_changed(update)
    slider_oy.on_changed(update)
    slider_phase.on_changed(update)
    check.on_clicked(on_check)

    plt.show()

if __name__ == "__main__":
    main()