# This program provides an interactive data fitting application
# for one-dimensional (x, y) data using Gaussian or Lorentzian functions.
# It is a simplified, standalone version of the fitting functionality found in the original App.py.

# Required libraries and versions (as specified in original file)
# Matplotlib Version: 3.10.3
# PyQt5 Version: 5.15.10
# Pandas Version: 2.2.3
# NumPy Version: 2.2.5
# Scipy Version: 1.15.3

import re
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import wofz
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QLineEdit,
    QLabel,
    QPushButton,
    QMessageBox,
    QDialog,
    QSplitter,
    QDialogButtonBox,
    QTextEdit,
    QFileDialog,
    QComboBox,
    QAction,
    QMenu
)
from PyQt5.QtGui import QFont, QIcon, QDoubleValidator
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


# --- Core Fitting Functions with Baseline Offset ---

def gaussian(x, amp, pos, fwhm, baseline):
    """
    Calculates the value of a Gaussian function with a baseline offset.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))  # Convert FWHM to sigma
    return amp * np.exp(-(x - pos) ** 2 / (2 * sigma ** 2)) + baseline


def multi_gaussian(x, *params):
    """
    Calculates the sum of multiple Gaussian functions with a single, global baseline.
    The baseline is the last parameter in the flattened list.
    """
    y = np.zeros_like(x)
    baseline = params [ -1 ]
    peak_params = params [ :-1 ]

    for i in range(0, len(peak_params), 3):  # 3 parameters per Gaussian peak
        amp, pos, fwhm = peak_params [ i:i + 3 ]
        y += gaussian(x, amp, pos, fwhm, 0)  # Add each peak relative to zero

    return y + baseline  # Add the single baseline offset at the end


def lorentzian(x, amplitude, mean, fwhm, baseline):
    """
    1D Lorentzian function for fitting with a baseline offset.
    fwhm: Full Width at Half Maximum
    """
    gamma = fwhm / 2.0
    return amplitude * (gamma ** 2 / ((x - mean) ** 2 + gamma ** 2)) + baseline


def multi_lorentzian(x, *params):
    """
    Sum of multiple 1D Lorentzian functions with a single, global baseline.
    The baseline is the last parameter in the flattened list.
    """
    y_sum = np.zeros_like(x, dtype = float)
    baseline = params [ -1 ]
    peak_params = params [ :-1 ]

    for i in range(0, len(peak_params), 3):  # 3 parameters per Lorentzian peak
        amplitude, mean, fwhm = peak_params [ i:i + 3 ]
        y_sum += lorentzian(x, amplitude, mean, fwhm, 0)  # Add each peak relative to zero

    return y_sum + baseline  # Add the single baseline offset at the end


def voigt(x, amp, pos, fwhm, eta, baseline):
    """
    Voigt profile function (real part of the Faddeeva function).
    fwhm_g and fwhm_l are the fwhm for the gaussian and lorentzian part.
    eta is the mixing parameter.
    """
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm / 2.0

    z = ((x - pos) + 1j * gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi)) + baseline


def multi_voigt(x, *params):
    """
    Sum of multiple Voigt functions with a single, global baseline.
    """
    y = np.zeros_like(x)
    baseline = params [ -1 ]
    peak_params = params [ :-1 ]

    for i in range(0, len(peak_params), 4):  # 4 parameters per Voigt peak
        amp, pos, fwhm, eta = peak_params [ i:i + 4 ]
        y += voigt(x, amp, pos, fwhm, eta, 0)

    return y + baseline


def pseudovoigt(x, amp, pos, fwhm, eta, baseline):
    """
    Pseudo-Voigt profile function (linear combination of Gaussian and Lorentzian).
    eta is the mixing parameter.
    """
    # Normalized Gaussian and Lorentzian functions
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    gamma = fwhm / 2.0

    gaussian_part = np.exp(-(x - pos) ** 2 / (2 * sigma ** 2))
    lorentzian_part = (gamma ** 2 / ((x - pos) ** 2 + gamma ** 2))

    return amp * (eta * lorentzian_part + (1 - eta) * gaussian_part) + baseline


def multi_pseudovoigt(x, *params):
    """
    Sum of multiple Pseudo-Voigt functions with a single, global baseline.
    """
    y = np.zeros_like(x)
    baseline = params [ -1 ]
    peak_params = params [ :-1 ]

    for i in range(0, len(peak_params), 4):  # 4 parameters per Pseudo-Voigt peak
        amp, pos, fwhm, eta = peak_params [ i:i + 4 ]
        y += pseudovoigt(x, amp, pos, fwhm, eta, 0)

    return y + baseline


# --- Fitter Application Window (Simplified from App.py) ---

class InteractiveFitterApp(QWidget):
    """
    A standalone PyQt Widget for interactive Gaussian or Lorentzian fitting.
    Allows users to visually select initial peak guesses and fit them to 1D data.
    """

    def __init__(self, x_data, y_data, fitting_function_type="Gaussian", xlabel="X-axis", ylabel="Y-axis"):
        super( ).__init__( )
        self.x_data = x_data
        self.y_data = y_data
        self.fitting_function_type = fitting_function_type.capitalize( )
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.setWindowTitle(f"Interactive {self.fitting_function_type} Fitter")

        # State variable to control if clicks are for guessing or normal plot interaction
        self.is_guessing_mode_active = False

        # Initial baseline value is the minimum of the data
        self.baseline_offset = np.min(self.y_data)

        self.init_ui( )
        self.init_fitter_variables( )
        self.update_plot( )

    def init_ui(self):
        # Main layout for the QWidget
        self.main_layout = QVBoxLayout(self)

        # 1. Create Matplotlib Figure and Canvas
        self.fig, self.ax = plt.subplots(figsize = (15, 10))
        self.canvas = FigureCanvas(self.fig)

        # 2. Create Navigation Toolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        # 3. Create Text Display Widget
        self.params_text_edit = QTextEdit( )
        self.params_text_edit.setReadOnly(True)
        self.params_text_edit.setMinimumHeight(100)
        self.params_text_edit.setStyleSheet("font-family: Consolas; font-size: 10pt;")

        # 4. Create Control Buttons
        self.start_guess_button = QPushButton("Start Initial Guess")
        self.fit_button = QPushButton(f"Fit {self.fitting_function_type}")
        self.clear_button = QPushButton("Clear Guesses")

        # 5. Add Baseline Input Field and Label
        self.baseline_layout = QHBoxLayout( )
        self.baseline_label = QLabel("Baseline:")
        self.baseline_input = QLineEdit(self)
        self.baseline_input.setValidator(QDoubleValidator( ))
        self.baseline_input.setText(f"{self.baseline_offset:.2f}")
        self.baseline_input.setFixedWidth(100)
        self.baseline_input.editingFinished.connect(self._on_baseline_text_changed)

        self.baseline_layout.addWidget(self.baseline_label)
        self.baseline_layout.addWidget(self.baseline_input)
        self.baseline_layout.addStretch( )

        # 6. Add Fit Type Selection
        self.fit_type_layout = QHBoxLayout( )
        self.fit_type_label = QLabel("Fit Type:")
        self.fit_type_combo = QComboBox(self)
        self.fit_type_combo.addItem("Gaussian")
        self.fit_type_combo.addItem("Lorentzian")
        self.fit_type_combo.addItem("Voigt")
        self.fit_type_combo.addItem("Pseudo-Voigt")
        self.fit_type_combo.setCurrentText(self.fitting_function_type)
        self.fit_type_combo.currentIndexChanged.connect(self._on_fit_type_changed)
        self.fit_type_layout.addWidget(self.fit_type_label)
        self.fit_type_layout.addWidget(self.fit_type_combo)
        self.fit_type_layout.addStretch( )

        # 7. Create Info Label
        self.info_label = QLabel(
            f"Use the toolbar for zoom/pan. Click 'Start Initial Guess' to define {self.fitting_function_type} parameters.")
        self.info_label.setWordWrap(True)

        # 8. Create Splitter for resizable plot/text areas
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(self.canvas)
        self.splitter.addWidget(self.params_text_edit)
        self.splitter.setSizes([ 700, 300 ])

        # 9. Create Control Panel Layout
        self.control_layout = QHBoxLayout( )
        self.control_layout.addWidget(self.start_guess_button)
        self.control_layout.addWidget(self.fit_button)
        self.control_layout.addWidget(self.clear_button)
        self.control_layout.addLayout(self.fit_type_layout)
        self.control_layout.addWidget(self.info_label)
        self.control_layout.setSpacing(10)

        # 10. Assemble Main Layout
        self.main_layout.addWidget(self.toolbar)
        self.main_layout.addLayout(self.control_layout)
        self.main_layout.addLayout(self.baseline_layout)
        self.main_layout.addWidget(self.splitter)

        # 11. Connect signals
        self.start_guess_button.clicked.connect(self._toggle_guessing_mode)
        self.fit_button.clicked.connect(self.on_fit)
        self.clear_button.clicked.connect(self.on_clear_guesses)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def init_fitter_variables(self):
        """Initializes variables used for interactive fitting."""
        self.amp = None
        self.pos = None
        self.fwhm = None
        self.eta = None  # New parameter for Voigt/Pseudo-Voigt
        self.temp_line = None
        self.fixed_peaks = [ ]
        self.fitted_params = None
        self.fitted_errors = None
        self.original_xlim = self.ax.get_xlim( )
        self.original_ylim = self.ax.get_ylim( )

    def _on_baseline_text_changed(self):
        """Updates the baseline value based on the text box input."""
        try:
            new_value = float(self.baseline_input.text( ))
            self.baseline_offset = new_value
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for the baseline.")
            self.baseline_input.setText(f"{self.baseline_offset:.2f}")  # Revert to previous value
        self.update_plot( )

    def _on_fit_type_changed(self):
        """Updates the fitting function based on the combo box selection."""
        self.fitting_function_type = self.fit_type_combo.currentText( )
        self.fit_button.setText(f"Fit {self.fitting_function_type}")
        self.info_label.setText(f"Fit type changed to {self.fitting_function_type}. Please clear old guesses.")
        self.on_clear_guesses( )  # It's best to clear old guesses when changing fit type
        self.update_plot( )

    def _get_single_function(self):
        """Returns the single peak function based on the chosen type."""
        if self.fitting_function_type == "Gaussian":
            return gaussian
        elif self.fitting_function_type == "Lorentzian":
            return lorentzian
        elif self.fitting_function_type == "Voigt":
            return voigt
        elif self.fitting_function_type == "Pseudo-Voigt":
            return pseudovoigt
        else:
            raise ValueError("Invalid fitting function type selected.")

    def _get_multi_function(self):
        """Returns the multi-peak function based on the chosen type."""
        if self.fitting_function_type == "Gaussian":
            return multi_gaussian
        elif self.fitting_function_type == "Lorentzian":
            return multi_lorentzian
        elif self.fitting_function_type == "Voigt":
            return multi_voigt
        elif self.fitting_function_type == "Pseudo-Voigt":
            return multi_pseudovoigt
        else:
            raise ValueError("Invalid fitting function type selected.")

    def _toggle_guessing_mode(self):
        """Toggles the interactive guessing mode."""
        self.is_guessing_mode_active = not self.is_guessing_mode_active
        if self.is_guessing_mode_active:
            self.amp, self.pos, self.fwhm, self.temp_line = None, None, None, None
            self.start_guess_button.setText("Stop Initial Guess")
            self.info_label.setText("Guessing Mode: ON. Click on the plot for peak position and amplitude.")
        else:
            self.start_guess_button.setText("Start Initial Guess")
            self.info_label.setText(
                "Guessing Mode: OFF. Use the toolbar for zoom/pan. Click 'Start Initial Guess' to define parameters.")
            self.amp, self.pos, self.fwhm, self.temp_line = None, None, None, None
        self.update_plot( )

    def on_click(self, event):
        """Handles mouse button press events on the Matplotlib canvas."""
        if event.inaxes != self.ax or not self.is_guessing_mode_active:
            return

        if event.button == 1:  # Left click
            if self.amp is None:
                self.amp = event.ydata - self.baseline_offset
                self.pos = event.xdata
                self.fwhm = (self.x_data [ -1 ] - self.x_data [ 0 ]) / 10 if (self.x_data [ -1 ] - self.x_data [
                    0 ]) != 0 else 1.0
                # Set default eta for Voigt/Pseudo-Voigt
                if self.fitting_function_type in [ "Voigt", "Pseudo-Voigt" ]:
                    self.eta = 0.5
                else:
                    self.eta = None
                self.start_x = event.xdata
                self.info_label.setText("Drag mouse to adjust FWHM, then click again to fix.")
            else:
                if self.fitting_function_type in [ "Voigt", "Pseudo-Voigt" ]:
                    self.fixed_peaks.append((self.amp, self.pos, self.fwhm, self.eta))
                else:
                    self.fixed_peaks.append((self.amp, self.pos, self.fwhm))
                self.amp, self.pos, self.fwhm, self.eta, self.temp_line = None, None, None, None, None
                self.info_label.setText(
                    f"{self.fitting_function_type} {len(self.fixed_peaks)} fixed. Click for next, or 'Stop Initial Guess'.")
            self.update_plot( )
        elif event.button == 3:  # Right click to cancel
            if self.amp is not None:
                self.amp, self.pos, self.fwhm, self.eta, self.temp_line = None, None, None, None, None
                self.info_label.setText("Current peak selection cancelled. Click to start new guess.")
            self.update_plot( )

    def on_motion(self, event):
        """Handles mouse motion events on the Matplotlib canvas to adjust FWHM."""
        if event.inaxes != self.ax or self.amp is None or not self.is_guessing_mode_active:
            return
        self.fwhm = 2 * abs(event.xdata - self.start_x)
        if self.fwhm < 0.001: self.fwhm = 0.001
        self.update_plot( )

    def on_fit(self):
        """Performs the fitting using scipy.optimize.curve_fit."""
        if not self.fixed_peaks:
            QMessageBox.warning(self, "No Peaks",
                                f"Please fix at least one {self.fitting_function_type} guess before fitting.")
            return

        if self.is_guessing_mode_active:
            self._toggle_guessing_mode( )

        peak_params = np.array(self.fixed_peaks).flatten( )
        initial_params = np.append(peak_params, self.baseline_offset)

        # Define bounds for all parameters
        lower_bounds = [ ]
        upper_bounds = [ ]

        # Determine number of parameters per peak based on fit type
        n_params_per_peak = 3
        if self.fitting_function_type in [ "Voigt", "Pseudo-Voigt" ]:
            n_params_per_peak = 4

        for _ in range(len(self.fixed_peaks)):
            lower_bounds.extend([ -np.inf, self.x_data.min( ), 0.001 ])
            upper_bounds.extend([ np.inf, self.x_data.max( ), np.inf ])
            if n_params_per_peak == 4:
                lower_bounds.append(0.0)  # Lower bound for eta
                upper_bounds.append(1.0)  # Upper bound for eta

        # Add bounds for the single baseline parameter
        lower_bounds.append(-np.inf)
        upper_bounds.append(np.inf)

        try:
            self.fitted_params, pcov = curve_fit(self._get_multi_function( ), self.x_data, self.y_data,
                                                 p0 = initial_params,
                                                 bounds = (lower_bounds, upper_bounds))
            self.fitted_errors = np.sqrt(np.diag(pcov))
            self.update_plot( )
            self.display_fitted_parameters( )
            self.info_label.setText("Fitting complete. See fitted parameters below.")
        except RuntimeError as e:
            QMessageBox.critical(self, "Fitting Error",
                                 f"Failed to fit {self.fitting_function_type}: {e}. Adjust peaks and try again.")
            self.info_label.setText("Fitting failed. Adjust guesses and try again.")
        except ValueError as e:
            QMessageBox.critical(self, "Fitting Error", f"Fitting input error: {e}. Check data and initial parameters.")
            self.info_label.setText("Fitting failed due to input error. Check data.")

    def on_clear_guesses(self):
        """Clears all fixed peak guesses and resets the fitter state."""
        self.fixed_peaks = [ ]
        self.amp, self.pos, self.fwhm, self.eta, self.temp_line = None, None, None, None, None
        self.fitted_params = None
        self.fitted_errors = None
        self.params_text_edit.clear( )
        self.info_label.setText("All peak guesses cleared. Click 'Start Initial Guess' to define new ones.")
        self.update_plot( )

    def display_fitted_parameters(self):
        """Formats and displays the fitted parameters."""
        if self.fitted_params is None:
            self.params_text_edit.clear( )
            return

        # Calculate fit statistics
        y_fit = self._get_multi_function( )(self.x_data, *self.fitted_params)
        residuals = self.y_data - y_fit
        ss_res = np.sum(residuals ** 2)  # Sum of squared residuals
        ss_tot = np.sum((self.y_data - np.mean(self.y_data)) ** 2)  # Total sum of squares
        r_squared = 1 - (ss_res / ss_tot)

        # Number of parameters: depends on fit type
        n_params = len(self.fitted_params)
        n_data = len(self.y_data)
        dof = n_data - n_params  # Degrees of freedom

        # Chi-square and reduced chi-square (assuming no experimental errors)
        chi_square = ss_res
        reduced_chi_square = chi_square / dof if dof > 0 else np.nan

        # Extract the single fitted baseline from the end of the array
        fitted_baseline = self.fitted_params [ -1 ]
        fitted_baseline_error = self.fitted_errors [ -1 ]

        output_text = f"Fitted {self.fitting_function_type} Parameters:\n"
        output_text += "--------------------------------------------------\n"

        n_params_per_peak = 3
        if self.fitting_function_type in [ "Voigt", "Pseudo-Voigt" ]:
            n_params_per_peak = 4

        for i in range(0, len(self.fitted_params) - 1, n_params_per_peak):
            amp, pos, fwhm = self.fitted_params [ i:i + 3 ]
            amp_err, pos_err, fwhm_err = self.fitted_errors [ i:i + 3 ]

            output_text += (f"Peak {i // n_params_per_peak + 1}:\n"
                            f"  Amplitude (Amp): {amp:.4g} ± {amp_err:.2g}\n"
                            f"  Position (Pos):  {pos:.4g} ± {pos_err:.2g}\n"
                            f"  FWHM:            {fwhm:.4g} ± {fwhm_err:.2g}\n")

            if n_params_per_peak == 4:
                eta = self.fitted_params [ i + 3 ]
                eta_err = self.fitted_errors [ i + 3 ]
                output_text += f"  Eta ($\eta$): {eta:.4g} ± {eta_err:.2g}\n"

            output_text += "--------------------------------------------------\n"

        # Display the single, global baseline and its error at the end
        output_text += f"Global Baseline: {fitted_baseline:.4g} ± {fitted_baseline_error:.2g}\n"
        output_text += "--------------------------------------------------\n"

        # Display fit statistics
        output_text += "Fit Statistics:\n"
        output_text += f"  R-squared: {r_squared:.4f}\n"
        output_text += f"  Sum of Squared Residuals (RSS): {ss_res:.4g}\n"
        output_text += f"  Chi-square: {chi_square:.4g}\n"
        output_text += f"  Reduced Chi-square: {reduced_chi_square:.4g}\n"
        output_text += f"  Degrees of Freedom: {dof}\n"
        output_text += "--------------------------------------------------\n"

        self.params_text_edit.setText(output_text)

    def update_plot(self):
        """Clears the plot and redraws all elements."""
        self.ax.clear( )
        self.ax.plot(self.x_data, self.y_data, 'b-', label = 'Data')
        legend_entries = [ 'Data' ]

        if self.fitted_params is not None:
            self.ax.plot(self.x_data, self._get_multi_function( )(self.x_data, *self.fitted_params),
                         'g-', label = 'Fitted Curve', linewidth = 2)
            legend_entries.append('Fitted Curve')
            fitted_baseline = self.fitted_params [ -1 ]
            fitted_peak_params = self.fitted_params [ :-1 ]

            n_params_per_peak = 3
            if self.fitting_function_type in [ "Voigt", "Pseudo-Voigt" ]:
                n_params_per_peak = 4

            for i in range(0, len(fitted_peak_params), n_params_per_peak):
                if n_params_per_peak == 4:
                    amp, pos, fwhm, eta = fitted_peak_params [ i:i + 4 ]
                    self.ax.plot(self.x_data,
                                 self._get_single_function( )(self.x_data, amp, pos, fwhm, eta, fitted_baseline),
                                 '--', alpha = 0.7, linewidth = 1.5, label = f'Peak {i // n_params_per_peak + 1}')
                else:
                    amp, pos, fwhm = fitted_peak_params [ i:i + 3 ]
                    self.ax.plot(self.x_data,
                                 self._get_single_function( )(self.x_data, amp, pos, fwhm, fitted_baseline),
                                 '--', alpha = 0.7, linewidth = 1.5, label = f'Peak {i // n_params_per_peak + 1}')

        else:
            n_params_per_peak = 3
            if self.fitting_function_type in [ "Voigt", "Pseudo-Voigt" ]:
                n_params_per_peak = 4

            for i, peak_params in enumerate(self.fixed_peaks):
                label = f'Initial Guess {i + 1}' if i == 0 else ""

                if n_params_per_peak == 4:
                    amp, pos, fwhm, eta = peak_params
                    self.ax.plot(self.x_data,
                                 self._get_single_function( )(self.x_data, amp, pos, fwhm, eta, self.baseline_offset),
                                 'r--', alpha = 0.5, label = label)
                else:
                    amp, pos, fwhm = peak_params
                    self.ax.plot(self.x_data,
                                 self._get_single_function( )(self.x_data, amp, pos, fwhm, self.baseline_offset),
                                 'r--', alpha = 0.5, label = label)

                if i == 0: legend_entries.append('Initial Guesses')

        if self.amp is not None:
            if self.fwhm == 0: self.fwhm = 0.001

            if self.fitting_function_type in [ "Voigt", "Pseudo-Voigt" ]:
                self.temp_line, = self.ax.plot(self.x_data,
                                               self._get_single_function( )(self.x_data, self.amp, self.pos, self.fwhm,
                                                                            self.eta, self.baseline_offset),
                                               'g--', alpha = 0.5, label = 'Adjusting Width')
            else:
                self.temp_line, = self.ax.plot(self.x_data,
                                               self._get_single_function( )(self.x_data, self.amp, self.pos, self.fwhm,
                                                                            self.baseline_offset),
                                               'g--', alpha = 0.5, label = 'Adjusting Width')
            if 'Adjusting Width' not in legend_entries: legend_entries.append('Adjusting Width')

        self.ax.set_ylabel(self.ylabel, fontsize = 14)
        self.ax.set_xlabel(self.xlabel, fontsize = 14)
        self.ax.grid(True)
        self.ax.legend(loc = 'best', fontsize = 12)
        self.ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
        self.canvas.draw( )


# --- Main Window for File Import ---

class FitterLauncherApp(QMainWindow):
    """
    Main application window that handles file import and launches the
    InteractiveFitterApp for plotting and fitting 1D data.
    """

    def __init__(self):
        super( ).__init__( )
        self.setWindowTitle("Data Fitter Launcher")
        self.setGeometry(200, 200, 500, 200)

        self.central_widget = QWidget( )
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        self.info_label = QLabel("Click 'Import Data' to load a file with two columns of data.")
        self.info_label.setAlignment(Qt.AlignCenter)
        self.info_label.setFont(QFont("Arial", 12))
        self.main_layout.addWidget(self.info_label)

        self.import_button = QPushButton("Import Data")
        self.import_button.setMinimumHeight(50)
        self.import_button.setFont(QFont("Arial", 14, QFont.Bold))
        self.import_button.clicked.connect(self.on_import_button_clicked)
        self.main_layout.addWidget(self.import_button)
        self.fitter_window = None

    def on_import_button_clicked(self):
        """
        Opens a file dialog, imports the data, and launches the fitter window.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "CSV Files (*.csv);;Text Files (*.txt);;All Files (*)"
        )

        if file_path:
            try:
                # Read data assuming no header and two columns
                data = pd.read_csv(file_path, header = None, usecols = [ 0, 1 ]).values
                if data.shape [ 1 ] != 2:
                    raise ValueError("The selected file must contain exactly two columns.")

                x_data = data [ :, 0 ].astype(float)
                y_data = data [ :, 1 ].astype(float)

                if x_data.size < 3:
                    raise ValueError("Data file must contain at least 3 points for fitting.")

                # If a previous fitter window exists, close it
                if self.fitter_window:
                    self.fitter_window.close( )

                # Launch the new fitter window
                self.fitter_window = InteractiveFitterApp(x_data, y_data)
                self.fitter_window.show( )

                QMessageBox.information(self, "Success", "File imported successfully. Fitter window launched.")

            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import data: {e}")
                print(f"Error importing file: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FitterLauncherApp( )
    window.show( )
    sys.exit(app.exec_( ))
