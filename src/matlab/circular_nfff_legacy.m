% --- 1D Circular Near-Field to Far-Field Transformation ---

% 1. Define physical constants and measurement parameters
freq = 10e9;               % 10 GHz
c = 3e8;                   % Speed of light
k = (2 * pi * freq) / c;   % Wavenumber

% IMPORTANT: Enter the exact physical distance from the rotation center of 
% your AUT to the face of your wall-mounted probe in meters.
R_probe = 0.5;             % Example: 50 cm (0.5 meters)

% --- Spatial Nyquist Criterion Check ---
% To capture all propagating modes, the maximum measureable mode (N/2) must 
% be greater than the physical cutoff (k * R_probe). If k*R > 180 (for 1-deg steps),
% we violate the spatial Nyquist limit and will experience spatial aliasing.
max_physical_mode = k * R_probe;
if max_physical_mode > 180
    warning('Spatial Nyquist Violation: R_probe is too large for 1-degree steps. You need a finer angular resolution!');
end

% 2. Load the cleaned Near-Field Data from the Python Autoencoder
filename = '../../data/processed/cleaned_sweep.csv';
% readmatrix automatically skips the CSV header row
data = readmatrix(filename);

angles_deg = data(:, 1);
mag_dB = data(:, 2);
phase_deg = data(:, 3);

% The Python autoencoder already outputs a perfect 360-degree padded sweep,
% so we no longer need the manual stepper motor interpolation or padding fix!

% 3. Convert magnitude (dB) and phase (degrees) to complex electric field
mag_linear = 10 .^ (mag_dB / 20);
phase_rad = deg2rad(phase_deg);
E_NF = mag_linear .* exp(1i * phase_rad);

% 4. Mode Extraction via 1D FFT
N = length(angles_deg);
% Use fftshift to center the spatial frequencies (modes)
% --- Absolute Scaling Fix ---
% MATLAB's fft does not scale by 1/N, but ifft scales by 1/N.
% We divide by N here to get the true Fourier series coefficients,
% preserving absolute magnitude for future Gain (dBi) calculations.
modes_NF = fftshift(fft(E_NF)) / N;

% 5. Create the mode indices (n)
% For an N-point FFT, the modes range from -N/2 to N/2
n = -floor(N/2) : floor((N-1)/2);
n = n.'; % Transpose to column vector to match data

% 6. Apply Cylindrical Mode Expansion (CME) Compensation
% Calculate the Hankel function of the 2nd kind for the near-field distance
H2_near = besselh(n, 2, k * R_probe);

% Divide out the near-field behavior and multiply by the far-field phase factor
% The (1i).^n projects the cylindrical modes to infinity
modes_FF = (modes_NF ./ H2_near) .* (1i).^n;

% --- CRITICAL FIX: Evanescent Mode Filtering ---
% For modes where |n| > k * R_probe, the Hankel function becomes extremely small.
% Dividing by it amplifies tiny numerical noise into massive spikes (the red spikes in the plot).
% These are "evanescent" (non-propagating) modes and must be filtered out.
max_mode = floor(k * R_probe); 
modes_FF(abs(n) > max_mode) = 0;

% 7. Reconstruct the Far-Field Pattern via Inverse FFT
E_FF = ifft(ifftshift(modes_FF));

% 8. Normalize and Convert back to Decibels (dB)
FF_mag_dB = 20 * log10(abs(E_FF));
FF_mag_dB = FF_mag_dB - max(FF_mag_dB); % Normalize peak to 0 dB

% 9. Plot the Results (Polar Coordinate System)
figure;

% Near-Field Plot
subplot(1,2,1);
polarplot(deg2rad(angles_deg), mag_dB - max(mag_dB), 'b', 'LineWidth', 1.5);
rlim([-40 0]); % Set dynamic range to 40 dB
title('Measured Near-Field Pattern');
set(gca, 'ThetaZeroLocation', 'top');

% Far-Field Plot
subplot(1,2,2);
polarplot(deg2rad(angles_deg), FF_mag_dB, 'r', 'LineWidth', 1.5);
rlim([-40 0]);
title('Transformed Far-Field Pattern');
set(gca, 'ThetaZeroLocation', 'top');

sgtitle('1D Circular NF-FF Transformation (10 GHz)');