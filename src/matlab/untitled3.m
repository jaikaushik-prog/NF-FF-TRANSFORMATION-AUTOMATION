% --- 1D Circular Near-Field to Far-Field Transformation ---

% 1. Define physical constants and measurement parameters
freq = 10e9;               % 10 GHz
c = 3e8;                   % Speed of light
k = (2 * pi * freq) / c;   % Wavenumber

% IMPORTANT: Enter the exact physical distance from the rotation center of 
% your AUT to the face of your wall-mounted probe in meters.
R_probe = 0.5;             % Example: 50 cm (0.5 meters)
% 2. Load the measured Near-Field Data
filename = 'Simulated_NF_Data.txt'; % Ensure this matches your file
data = readmatrix(filename);

angles_raw = data(:, 1);
mag_raw = data(:, 2);
phase_raw = data(:, 3);

% --- THE STEPPER MOTOR FIX: Interpolate missing degrees ---
% 1. Shift angles by 180° to prevent the 0° crossing from breaking the interpolation
shifted_angles = mod(angles_raw + 180, 360);

% 2. Sort the data strictly by the new shifted angles
[shifted_angles, sort_idx] = sort(shifted_angles);
mag_sorted = mag_raw(sort_idx);
phase_sorted = phase_raw(sort_idx);

% 3. Remove any duplicate angles caused by motor jitter
[shifted_angles, unique_idx] = unique(shifted_angles, 'stable');
mag_sorted = mag_sorted(unique_idx);
phase_sorted = phase_sorted(unique_idx);

% 4. Create a perfectly uniform 1-degree vector for the measured arc
query_shifted = floor(min(shifted_angles)) : 1 : ceil(max(shifted_angles));

% 5. Interpolate to fill the stepper motor gaps
mag_interp = interp1(shifted_angles, mag_sorted, query_shifted, 'linear');
phase_interp = interp1(shifted_angles, phase_sorted, query_shifted, 'linear');

% 6. Shift the coordinates back to reality (0° to 360°)
true_angles = mod(query_shifted - 180, 360);

% --- THE PADDING FIX: Apply to 360 Canvas ---
noise_floor_dB = -50;
mag_dB = ones(360, 1) * noise_floor_dB;
phase_deg = zeros(360, 1);

for i = 1:length(true_angles)
    % Map the interpolated angles precisely to integer indices
    idx = mod(round(true_angles(i)), 360) + 1; 
    mag_dB(idx) = mag_interp(i);
    phase_deg(idx) = phase_interp(i);
end

% 3. Convert magnitude (dB) and phase (degrees) to complex electric field
mag_linear = 10 .^ (mag_dB / 20);
phase_rad = deg2rad(phase_deg);
E_NF = mag_linear .* exp(1i * phase_rad);
% 4. Mode Extraction via 1D FFT
N = length(angles_deg);
% Use fftshift to center the spatial frequencies (modes)
modes_NF = fftshift(fft(E_NF));

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