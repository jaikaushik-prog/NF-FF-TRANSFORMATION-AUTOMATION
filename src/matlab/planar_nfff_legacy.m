% 1. Load the data 
% (importdata automatically skips the text headers and grabs the numbers)
filename = '../../data/raw/Simulated_NF_Data.txt';
imported_data = importdata(filename);
raw_matrix = imported_data.data; 

% 2. Extract the columns based on your file structure
% Col 1: x, Col 2: y, Col 4: ExRe, Col 5: ExIm, Col 6: EyRe, Col 7: EyIm
x_vals = raw_matrix(:, 1);
y_vals = raw_matrix(:, 2);
ExRe = raw_matrix(:, 4);
ExIm = raw_matrix(:, 5);
EyRe = raw_matrix(:, 6);
EyIm = raw_matrix(:, 7);

% 3. Form the complex electric field components
Ex = ExRe + 1i * ExIm;
Ey = EyRe + 1i * EyIm;

% 4. Reshape the 1D arrays back into 2D spatial grids
% X ranges from -50 to 50 in steps of 10 (11 points)
% Y ranges from -40 to 40 in steps of 10 (9 points)
Nx = 11;
Ny = 9;

% We transpose the reshape to align X to the columns and Y to the rows
Ex_2D = reshape(Ex, [Nx, Ny]).';
Ey_2D = reshape(Ey, [Nx, Ny]).';

% 5. Plot a Near-Field Sanity Check
figure;
% Plotting the magnitude in Decibels (dB)
imagesc(linspace(-50, 50, Nx), linspace(-40, 40, Ny), 20*log10(abs(Ex_2D)));
colorbar;
title('Near-Field Amplitude |Ex| (dB)');
xlabel('X (mm)');
ylabel('Y (mm)');
set(gca, 'YDir', 'normal'); % Corrects the Y-axis orientation
% --- Phase 2: NF-FF Transformation via 2D FFT ---

% 1. Define physical constants and simulation parameters
freq = 10e9;               % 10 GHz
c = 3e8;                   % Speed of light
lambda_mm = (c / freq) * 1000; % Wavelength in mm (30 mm)
dx = 10;                   % Step size X (mm)
dy = 10;                   % Step size Y (mm)

% 2. Zero-Padding 
% Your current grid is small (11x9). We pad it with zeros to a larger 
% grid (e.g., 256x256) before the FFT. This acts as mathematical interpolation
% and gives you a smooth, high-resolution Far-Field plot.
N_fft = 256;

% 3. Calculate the Plane-Wave Spectrum (PWS)
% We perform a 2D FFT on the co-polarized field (Ey).
% fftshift moves the zero-frequency (boresight) to the center of the matrix.
PWS_y = fftshift(fft2(Ey_2D, N_fft, N_fft));

% 4. Create the Angular Coordinate System (U-V Space)
% U = sin(theta)*cos(phi)
% V = sin(theta)*sin(phi)
% The maximum observable angle depends on the spatial sampling theorem.
max_u = lambda_mm / (2 * dx); 
max_v = lambda_mm / (2 * dy);

u = linspace(-max_u, max_u, N_fft);
v = linspace(-max_v, max_v, N_fft);
[U, V] = meshgrid(u, v);

% 5. Filter out Evanescent Waves
% Electromagnetic waves where U^2 + V^2 > 1 are "evanescent" (decaying exponentially)
% and do not propagate to the far-field. We mask them out.
visible_region = (U.^2 + V.^2) <= 1;

% 6. Calculate Far-Field Amplitude in dB
FarField_Ey = abs(PWS_y) .* visible_region;
FarField_dB = 20*log10(FarField_Ey / max(FarField_Ey(:))); % Normalize to 0 dB

% Drop everything outside the visible circle to -60 dB for a clean visual
FarField_dB(~visible_region) = -60;

% 7. Plot the 2D Far-Field Pattern
figure;
imagesc(u, v, FarField_dB);
axis xy image;
caxis([-40 0]); % Limit dynamic range to 40 dB below the peak
colormap('jet');
colorbar;

% Overlay a circle to show the visible region boundary
hold on;
vis_circle = exp(1i * linspace(0, 2*pi, 100));
plot(real(vis_circle), imag(vis_circle), 'w--', 'LineWidth', 1.5);
hold off;

title('Far-Field Amplitude (Ey) in U-V Space');
xlabel('U = sin(\theta)cos(\phi)');
ylabel('V = sin(\theta)sin(\phi)');
xlim([-1 1]);
ylim([-1 1]);