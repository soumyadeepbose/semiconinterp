% =========================================================================
% compute_Z_from_SSEC.m
% =========================================================================
% Octave/MATLAB script to compute Z-parameter (Z11) of the SSEC model
% Frequency range: 75 kHz to 43.5 GHz (log spaced)
% Saves optional CSV and plots Re(Z), Im(Z), and S11 magnitude/phase.
% =========================================================================

clc; 
clear; 
close all; 

fprintf('Compute Z_{11} from SSEC lumped parameters (one-port)\n\n');

%% --- User Inputs ---
RA = input('Enter R_A (ohms) [default 50]: ');
if isempty(RA), RA = 50; end

RD = input('Enter R_D (ohms) [default 15.5e3]: ');
if isempty(RD), RD = 15.5e3; end

LD = input('Enter L_D (henry) [default 30e-6]: ');
if isempty(LD), LD = 30e-6; end

CD = input('Enter C_D (farad) [default 2.1e-15]: ');
if isempty(CD), CD = 2.1e-15; end

RB = input('Enter R_B (ohms) [default 8e3]: ');
if isempty(RB), RB = 8e3; end

CB = input('Enter C_B (farad) [default 120e-15]: ');
if isempty(CB), CB = 120e-15; end

N = input('Enter number of frequency points (suggest 500-2000) [default 1000]: ');
if isempty(N), N = 1000; end

%% --- Frequency Vector ---
f_start = 75e3;   % 75 kHz
f_stop  = 43.5e9;  % 43.5 GHz

% Generate log-spaced column vector
f = logspace(log10(f_start), log10(f_stop), N).'; 
omega = 2 * pi * f; 

%% --- Core Network Calculations ---
% Branch Impedances
Zd = RD + 1j * omega * LD; 
Zb = RB + 1 ./ (1j * omega * CB); 

% Branch Admittances
Yd = 1 ./ Zd; 
Yb = 1 ./ Zb; 
Yc = 1j * omega * CD; 

% Total parallel admittance at Node A: Y = 1/Zd + 1/Zb + j*omega*Cd
Ytot = Yd + Yb + Yc; 

% Parallel impedance seen to ground: Zp = 1/Y
Zp = 1 ./ Ytot; 

% Total input impedance: Zin = RA + Zp
Zin = RA + Zp; 
Z11 = Zin; % For a one-port model, Z11 matches Zin

%% --- Display Sample Results ---
fprintf('\nSample results:\n');
idx_low  = 1;
idx_mid  = round(N/2);
idx_high = N;

fprintf('f_low  = %.3g Hz | Re(Zin) = %.3g, Im(Zin) = %.3g\n', f(idx_low),  real(Zin(idx_low)),  imag(Zin(idx_low)));
fprintf('f_mid  = %.3g Hz | Re(Zin) = %.3g, Im(Zin) = %.3g\n', f(idx_mid),  real(Zin(idx_mid)),  imag(Zin(idx_mid)));
fprintf('f_high = %.3g Hz | Re(Zin) = %.3g, Im(Zin) = %.3g\n', f(idx_high), real(Zin(idx_high)), imag(Zin(idx_high)));

%% --- Plotting Z-Parameters ---
figure('Name','Z_{11} (Input Impedance)','NumberTitle','off','Position',[100 100 900 600]);

subplot(2,1,1);
semilogx(f, real(Zin), 'LineWidth', 1.3);
xlabel('Frequency (Hz)');
ylabel('Re(Z_{11}) [\Omega]');
grid on;
title('Real Part of Z_{11}');

subplot(2,1,2);
semilogx(f, imag(Zin), 'LineWidth', 1.3);
xlabel('Frequency (Hz)');
ylabel('Im(Z_{11}) [\Omega]');
grid on;
title('Imaginary Part of Z_{11}');

%% --- S-Parameter Conversion ---
Z0 = input('\nEnter reference impedance Z0 for S11 conversion (default 50): ');
if isempty(Z0), Z0 = 50; end

S11 = (Zin - Z0) ./ (Zin + Z0);

figure('Name','S11 Magnitude and Phase','NumberTitle','off','Position',[120 120 900 500]);

subplot(2,1,1);
semilogx(f, 20*log10(abs(S11)), 'LineWidth', 1.3);
grid on;
xlabel('Frequency (Hz)');
ylabel('|S_{11}| (dB)');
title('S_{11} Magnitude');

subplot(2,1,2);
semilogx(f, angle(S11)*180/pi, 'LineWidth', 1.3);
grid on;
xlabel('Frequency (Hz)');
ylabel('Phase(S_{11}) (deg)');
title('S_{11} Phase');

%% --- Save Data to CSV ---
saveOpt = input('\nSave results to CSV file? (y/n): ','s');
if ~isempty(saveOpt) && lower(saveOpt) == 'y'
    outname = input('Enter filename (e.g. zin_results.csv): ','s');
    if isempty(outname), outname = 'zin_results.csv'; end
    
    T = [f, real(Zin), imag(Zin)];
    try
        writematrix(T, outname);
        fprintf('Saved results to %s\n', outname);
    catch
        % Fallback syntax for older Octave/MATLAB installations
        csvwrite(outname, T);
        fprintf('Saved results to %s (csvwrite used)\n', outname);
    end
end

fprintf('\nDone.\n');