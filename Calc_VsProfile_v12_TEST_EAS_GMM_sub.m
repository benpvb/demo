clc; clear; close all;
addpath('D:\UC_Berkeley_study\New_Approach_for_GMPE\Program\data_');
%% Read input parameters and site amp factors.

%% read kappa, f30, fmax and Vs30;
Vs30 = 270;
str = strcat('SiteAmp_EASGMM_Vs',num2str(Vs30),'.txt');
dk = readtable('kappa_allVs30.csv');
kappa_ref = dk.kappa_ref(1);
kappamin = 0;
bmin = 0;
aoi = 0;
Vs_ref  = 3.5;
rho_ref  = 2.75;
% f30 = dk.f30(1);
f30 = 10;
fmax = 25;

header = 'Boore1000';
name = '% pvb IQWL';
title_txt = 'EASGMM';
%% Coefficients of density model; 
rho1 = 2.2026;
rho2 = 0.1466;
%% frequency vector from 0.1 to 100 Hz
[~, ~, f_all] = Simple_EAS_GMMv6ca(5,5,1000,0,1,[]); f_all = f_all';
%% Read relative site amp and compute total site amp
 df = readtable('SiteAmp_EASGMM_allVs30.csv');
 [f_rel,id] = sort(df.freq,'descend');
  A_rel = df.(char(['ReSiteAmpResp_Vs',num2str(Vs30)]));
  A_rel = A_rel(id);
%% Read reference site amp
    f_ref = df.freq;
    A_ref = df.SR1000;
%% Multiply ref site amp with kappa operator
    A_ref_k = A_ref .* exp(-pi * kappa_ref * f_ref );
%%
A_ref_k_1 = interpolate(f_ref, A_ref_k, f_rel, 3);
A_total = A_rel .* A_ref_k_1;
f_initial = f_rel;
%% 
MN_0 = 50;
flag = 0;
%% 1. Plot total site amp and smoothed site amp
figure1 = figure('Color',[1 1 1],'PaperPosition',[1.25 1.5 5.0 4.0]);
axes1 = axes(...
    'XGrid','on',...
    'YGrid','on',...
    'xScale','log',...
    'xlim', [0.01 100], ...
    'XTickLabel',...
    {'0.01','0.1','1','10','100'}, ...
    'Parent',figure1);
hold on;
plot( f_initial, A_total, '-b', 'Linewidth', 2); 
title( strcat(title_txt, ': Vs30 = ', num2str(Vs30), ' m/sec') );
% Label axes
xlabel ('Frequency (Hz)');
ylabel ('Site Amp');
grid on;
box on;
while ( flag == 0 ) % if smoothness is not good enough, the applying smooth.
    A_initial = smooth(A_total,MN_0,'lowess');
    A_initial (A_initial < 0.0 ) = 0;
    plot( f_initial, A_initial, '--r', 'Linewidth', 2);
    plot(f_initial(f_initial>=f30 & f_initial<=fmax), A_initial(f_initial>=f30 & f_initial<=fmax), '-g', 'Linewidth', 2);
    legend( {'TotalSiteResp', strcat('TotalSiteResp - Smooth - MN = ', num2str(MN_0))}, 'location', 'SouthWest' );
    
    prompt = 'Repeat smoothing: [n]';
    repeat_ = input(prompt, 's');
    if ( isempty(repeat_) || strcmpi(repeat_,'n') == 1 )
        flag = 1;
    elseif ( strcmpi(repeat_,'y') == 1 )
        flag = 0;
        prompt = '    Enter new MN: ';
        MN_ = input(prompt, 's');
        MN_0 =  cell2mat(textscan(MN_,'%f'));
        children = get(gca, 'children');
        delete(children(1));            
    end    
    
end
A_initial = A_total;


%% Initialize f, A, and calculate tt corresponding to given Vs30
tt30 = 30 / Vs30; % travel - time from surface to 30 m.
[f_initial,I] = sort (f_initial, 'descend');    %% sort f and A to be from high to low freq
A_initial = A_initial(I);
%% Solve for kappa and Vs = a*z^b in the top 30m of profile constrained by assined Vs30
if ( max(f_initial) < 100 )
    ind = find( f_all > f_initial(1) );
    f0 = vertcat( f_all(ind), f_initial );   
else
    f0 = f_initial;
end
f_fit = f_initial( f_initial >= f30 & f_initial <= fmax );
A_fit = A_initial( f_initial >= f30 & f_initial <= fmax );

% carry out the fit using Eq.11
[A_fitted, a_fit, b_fit, kappa, f30_calc, Vs30_fit ] = SiteAmp_Fit( f_fit, A_fit, Vs30, 1.742 , 0.2875, Vs_ref, rho_ref, f0, bmin, kappamin );
%[A_fitted, a_fit, b_fit, kappa, f30_calc, Vs30_fit ] = SiteAmp_Fit_Vs30 (f_fit, A_fit, Vs30, 1.742 , 0.2875, Vs_ref, rho_ref, f0, 0, 0  );        % with Vs30 unconstrained
    fprintf( 'Freq (Hz) corresponding to 30m of profile, initial and calc: ');
    disp([f30, f30_calc]);
    
%% Correct site amp to remove kappa
ind = find ( f0 >= f30 );
A_mod  = A_fitted(ind);   
A_mod = vertcat (A_mod, A_initial( f_initial < f30));    
A_kcorr = A_mod ./ exp(-pi * kappa * f0 );
A_orig = A_initial ./ exp(-pi * kappa * f0 );
%%
MN = 70;
flag = 0;
%% 2. Plot total site amp before and after kappa correction
figure1 = figure('Color',[1 1 1],'PaperPosition',[1.25 1.5 5.0 4.0]);
axes1 = axes(...
    'XGrid','on',...
    'YGrid','on',...
    'xScale','log',...
    'xlim', [0.01 100], ...
    'XTickLabel',...
    {'0.01','0.1','1','10','100'}, ...
    'Parent',figure1);
hold on;
plot( f_initial, A_initial,  '.-k' ,'markersize',8);
plot( f0, A_mod, '-r', 'Linewidth', 2);
plot( f0, A_kcorr, '-r', 'Linewidth', 2);
% plot( f0, A_orig, '.-b', 'Linewidth', 2);
title( strcat('Vs30 = ', num2str(Vs30), ' m/sec'));
% Label axes
xlabel ('Frequency (Hz)');
ylabel ('Site Amp');
grid on;
box on;
while ( flag == 0 )       %% Apply Lowess
    A0 = smooth(A_kcorr,MN,'lowess');
    plot( f0, A0, '--b', 'Linewidth', 2);
    legend( {'Initial', 'Fit', 'Kappa and Vs30 Corr', ['Kappa and Vs30 Corr - Smooth - MN = ', num2str(MN)]}, 'location', 'Best' );
    prompt = 'Repeat smoothing: [n]';
    repeat_ = input(prompt, 's');
    if ( isempty(repeat_) || strcmpi(repeat_,'n') == 1 )
        flag = 1;
    elseif ( strcmpi(repeat_,'y') == 1 )
        flag = 0;
        prompt = '    Enter new MN: ';
        MN_ = input(prompt, 's');
        MN =  cell2mat(textscan(MN_,'%f'));
        children = get(gca, 'children');
        delete(children(1));            
    end       
end
fig_name = strcat('_TotSiteAmp_kappafit',num2str(Vs30),'_boore760.jpg');
saveas (figure1, ['D:\UC_Berkeley_study\New_Approach_for_GMPE\Program\figures\',fig_name]);  
%% Extrapolate f and A if f does not reach 0.01Hz
f = f0;
A = A0;
if ( f0(length(f0)) > 0.01 )
    n_extra = 100;    % Add 100 points to f-Amp function for extrapolation - CAN BE CHANGED
    for i = 1: n_extra
        f_temp = 0.01 * (f(length(f0))/0.01)^((i-1)/ (n_extra-1));
        f(length(f0)+n_extra-i) = f_temp;
    end
end
%% Compute inverted velocity and depth (piecewise continuous)
flag = 0;
i = 1;
while ( i <= length(f) )
   tt(i) = 1/ (4 * f(i) );    %% travel time based on quarter-wavelength assumption   
   if (i == 1) 
       %% Solve for Vs(1) and z(1)
       const = - rho_ref * Vs_ref / A(i)^2;
       coeffs = [0.2875 1.742 const];
       Vs_temp = roots(coeffs);
       ind = find(Vs_temp>0);
       if ( length(ind) > 1 )   
           stop;
       end
       Vs(i) = Vs_temp(ind);
       rho(i) = 1.742 + 0.2875*Vs(i);
       z(i) = Vs(i) / ( 4 * f(i) );
       thck(i) = z(i);
       Vs_layer(i) = Vs(i);
       
       rho_avg(i) = rho(i);
       Vs_avg(i) = Vs(i);
   elseif ( i <= length(f0) )     
       delta_tt = tt(i) - tt(i-1);
       coeffs(1) = 0.2875*delta_tt;
       coeffs(2) = 1.742*delta_tt;
       coeffs(3) = z(i-1)*rho_avg(i-1) - rho_ref*Vs_ref*tt(i)/A(i)^2;
       Vs_temp = roots(coeffs);
       ind = find(Vs_temp>0);
       if ( length(ind) > 1 )   
           stop;
       end
       Vs(i) = Vs_temp(ind);
       rho(i) = 1.742 + 0.2875*Vs(i);
       z(i) = z(i-1) + Vs(i) * delta_tt;
       if ( f(i-1) >= 5 && f(i) < 5 )
           fprintf( '\nDepth(km) corresponding to ~ 5 Hz: ');
           disp(z(i));           
       end
       if ( z(i) >= 0.03 && z(i-1) < 0.03 ) 
           %% Compute tt based on inverted profile
           tt30_calc = tt(i-1) + (tt(i)-tt(i-1)) / (z(i)-z(i-1)) * (0.03 - z(i-1));
           Vs30_calc = 30 / tt30_calc;
           diff = tt30 - tt30_calc;
           fprintf( 'Freq (Hz) corresponding to 30m of profile: ');
           disp(f(i));
           if ( abs(diff) > 1e-5 )
               i0 = i;
               flag = 1;
           end
       end
       thck(i) = z(i)-z(i-1);
       Vs_layer(i) = thck(i) / delta_tt;
       
       Vs_avg(i) = z(i) / tt(i); 
       rho_avg(i) = 1/z(i) * ( z(i-1) * rho_avg(i-1) + (z(i)-z(i-1))*rho(i) );
       if ( i == length(f0) )
           z_tot = z(i);
       end
   elseif ( i > length(f0) )
       delta_tt = tt(i) - tt(i-1);
       z(i) = z(i-1) + Vs_ref * delta_tt;
       Vs_avg(i) = z(i) / tt(i); 
       thck(i) = z(i)-z(i-1);
       Vs_layer(i) = thck(i) / delta_tt;
       Vs(i) = Vs_layer(i);
       rho(i) = 1.742 + 0.2875*Vs(i);
       rho_avg(i) = 1/z(i) * ( z(i-1) * rho_avg(i-1) + (z(i)-z(i-1))*rho(i) );
       A(i) = sqrt( (Vs_ref * rho_ref) / (Vs_avg(i)*rho_avg(i)) );
   end
   i = i + 1; 
end
%% Fit a series of power laws to the computed Vs profile. The Vs-profile fit.
%% will match given Vs30 and reference Vs at depth.
[z1_out, Vs1_out, z1, Vs1 ,nfits,z_brk,a,b,c] = VsProfile_Fit_v1(z, Vs, Vs30, Vs30_calc, str, name, header, length(f0) );
 rho1_out = 1.742 + 0.2875*Vs1_out; 
%% Adjust site amp after adjusting Vs profile using the values from the fit
for i = 1: length(z1)
    if ( i == 1 )
        Vs_layer_1(i) = Vs1(i);
        Vs_avg_1(i) = Vs_layer_1(i);
        rho1(i) = 1.742 + 0.2875*Vs1(i);        
        rho_avg_1(i) = rho1(i);
        tt1(i) = z1(i) / Vs_avg_1(i); 
    else
        Vs_layer_1(i) = Vs1(i);
        delta_z = z1(i) - z1(i-1);
        tt1(i) = tt1(i-1) + delta_z / Vs1(i);
        Vs_avg_1(i) = z1(i) / tt1(i);
        rho1(i) = 1.742 + 0.2875*Vs1(i);
        rho_avg_1(i) = 1/z1(i) * ( z1(i-1) * rho_avg_1(i-1) + (z1(i)-z1(i-1))*rho1(i) );
        
       if ( z1(i) >= 0.03 && z1(i-1) < 0.03 ) 
       %% Compute tt based on inverted profile
           tt30_calc = tt1(i-1) + (tt1(i)-tt1(i-1)) / (z1(i)-z1(i-1)) * (0.03 - z1(i-1));
           diff = tt30 - tt30_calc;
           if ( abs(diff) > 1e-4 )
               i0 = i;
               flag = 1;
           end
       end      
    end
end

f1 = f;
for i = 1: length(f1)
   tt2(i) = 1 / ( 4 * f1(i) );
   % Find location of the layer above which tt2(i) occurs
   ind = find ( tt1 < tt2(i) );
   if ( tt2(i) <= tt1(1) )
       j = 1;
   else
       j = ind(length(ind));
   end
   
   % Compute depth corresponding to tt2(i)
   delta_tt = tt2(i) - tt1(j);
   if ( j == length(Vs_layer_1) )
       delta_z = Vs_layer_1(j) * delta_tt;
   else
       delta_z = Vs_layer_1(j+1) * delta_tt;
   end
   z2(i) = z1(j) + delta_z;
   
   % Compute avgvel, avgdens
   Vs_avg_2(i) = z2(i) / tt2(i);
   if ( j == length(Vs_layer_1) )
       rho_avg_2(i) = 1/z2(i) * ( z1(j) * rho_avg_1(j) + delta_z*rho1(j) );
   else
       rho_avg_2(i) = 1/z2(i) * ( z1(j) * rho_avg_1(j) + delta_z*rho1(j+1) );
   end
   
   % Compute vel and density layer
   if ( i == 1 )
       thck2(i) = z2(i);
       Vs2(i) = Vs_avg_2(i);
       Vs_layer_2(i) = Vs2(i);
       rho2(i) = rho_avg_2(i);      
   else
       thck2(i) = z2(i) - z2(i-1);
       Vs2(i) = thck2(i) / ( tt2(i) - tt2(i-1) );
       Vs_layer_2(i) = Vs2(i);
       rho2(i) = ( z2(i) * rho_avg_2(i) - z2(i-1) * rho_avg_2(i-1) ) /...
           thck2(i);
   end
   
        A1(i) = sqrt ( ( Vs_ref * rho_ref ) / (Vs_avg_2(i)*rho_avg_2(i)) );
   
end

%% 3. Plot inverted Vs profile and Vs fit 
% dA = (A0 - A_initial); [~, imin] = min(dA);
% [~, id]= max(A0); f_pk = f(id);
figure1 = figure('Color',[1 1 1],'PaperPosition',[1.25 1.5 5.0 6.0]);
axes1 = axes(...
    'XGrid','on',...
    'YGrid','on',...
    'ydir','reverse',...
    'ylim', [0 z_tot], ...
    'Parent',figure1);
hold on;
plot( Vs(1:length(f0)), z(1:length(f0)),  '.-k' );
plot( Vs1, z1,  '.-r' );
ylim([0 2]);
legend({'Inverted Profile','power law fit'},'location','southwest');
xlabel ('Vs (km/sec)');
ylabel ('Depth (km)');
grid on;
box on;
fig_name = strcat('June5_Vs',num2str(Vs30),'_inv_Vs_pro.jpg');
saveas (figure1, ['D:\UC_Berkeley_study\New_Approach_for_GMPE\Program\figures\',fig_name]); 
%% 
tlb = table(z',f,Vs',A,thck',rho',rho_avg','VariableNames',...
    {'Z_km','freq','Vs','A','thck','rho','rho_avg'});

z30 = z(f>3.0); s1 = 1./Vs(f>3.0);
Vs30_cal = max(z30)/trapz(z30,s1);
%% 4. Plot site amp fit with data.
figure1 = figure('Color',[1 1 1],'PaperPosition',[1.25 1.5 5.0 4.0]);
axes1 = axes(...
    'XGrid','on',...
    'YGrid','on',...
    'xScale','log',...
    'yScale','log',...
    'xlim', [0.01 100], ...
    'XTickLabel',...
    {'0.01','0.1','1','10','100'}, ...
    'Parent',figure1);
hold on;
plot( f0, A0,  '.-b', 'Linewidth', 2);
plot( f1(1:length(f0)), A1(1:length(f0)), '-r', 'Linewidth', 2);
legend( {'Initial', 'Fit'}, 'location', 'NorthWest' );
title( strcat('Vs30 = ', num2str(Vs30), ' m/sec'));
axis([0.01 100 2 7]);

xlabel ('Frequency (Hz)');
ylabel ('Site Amp');
grid on;
box on;

fig_name = strcat('June5_Vs',num2str(Vs30),'_totSiteAmp_wok.jpg');
saveas (figure1, ['D:\UC_Berkeley_study\New_Approach_for_GMPE\Program\figures\',fig_name]);
%% savedata
table_array = table(f0,A0,f1(1:length(f0)),A1(1:length(f0))','VariableNames',...
     {'f0','A0','f1','A1'});
writetable(table_array,['D:\UC_Berkeley_study\New_Approach_for_GMPE\Program\data_\totSiteAmp_Vs',num2str(Vs30),'.csv'],'Delimiter',',','QuoteStrings',true);

%% 5. Plot site amp comparison inclusing kappa
figure1 = figure('Color',[1 1 1],'PaperPosition',[1.25 1.5 5.0 4.0]);
axes1 = axes(...
    'XGrid','on',...
    'YGrid','on',...
    'xScale','log',...
    'xlim', [0.01 100], ...
    'XTickLabel',...
    {'0.01','0.1','1','10','100'}, ...
    'Parent',figure1);
hold on;
plot( f_initial, A_initial,  '.-b' );
plot( f1, A1 .* exp(-pi*kappa*f1'), '-r', 'Linewidth', 2);
legend( {'Initial', 'Final'}, 'location', 'NorthWest' );
title( strcat('Vs30 = ', num2str(Vs30), ' m/sec'));
% Label axes
xlabel ('Frequency (Hz)');
ylabel ('Site Amp');
grid on;
box on;

fig_name = strcat('June5_Vs',num2str(Vs30),'_totSiteAmp_wk.jpg');
saveas (figure1, ['D:\UC_Berkeley_study\New_Approach_for_GMPE\Program\figures\',fig_name]);

%% 6. Plot relative site amp comparison inclusing kappa
% figure1 = figure('Color',[1 1 1],'PaperPosition',[1.25 1.5 5.0 4.0]);
% axes1 = axes(...
%     'XGrid','on',...
%     'YGrid','on',...
%     'xScale','log',...
%     'xlim', [0.01 100], ...
%     'XTickLabel',...
%     {'0.01','0.1','1','10','100'}, ...
%     'Parent',figure1);
% hold on;
% plot( f_rel, A_rel,  '.-k' );
% A_ref_k_2 = interpolate(f_ref, A_ref_k, f1, 3);
% A_rel_calc = (A1 .* exp(-pi*kappa*f1')) ./ A_ref_k_2';
% plot( f1, A_rel_calc, '-r', 'Linewidth', 2 );
% legend( {'Initial', 'Calculated'}, 'location', 'SouthWest' );
% title( strcat('Vs30 = ', num2str(Vs30), ' m/sec'));
% % Label axes
% xlabel ('Frequency (Hz)');
% ylabel ('Relative Site Amp');
% grid on;
% box on;
% fig_name = strcat (extractBefore(str,'.'), '_Comp_RelativeSiteAmpwithk.jpg');
% saveas (figure1, ['D:\UC_Berkeley_study\New_Approach_for_GMPE\Program\figures\',fig_name]);

%% Compute MSE
[temp, ind, ib] = intersect (f1, f_initial,'stable');
A_final = A1 .* exp(-pi*kappa*f1');
A_final = A_final(ind)';
N = length(A_final);
SE_smooth = sum ( ( A_initial - A_final ) .^2 );
SE_0 = sum ( ( A_total - A_final ) .^2 );
disp ( SE_smooth) ;
disp ( SE_0 );

