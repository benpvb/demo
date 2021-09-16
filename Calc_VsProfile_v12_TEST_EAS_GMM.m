clc; clear; close all;

addpath('D:\UC_Berkeley_study\New_Approach_for_GMPE\Program\data_');
%% Read input parameters and site amp factors
% str = 'SiteAmp_CY14_Vs760.txt';
% fid = fopen(str,'r');
% name = fgetl(fid);
% fgetl(fid);
% title_txt = fgetl(fid);
% fgetl(fid);
% Vs30 = fscanf(fid, '%f \n', [1 1]);
% fgetl(fid);
% txt = fgetl(fid);
% txt1 = textscan ( txt, '%s %s %s');
% file_refsiteamp = txt1{1};
% header = txt1{2};
% fgetl(fid);
% data = fscanf(fid, '%f %f\n', [1 2]);      %% read preliminary frequency corresponding to 30m of the Vs profile and max freq to use for site amp fit
% f30 = data(:,1);    fmax = data(:,2);
% fgetl(fid);
% bmin = fscanf(fid, '%f \n', [1 1]);      %% minimum b value in top 30m to avoid a constant velocity
% fgetl(fid);
% kappamin = fscanf(fid, '%f \n', [1 1]);      %% minimum kappa value
% fgetl(fid);
% aoi = fscanf(fid, '%f \n', [1 1]);
% fgetl(fid);
% Vs_ref  = fscanf(fid, '%f \n', [1,1]);      %% km/sec
% rho_ref  = fscanf(fid, '%f \n', [1,1]);     %% g/cm3
% fgetl(fid);
% data = fscanf(fid, '%f %f\n', [2 inf]);
% data = data';
% fclose (fid);
%% read kappa, f30, fmax and Vs30;
str = 'boore760_Vs360.txt';
dk = readtable('kappa_Vs360.csv');
kappa_ref = dk.kappa;
Vs30 = dk.Vs30;
kappamin = 0;
bmin = 0;
aoi = 0;
Vs_ref  = 3.5;
rho_ref  = 2.75;
% f30 = dk.fmin;
f30 = 2.5;
fmax = dk.fmax;
header = 'Boore1000';
name = '% WUS - Vs30 = 360 m/sec - 09/07/2020';
title_txt = 'EASGMM';
%% Coefficients of density model; 
rho1 = 2.2026;
rho2 = 0.1466;
%% frequency vector from 0.1 to 100 Hz
f_all = [ 100	97.7181	95.4883	93.3093	91.18	89.0994	87.0662	85.0794	83.138...
    81.2409	79.387	77.5754	75.8052	74.0754	72.3851	70.7333	69.1192	67.542...
    66.0007	64.4947	63.023	61.5848	60.1795	58.8062	57.4643	56.1531	54.8717...
    53.6196	52.396	51.2004	50.032	48.8903	47.7747	46.6845	45.6192	44.5782...
    43.561	42.567	41.5956	40.6465	39.7189	38.8126	37.9269	37.0614	36.2157...
    35.3893	34.5818	33.7926	33.0215	32.268	31.5317	30.8121	30.109	29.422...
    28.7506	28.0945	27.4534	26.827	26.2148	25.6166	25.032	24.4608	23.988321...
	23.442283	22.908672	22.38721	21.877611	21.37962	20.89296...
    20.41738	19.952621	19.498443	19.05461	18.62087	18.19701...
    17.782793	17.37801	16.98244	16.59587	16.218101	15.848933...
    15.48817	15.135614	14.79108	14.454392	14.12537	13.80384...
    13.489624	13.182563	12.882492	12.589251	12.302684	12.022642...
    11.748973	11.481534	11.220183	10.96478	10.715192	10.471284...
    10.23293	10	9.7723722	9.549923	9.332541	9.120107	8.912507...
    8.709635	8.5113792	8.3176364	8.1283044	7.9432821	7.762471...
    7.585776	7.413103	7.24436	7.0794563	6.9183082	6.7608284...
    6.606934	6.456542	6.309573	6.1659493	6.025596	5.8884363...
    5.7543992	5.623413	5.495409	5.3703184	5.248074	5.128613...
    5.011872	4.897787	4.7863001	4.677351	4.5708813	4.4668354...
    4.365158	4.2657952	4.168694	4.073803	3.981071	3.890451...
    3.8018932	3.715352	3.63078	3.548134	3.4673681	3.3884413...
    3.311311	3.235937	3.162278	3.090296	3.019952	2.951209...
    2.884031	2.818383	2.7542283	2.691535	2.630268	2.570396...
    2.5118863	2.454709	2.398833	2.344229	2.290868	2.2387211...
    2.187761	2.137962	2.089296	2.041738	1.9952621	1.9498444...
    1.905461	1.862087	1.819701	1.7782794	1.737801	1.698244...
    1.659587	1.62181	1.584893	1.5488164	1.513561	1.4791082...
    1.44544	1.4125374	1.3803842	1.348963	1.318257	1.28825	1.258926...
    1.230269	1.2022641	1.1748973	1.1481534	1.1220182	1.096478...
    1.0715192	1.047129	1.023293	1	0.9772371	0.9549925...
    0.93325424	0.9120108	0.8912509	0.8709636	0.8511381	0.8317637...
    0.81283044	0.79432821	0.7762471	0.7585776	0.74131023	0.72443592...
    0.7079458	0.69183093	0.676083	0.6606934	0.6456543	0.6309573...
    0.61659491	0.6025595	0.5888436	0.57543992	0.5623413	0.5495409...
    0.5370318	0.5248075	0.51286131	0.5011872	0.48977881	0.4786301...
    0.4677351	0.4570882	0.4466836	0.4365158	0.42657953	0.4168694...
    0.4073803	0.39810714	0.38904511	0.3801894	0.37153521	0.3630781...
    0.3548134	0.34673681	0.33884412	0.3311311	0.32359364	0.3162278...
    0.30902954	0.3019952	0.29512092	0.28840312	0.2818383	0.2754229...
    0.26915344	0.2630268	0.25703954	0.2511886	0.2454709	0.2398833...
    0.2344229	0.2290868	0.22387212	0.2187762	0.2137962	0.20892961...
    0.2041738	0.1995262	0.19498443	0.1905461	0.18620871	0.1819701...
    0.177828	0.1737801	0.1698244	0.1659587	0.162181	0.15848931...
    0.1548817	0.15135611	0.14791083	0.144544	0.14125373	0.13803841...
    0.1348963	0.1318257	0.12882494	0.12589254	0.1230269	0.12022643...
    0.1174898	0.1148154	0.11220184	0.10964782	0.10715192	0.1047129...
    0.1023293	0.1 ];
f_all = f_all';

%% Read relative site amp and compute total site amp
 df = readtable(['SiteAmp_EASGMM_Vs',num2str(Vs30),'.csv']);
 [f_rel,id] = sort(df.freq,'descend');
 A_rel = df.RelativeSiteAmp(id);
%% Read reference site amp
% T = readtable(char(file_refsiteamp), 'Sheet', 'SiteAmp');
f_ref = df.freq;
% col = T.Properties.VariableNames;
% ind = find(strcmp(col,header ));
A_ref = df.SR1000;
%% Multiply ref site amp with kappa operator
A_ref_k = A_ref .* exp(-pi * kappa_ref * f_ref );
%%
A_ref_k_1 = interpolate(f_ref, A_ref_k, f_rel, 3);
A_total = A_rel .* A_ref_k_1;
f_initial = f_rel;
%% 
MN_0 = 10;
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
while ( flag == 0 )
    A_initial = smooth(A_total,MN_0,'lowess');
    A_initial (A_initial < 0.0 ) = 0;
    plot( f_initial, A_initial, '--r', 'Linewidth', 2);
    plot(f_initial(f_initial>=f30 & f_initial<=fmax), A_total(f_initial>=f30 & f_initial<=fmax), '-g', 'Linewidth', 2);
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

fig_name = strcat (extractBefore(str,'.'), '_TotSiteAmp_.jpg');
saveas (figure1, ['D:\UC_Berkeley_study\New_Approach_for_GMPE\Program\figures\',fig_name]);  
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
[A_fitted, a_fit, b_fit, kappa, f30_calc, Vs30_fit ] = SiteAmp_Fit ( f_fit, A_fit, Vs30, 1.742 , 0.2875, Vs_ref, rho_ref, f0, bmin, kappamin );
%[A_fitted, a_fit, b_fit, kappa, f30_calc, Vs30_fit ] = SiteAmp_Fit_Vs30 (f_fit, A_fit, Vs30, 1.742 , 0.2875, Vs_ref, rho_ref, f0, 0, 0  );        % with Vs30 unconstrained
    fprintf( 'Freq (Hz) corresponding to 30m of profile, initial and calc: ');
    disp([f30, f30_calc]);
    
%% Correct site amp to remove kappa
ind = find ( f0 >= f30 );
A_mod  = A_fitted(ind);   
A_mod = vertcat (A_mod, A_initial( f_initial < f30));    
A_kcorr = A_mod ./ exp(-pi * kappa * f0 );

%% additional code
A_orig = A_initial ./ exp(-pi * kappa * f0 );
slm = slmengine(log(f0),A_orig,'degree',3,'knots',6,'plot','off');
A_smth = abs(slmeval(log(f0), slm));
%%
MN=40;
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
plot( f0, A_kcorr, '--r', 'Linewidth', 2);
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
fig_name = strcat (extractBefore(str,'.'),'_TotSiteAmp_kappafit.jpg');
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

%% Fit a series of power laws to the computed Vs profile. The Vs profile fit
%% will match given Vs30 and reference Vs at depth
[z1_out, Vs1_out, z1, Vs1 ,nfits,z_brk,a,b,c] = VsProfile_Fit_v1(z, Vs, Vs30, Vs30_calc, str, name, header, length(f0) );
rho1_out = 1.742 + 0.2875*Vs1_out; 
%% trial
% z1 = z;
% Vs1 = Vs;
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

%% Calculate Z1.0 and Z2.5
[~, ind] = unique(Vs);
Z1p0_init = interp1( Vs(ind) , z(ind) , 1.0 );
if ( Z1p0_init > z(length(f0)) );     Z1p0_init = NaN; end
Z2p5_init = interp1( Vs(ind) , z(ind) , 2.5 );
if ( Z2p5_init > z(length(f0)) );     Z2p5_init = NaN; end
[~, ind] = unique(Vs2);
Z1p0_final = interp1( Vs2(ind) , z2(ind) , 1.0 );
Z2p5_final = interp1( Vs2(ind) , z2(ind) , 2.5 );

Z1p0_ASK14 = 1 / 1000 * exp( -7.67/4 * log( (Vs30^4 + 610^4 ) / (1360^4 + 610^4 ) ) ) ;
Z1p0_CY14 = 1 / 1000 * exp( -7.15/4 * log( (Vs30^4 + 571^4 ) / (1360^4 + 571^4 ) ) ) ;
Z1p0 = ( Z1p0_ASK14 + Z1p0_CY14 ) / 2;
Z2p5 = exp( 7.089 - 1.144*log(Vs30) ) ;

%% 3. Plot inverted Vs profile and Vs fit 
figure1 = figure('Color',[1 1 1],'PaperPosition',[1.25 1.5 5.0 6.0]);
axes1 = axes(...
    'XGrid','on',...
    'YGrid','on',...
    'ydir','reverse',...
    'ylim', [0 z_tot], ...
    'Parent',figure1);
hold on;
plot( Vs(1:length(f0)), z(1:length(f0)),  '.-k' );
% plot( Vs1_out, z1_out, 'o-m');
plot( Vs2, z2, '.-r','Markersize',10);
legend( {['Inverted Profile - Vs30 = ', num2str(Vs30_calc, '%.0f'), 'm/sec', newline, 'Z1.0 = ', num2str(Z1p0_init,'%.3f'), ' , Z2.5 = ', num2str(Z2p5_init,'%.3f')], ...
    ['Fit, Z1.0 = ', num2str(Z1p0_final,'%.3f'), ' , Z2.5 = ', num2str(Z2p5_final,'%.3f')]}, 'location', 'SouthWest' );
title( strcat('Vs30 = ', num2str(Vs30), ' m/sec', ' - Z1.0 = ', num2str(Z1p0,'%.3f'), ' , Z2.5 = ', num2str(Z2p5,'%.3f'), ' km' ));
% Label axes
xlabel ('Vs (km/sec)');
ylabel ('Depth (km)');
grid on;
box on;
fig_name = strcat (extractBefore(str,'.'),'_Vs.jpg');
saveas (figure1, ['D:\UC_Berkeley_study\New_Approach_for_GMPE\Program\figures\',fig_name]);

%% 4. Plot site amp fit with data.
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
plot( f0, A0,  '.-b', 'Linewidth', 2);
plot( f1, A1, '-r', 'Linewidth', 2);
legend( {'Initial', 'Fit'}, 'location', 'NorthWest' );
title( strcat('Vs30 = ', num2str(Vs30), ' m/sec'));
% Label axes
xlabel ('Frequency (Hz)');
ylabel ('Site Amp');
grid on;
box on;
fig_name = strcat (extractBefore(str,'.'),'_Comp_SiteAmp.jpg');
saveas (figure1, ['D:\UC_Berkeley_study\New_Approach_for_GMPE\Program\figures\',fig_name]);

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
fig_name = strcat (extractBefore(str,'.'), '_Comp_SiteAmpwithk.jpg');
saveas (figure1, ['D:\UC_Berkeley_study\New_Approach_for_GMPE\Program\figures\',fig_name]);

%% 6. Plot relative site amp comparison inclusing kappa
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
plot( f_rel, A_rel,  '.-k' );
A_ref_k_2 = interpolate(f_ref, A_ref_k, f1, 3);
A_rel_calc = (A1 .* exp(-pi*kappa*f1')) ./ A_ref_k_2';
plot( f1, A_rel_calc, '-r', 'Linewidth', 2 );
legend( {'Initial', 'Calculated'}, 'location', 'SouthWest' );
title( strcat('Vs30 = ', num2str(Vs30), ' m/sec'));
% Label axes
xlabel ('Frequency (Hz)');
ylabel ('Relative Site Amp');
grid on;
box on;
fig_name = strcat (extractBefore(str,'.'), '_Comp_RelativeSiteAmpwithk.jpg');
saveas (figure1, ['D:\UC_Berkeley_study\New_Approach_for_GMPE\Program\figures\',fig_name]);

%% Compute MSE
[temp, ind, ib] = intersect (f1, f_initial,'stable');
A_final = A1 .* exp(-pi*kappa*f1');
A_final = A_final(ind)';
N = length(A_final);
SE_smooth = sum ( ( A_initial - A_final ) .^2 );
SE_0 = sum ( ( A_total - A_final ) .^2 );
disp ( SE_smooth) ;
disp ( SE_0 );


%% Write Output File
k = 1;
for i = 1: length(z)
    if ( i == 1 )
        z_out(k) =  0;
        Vs_out(k) = Vs2(i);
        rho_out(k) = rho2(i);
        
        z_out(k+1) =  z2(i);
        Vs_out(k+1) = Vs2(i);
        rho_out(k+1) = rho2(i);     
        
        z_out(k+2) =  z2(i);
        Vs_out(k+2) = Vs2(i+1);
        rho_out(k+2) = rho2(i+1);           
        k = k + 3;
    elseif ( i == length(z) )
        z_out(k) =  z2(i);
        Vs_out(k) = Vs2(i);
        rho_out(k) = rho2(i);        
    else
        z_out(k) =  z2(i);
        Vs_out(k) = Vs2(i);
        rho_out(k) = rho2(i);
        
        z_out(k+1) =  z2(i);
        Vs_out(k+1) = Vs2(i+1);
        rho_out(k+1) = rho2(i+1);    
        k = k + 2; 
    end
end
fileout = strcat (extractBefore(str,'.'), '_', char(header), '_out.csv');
x= [z_out;Vs_out;rho_out];
fid = fopen(fileout,'w');
fprintf(fid,'Program Version: %s\n',mfilename);
fprintf(fid,'Input file: %s\n',str);
fprintf(fid,'%s \n',name);
fprintf(fid,'Angle of incidence at source level =, %f \n',aoi);
fprintf(fid,'Source velocity =, %f, km/sec\n',Vs_ref);
fprintf(fid,'Source density =, %f, g/cm3 \n',rho_ref);
fprintf(fid,'Inverted Kappa =, %f, sec \n',kappa);
fprintf(fid,'Inverted a, b, Vs30 for Vs power law in top 30m of profile =, %f,%f,%f\n',a_fit, b_fit, Vs30_fit);
fprintf(fid,'Maximum depth for the model (this is depth corresponding to the quarter wavelength frequency for the smallest given frequency) =, %f, km \n',z_tot);
fprintf(fid,'Given Vs30 =, %f, m/sec \n',Vs30);
fprintf(fid,'Estimated Z1.0 and Z2.5 for given Vs30 =, %.3f, %.3f, km  \n',Z1p0, Z2p5);
fprintf(fid,'Initially Calculated Vs30 =, %f, m/sec \n',Vs30_calc);
fprintf(fid,'Initial Z1.0 and Z2.5 for inverted profile =, %.3f, %.3f, km  \n',Z1p0_init, Z2p5_init);
fprintf(fid,'Final Z1.0 and Z2.5 for inverted profile =, %.3f, %.3f, km  \n',Z1p0_final, Z2p5_final);

fprintf(fid,'z(km), Vs(km/sec), rho(g/cc),  , freq_given, amp_given, traveltime, thck, Vs_layer, z, Vs, Vs_fit, rho, avgVs, avgrho,  ,z, Vs, rho, traveltime, thck, Vs_layer, avgVs, avgrho, freq, amp\n');
nn=max(length(x),length(f));
out = cell(24,nn);
out(1:3, 1:length(x))=num2cell(x);
y = [f'; A'; tt; thck; Vs_layer; z; Vs; Vs1_out; rho; Vs_avg; rho_avg;...
    z2; Vs2; rho2; tt2; thck2; Vs_layer_2; Vs_avg_2; rho_avg_2; f1'; A1];
out(4:24,1:length(f1))=num2cell(y);
fprintf(fid,'%f, %f, %f,   ,%f, %f, %f, %f, %f, %f, %f, %f, %f, %f, %f,   ,%f, %f, %f, %f, %f, %f, %f, %f, %f, %f \r\n', out{:});

fclose all;