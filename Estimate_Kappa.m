clc; clear; close all;

scen.mag = [5,6,7,5,6,7,5,6,7];
scen.rrup = [5,5,5,10,10,10,15,15,15];
Ztor = 0; cm = hsv(9);


EAS_Array_Kappa = [];
fig=figure('position',[20 20 1280 480]);
subplot(1,2,1) 
for k = 1:9
    [~, EASf,f] = Simple_EAS_GMMv3(scen.mag(k),scen.rrup(k),760,Ztor,1);
    [~, SR760, v] = Linear_Site_Vs760_Response_func(f, 0, 'n');
    
    EAS_Array_Kappa = [EAS_Array_Kappa; EASf./SR760]; % devide by the linear site amp
    g(k)=semilogx(f,EASf,'-','color',cm(k,:),'LineWidth',1.5); hold on;
    h(k)=semilogx(f,EASf./SR760,'color',cm(k,:),'LineWidth',1.5);hold on;
    leg_str{k} = ['m=',num2str(scen.mag(k)),'& r',num2str(scen.rrup(k))];
end
legend(h,leg_str);    
xlabel('frequency, [hz]');
ylabel('FSA, [g*s]'); grid on;
title('EAS GMMs');

subplot(1,2,2) 
for k = 1:9  
    h(k)=semilogx(f,EAS_Array_Kappa(k,:),'color',cm(k,:),'LineWidth',1.5);hold on;
    leg_str{k} = ['m=',num2str(scen.mag(k)),'& r',num2str(scen.rrup(k))];
end
legend(h,leg_str);    
xlabel('frequency, [hz]');
ylabel('FSA, [g*s]'); grid on;
title('Modified FAS GMMs by the Vs- site Response');
% saveas(fig,'D:\UC_Berkeley_study\Latex_for_ben\figures_Vs\fig_estkappa.jpg');
%% select frequency range for determining kappa from GMMs
% select fmin and fmax
fmin = 2.5; fmax = 15;
fmin_max = f(f>=fmin & f<=fmax);
kappa_array = [];

for k = 1:9
    EAS_Array_Lin_min_max = EAS_Array_Kappa(k, f>=fmin & f<=fmax);
    mdl = fitlm(fmin_max, log(EAS_Array_Lin_min_max));
    slope = mdl.Coefficients.Estimate(2);
    kappa = slope/-pi;
    kappa_array = [kappa_array; kappa];
end


