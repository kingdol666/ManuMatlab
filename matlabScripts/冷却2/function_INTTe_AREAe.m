function [INTTe,AREAe]=function_INTTe_AREAe(JXYe,Te)
[fy,~,~,quan,det_J]=INT_S(JXYe);
INTTe=0;
AREAe=0;
for i=1:36
    INTTe=INTTe+quan(i)*det_J(i)*(fy(:,i)'*Te);
    AREAe=AREAe+quan(i)*det_J(i);
end
end

