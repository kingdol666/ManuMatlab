function [Me,CDe]=function_Me_CDe(JXYe,midu,Cv,k)
%%单元方程子块计算
Me=zeros(9,9);
CDe=zeros(9,9);
[fy,fy_x,fy_y,quan,det_J]=INT_S(JXYe);
for i=1:36
    Me=Me+midu*Cv*quan(i)*(fy(:,i)*fy(:,i)')*det_J(i);
    CDe=CDe+k*quan(i)*det_J(i)*(fy_x(:,i)*fy_x(:,i)'+fy_y(:,i)*fy_y(:,i)');
end
end

