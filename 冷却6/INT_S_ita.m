function [fy,fy_x,fy_y,quan,det_J]=INT_S_ita(JXYe,ita)
gp=[0.932469514203152 -0.932469514203152 0.661209386466265 -0.661209386466265...
    0.238619186083197 -0.238619186083197];
gw=[0.171324492379170 0.171324492379170 0.360761573048139 0.360761573048139...
    0.467913934572691 0.467913934572691];
kesi=gp;
m=1;
fy=zeros(9,3);
fy_x=zeros(9,3);
fy_y=zeros(9,3);
quan=zeros(1,6);
det_J=zeros(1,6);
for i=1:6
    fy(:,m)=[1/4*kesi(i)*ita*(kesi(i)-1)*(ita-1)
        1/2*ita*(ita-1)*(1-kesi(i)^2)
        1/4*kesi(i)*ita*(kesi(i)+1)*(ita-1)
        1/2*kesi(i)*(kesi(i)-1)*(1-ita^2)
        (1-kesi(i)^2)*(1-ita^2)
        1/2*kesi(i)*(kesi(i)+1)*(1-ita^2)
        1/4*kesi(i)*ita*(kesi(i)-1)*(ita+1)
        1/2*ita*(ita+1)*(1-kesi(i)^2)
        1/4*kesi(i)*ita*(kesi(i)+1)*(ita+1)];
    fy_kesi=[1/4*ita*(kesi(i)-1)*(ita-1)+1/4*kesi(i)*ita*(ita-1)
        -kesi(i)*ita*(ita-1)
        1/4*ita*(kesi(i)+1)*(ita-1)+1/4*kesi(i)*ita*(ita-1)
        1/2*(kesi(i)-1)*(1-ita^2)+1/2*kesi(i)*(1-ita^2)
        -2*kesi(i)*(1-ita^2)
        1/2*(kesi(i)+1)*(1-ita^2)+1/2*kesi(i)*(1-ita^2)
        1/4*ita*(kesi(i)-1)*(ita+1)+1/4*kesi(i)*ita*(ita+1)
        -kesi(i)*ita*(ita+1)
        1/4*ita*(kesi(i)+1)*(ita+1)+1/4*kesi(i)*ita*(ita+1)];
    fy_ita=[1/4*kesi(i)*(kesi(i)-1)*(ita-1)+1/4*kesi(i)*ita*(kesi(i)-1)
        1/2*(ita-1)*(1-kesi(i)^2)+1/2*ita*(1-kesi(i)^2)
        1/4*kesi(i)*(kesi(i)+1)*(ita-1)+1/4*kesi(i)*ita*(kesi(i)+1)
        -kesi(i)*ita*(kesi(i)-1)
        -2*ita*(1-kesi(i)^2)
        -kesi(i)*ita*(kesi(i)+1)
        1/4*kesi(i)*(kesi(i)-1)*(ita+1)+1/4*kesi(i)*ita*(kesi(i)-1)
        1/2*(ita+1)*(1-kesi(i)^2)+1/2*ita*(1-kesi(i)^2)
        1/4*kesi(i)*(kesi(i)+1)*(ita+1)+1/4*kesi(i)*ita*(kesi(i)+1)];
    dx_kesi=JXYe(:,1)'*fy_kesi;
    dx_ita=JXYe(:,1)'*fy_ita;
    dy_kesi=JXYe(:,2)'*fy_kesi;
    dy_ita=JXYe(:,2)'*fy_ita;
    J=[dx_kesi,dy_kesi;dx_ita,dy_ita];
    dif_fy=J\[fy_kesi';fy_ita'];
    fy_x(:,m)=dif_fy(1,:)';
    fy_y(:,m)=dif_fy(2,:)';
    quan(m)=gw(i);
    det_J(m)=det(J);
    m=m+1;
end
end

