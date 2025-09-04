function [fy,fy_x,fy_y,quan,det_J]=INT_S(JXYe)
gp=[0.932469514203152 -0.932469514203152 0.661209386466265 -0.661209386466265...
    0.238619186083197 -0.238619186083197];
gw=[0.171324492379170 0.171324492379170 0.360761573048139 0.360761573048139...
    0.467913934572691 0.467913934572691];
kesi=gp;
ita=gp;
m=1;
fy=zeros(9,9);
fy_x=zeros(9,9);
fy_y=zeros(9,9);
quan=zeros(1,36);
det_J=zeros(1,36);
for i=1:6
    for j=1:6
        fy(:,m)=[1/4*kesi(i)*ita(j)*(kesi(i)-1)*(ita(j)-1)
            1/2*ita(j)*(ita(j)-1)*(1-kesi(i)^2)
            1/4*kesi(i)*ita(j)*(kesi(i)+1)*(ita(j)-1)
            1/2*kesi(i)*(kesi(i)-1)*(1-ita(j)^2)
            (1-kesi(i)^2)*(1-ita(j)^2)
            1/2*kesi(i)*(kesi(i)+1)*(1-ita(j)^2)
            1/4*kesi(i)*ita(j)*(kesi(i)-1)*(ita(j)+1)
            1/2*ita(j)*(ita(j)+1)*(1-kesi(i)^2)
            1/4*kesi(i)*ita(j)*(kesi(i)+1)*(ita(j)+1)];
        fy_kesi=[1/4*ita(j)*(kesi(i)-1)*(ita(j)-1)+1/4*kesi(i)*ita(j)*(ita(j)-1)
            -kesi(i)*ita(j)*(ita(j)-1)
            1/4*ita(j)*(kesi(i)+1)*(ita(j)-1)+1/4*kesi(i)*ita(j)*(ita(j)-1)
            1/2*(kesi(i)-1)*(1-ita(j)^2)+1/2*kesi(i)*(1-ita(j)^2)
            -2*kesi(i)*(1-ita(j)^2)
            1/2*(kesi(i)+1)*(1-ita(j)^2)+1/2*kesi(i)*(1-ita(j)^2)
            1/4*ita(j)*(kesi(i)-1)*(ita(j)+1)+1/4*kesi(i)*ita(j)*(ita(j)+1)
            -kesi(i)*ita(j)*(ita(j)+1)
            1/4*ita(j)*(kesi(i)+1)*(ita(j)+1)+1/4*kesi(i)*ita(j)*(ita(j)+1)];
        fy_ita=[1/4*kesi(i)*(kesi(i)-1)*(ita(j)-1)+1/4*kesi(i)*ita(j)*(kesi(i)-1)
            1/2*(ita(j)-1)*(1-kesi(i)^2)+1/2*ita(j)*(1-kesi(i)^2)
            1/4*kesi(i)*(kesi(i)+1)*(ita(j)-1)+1/4*kesi(i)*ita(j)*(kesi(i)+1)
            -kesi(i)*ita(j)*(kesi(i)-1)
            -2*ita(j)*(1-kesi(i)^2)
            -kesi(i)*ita(j)*(kesi(i)+1)
            1/4*kesi(i)*(kesi(i)-1)*(ita(j)+1)+1/4*kesi(i)*ita(j)*(kesi(i)-1)
            1/2*(ita(j)+1)*(1-kesi(i)^2)+1/2*ita(j)*(1-kesi(i)^2)
            1/4*kesi(i)*(kesi(i)+1)*(ita(j)+1)+1/4*kesi(i)*ita(j)*(kesi(i)+1)];
        dx_kesi=JXYe(:,1)'*fy_kesi;
        dx_ita=JXYe(:,1)'*fy_ita;
        dy_kesi=JXYe(:,2)'*fy_kesi;
        dy_ita=JXYe(:,2)'*fy_ita;
        J=[dx_kesi,dy_kesi;dx_ita,dy_ita];
        dif_fy=J\[fy_kesi';fy_ita'];
        fy_x(:,m)=dif_fy(1,:)';
        fy_y(:,m)=dif_fy(2,:)';
        quan(m)=gw(i)*gw(j);
        det_J(m)=det(J);
        m=m+1;
    end
end
end

