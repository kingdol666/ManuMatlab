clc
% No clear variables to allow external inputs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%单元结点信息
Dx=L/(2*Nx);
Dy=H/(2*Ny);
E=Nx*Ny;%%%%%%%%%%%%%%%%%%%%%%单元数
Ns=(2*Nx+1)*(2*Ny+1);%%%%%%%%%二次矩形单元结点数
AAA=zeros(2*Nx+1,2*Ny+1);%%%%%单元结点分布矩阵
k=0;
for i=1:2:2*Ny+1
    for j=1:2:2*Nx+1
        k=k+1;
        AAA(i,j)=k;
    end
end
for i=1:2*Ny+1
    for j=1:2*Nx+1
        if any(1:2:2*Ny+1==i)
            if any(2:2:2*Nx==j)
                k=k+1;
                AAA(i,j)=k;
            end
        elseif any(2:2:2*Ny==i)
            k=k+1;
            AAA(i,j)=k;
        end
    end
end
JMV=zeros(E,9);%%%%%%%%%%%%%单元结点编号
k=0;
for i=1:2:2*Ny-1
    for j=1:2:2*Nx-1
        k=k+1;
        JMV(k,:)=[AAA(i,j) AAA(i,j+1) AAA(i,j+2)...
            AAA(i+1,j) AAA(i+1,j+1) AAA(i+1,j+2)...
            AAA(i+2,j) AAA(i+2,j+1) AAA(i+2,j+2)];
    end
end
JXYV=zeros(Ns,2);%%%%%%%%%%%%结点坐标
for i=1:2*Ny+1
    for j=1:2*Nx+1
        JXYV(AAA(i,j),1)=Dx*(j-1);
        JXYV(AAA(i,j),2)=Dy*(i-1);
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%边界信息
%%%%%%%%%%%%%边界结点编号
BN1=AAA(1,:);
BN2=AAA(:,2*Nx+1);
BN3=AAA(2*Ny+1,2*Nx+1:-1:1);
BN4=AAA(2*Ny+1:-1:1,1);
%%%%%%%%%%%%%
%%%%%%%%%%%%边界单元编号
BE1=(1:Nx);
BE2=(Nx:Nx:Nx*Ny)';
BE3=(Nx*Ny-Nx+1:Nx*Ny);
BE4=(1:Nx:Nx*Ny-Nx+1)';
%%%%%%%%%%%%%
%%%%%%%%%%%
BT1=[BE1',ones(size(BE1'))];
BT2=[BE2,ones(size(BE2))*2];
BT3=[BE3',ones(size(BE3'))*3];
BT4=[BE4,ones(size(BE4))*4];
%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear AAA Dx Dy BE1 BE2 BE3 BE4 
clear H L i j k Nx Ny
rectangle_grid(JMV,JXYV);
save mesh
