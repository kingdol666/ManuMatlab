TT=round(t*10);
JXYV2=zeros(Ns,2*TT);

for i=1:2:2*TT
    for j=2:2:2*TT
       JXYV2(:,i)=JXYV(:,1)+(i-1)/2-0.5;
       JXYV2(:,j)=JXYV(:,2);
    end
end


weizi=2*TT;
xxxx=JXYV2(BN2,1:2:weizi)*0.1; %%%**辊上的位置
yyyy=JXYV2(BN2,2:2:weizi+1);

tttt=T(BN2,1:(weizi+1)/2);


contourf(xxxx,yyyy,tttt,20,'linestyle','none')       %%%去掉轮廓线
colorbar
xlabel('时间/秒')
ylabel('PBAT的厚度(单位m)')
% axis off
% title('Mass concentration of dichloromethane','FontSize',16)
% set (gcf,'Position',[400,100,2000,300])