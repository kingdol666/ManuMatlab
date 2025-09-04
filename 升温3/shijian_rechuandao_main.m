% clc
% clear
load mesh
%%%%%%%%%%%%%%%%%%%%%流体参数
% k=0.2;        %%%%%%导热系数，W/(m.K) 0.2387
% midu=1238;     %%密度,kg/m3
% Cv=1450;       %%%%%比热容,J/(kg.K)
% % q=-0.5;         %%%%%热流密度，W/m2
% q=0;         %%%%%热流密度，W/m2
% alpha=15;        %%%%对流换热系数，W/(m2.K)
% alpha1=15;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%初始温度
% T_kongqi=388.15;       %%%%初始空气温度，K

% T_kongqi=293.15;          %%%%初始空气温度，K
% T0=293.15;                 %薄膜初始温度                                                                              .05;            %%%%%%%%%%初始流体温度，K
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%时间迭代参数
% t=0;%%%%时间
% dt=0.1;%%%%时间步长,s
if exist('t_up_input', 'var')
    t_up = t_up_input;
    T_GunWen = T_GunWen_Input;
else
    t_up = 0.675;
    T_GunWen = 375;
end
jilu=1;
jilu_buchang=1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% T1=ones(Ns,1)*T0;
T(:,jilu)=T1;
times=0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%平均温度
INTT=0;
AREA=0;
for i=1:E
    JXYe=JXYV(JMV(i,:),:);
    Te=T1(JMV(i,:),1);
    [INTTe,AREAe]=function_INTTe_AREAe(JXYe,Te);
    AREA=AREA+AREAe;
    INTT=INTT+INTTe;
end
T_ave(times+1)=INTT/AREA;
Time(times+1)=t;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
norm_T_ave=1;
while (norm_T_ave>=0.00001 && t<=t_up)
    T_n=T1;
    T_n_k1=T_n;
    norm_T_n_k1=1;
    times1=0;
    while (norm_T_n_k1>=0.001 && times1<1000)
        T_n_k=T_n_k1;
        
        %%%温度第一边界条件
        JBT13=[BN3',T_GunWen*ones(size(BN3))'];

        JBT1=[JBT13];
%         %%%温度第二边界条件
%         JBT21=[BT1,ones(size(BT1(:,1)))*q,ones(size(BT1(:,1)))*q,ones(size(BT1(:,1)))*q];
        JBT21=[BT1,ones(size(BT1(:,1)))*0,ones(size(BT1(:,1)))*0,ones(size(BT1(:,1)))*0];
        JBT22=[BT2,ones(size(BT2(:,1)))*0,ones(size(BT2(:,1)))*0,ones(size(BT2(:,1)))*0];
        JBT23=[BT3,ones(size(BT3(:,1)))*0,ones(size(BT3(:,1)))*0,ones(size(BT3(:,1)))*0];
        JBT24=[BT4,ones(size(BT4(:,1)))*0,ones(size(BT4(:,1)))*0,ones(size(BT4(:,1)))*0];
         for i=1:length(BT1(:,1))
            danyuan=BT1(i,1);
            jiedian=[JMV(danyuan,1),JMV(danyuan,2),JMV(danyuan,3)];
            qt=alpha1*(T_n_k(jiedian,1)'-ones(1,3)*T_kongqi);
            JBT21(i,:)=[BT1(i,:),qt];
         end
          for i=1:length(BT3(:,1))
            danyuan=BT3(i,1);
            jiedian=[JMV(danyuan,9),JMV(danyuan,8),JMV(danyuan,7)];
            qt=alpha1*(T_n_k(jiedian,1)'-ones(1,3)*T_kongqi);
            JBT23(i,:)=[BT3(i,:),qt];
          end
          for i=1:length(BT4(:,1))
            danyuan=BT4(i,1);
            jiedian=[JMV(danyuan,7),JMV(danyuan,4),JMV(danyuan,1)];
            qt=alpha*(T_n_k(jiedian,1)'-ones(1,3)*T_kongqi);
            JBT24(i,:)=[BT4(i,:),qt];
         end
        JBT2=[JBT21;JBT22;JBT23;JBT24];
        M=zeros(Ns,Ns);
        CD=zeros(Ns,Ns);
        CDB=zeros(Ns,1);
        for i=1:E
            JXYe=JXYV(JMV(i,:),:);
            [Me,CDe]=function_Me_CDe(JXYe,midu,Cv,k);
            for ii=1:9
                for jj=1:9
                    M(JMV(i,ii),JMV(i,jj))=M(JMV(i,ii),JMV(i,jj))+Me(ii,jj);
                    CD(JMV(i,ii),JMV(i,jj))=CD(JMV(i,ii),JMV(i,jj))+CDe(ii,jj);
                end
            end
        end
        for i=1:length(JBT2(:,1))
            JXYe=JXYV(JMV(JBT2(i,1),:),:);
            CDBe=function_CDBe(JXYe,JBT2(i,:));
            for ii=1:9
                CDB(JMV(JBT2(i,1),ii),1)=CDB(JMV(JBT2(i,1),ii),1)+CDBe(ii,1);
            end
        end

        
        theta=0.5;
        K=M+CD*theta*dt;
        b=dt*CDB+(M-CD*(1-theta)*dt)*T_n;
        
        %%%****对角线归一代入法
        N_matrix=Ns;
        for i=1:length(JBT1(:,1))
            II=JBT1(i,1);
            temp=JBT1(i,2);
            for J=1:N_matrix
                b(J)=b(J)-K(J,II)*temp;
            end
            K(II,:)=zeros(1,N_matrix);
            K(:,II)=zeros(N_matrix,1);
            K(II,II)=1;
            b(II)=temp;
        end
        %%%****
        

        T_n_k1=K\b;
        if norm(T_n_k1-T_n_k)<1e-10
            norm_T_n_k1=0;
        else
            norm_T_n_k1=norm(T_n_k1-T_n_k)/norm(T_n_k);
        end
        times1=times1+1;  
    end
    T1=T_n_k1;
    times=times+1;
    t=t+dt;
    if mod(times,jilu_buchang)==0
        jilu=jilu+1;
        T(:,jilu)=T1;
        fprintf('times1=%d\n',times1);
        fprintf('times=%d\n',times);
        fprintf('jilu=%d\n',jilu);
        fprintf('t=%d\n\n',t);
    end
    INTT=0;
    AREA=0;
    for i=1:E
        JXYe=JXYV(JMV(i,:),:);
        Te=T1(JMV(i,:),1);
        [INTTe,AREAe]=function_INTTe_AREAe(JXYe,Te);
        AREA=AREA+AREAe;
        INTT=INTT+INTTe;
    end
    T_ave(times+1)=INTT/AREA;
    Time(times+1)=t;
    if norm(T_ave(times+1)-T_ave(times))<1e-10
        norm_T_ave=0;
    else
        norm_T_ave=norm(T_ave(times+1)-T_ave(times))/norm(T_ave(times));
    end
end
JMV4=[JMV(:,1),JMV(:,3),JMV(:,9),JMV(:,7)];
