function [CDBe]=function_CDBe(JXYe,JBT2e)
%%边界项方程子块计算
CDBe=zeros(9,1);
bian=JBT2e(1,2);
q1=JBT2e(1,3);
q2=JBT2e(1,4);
q3=JBT2e(1,5);
if bian==1
   q=[q1,q2,q3,0,0,0,0,0,0]';
   [fy,~,~,quan,~]=INT_S_ita(JXYe,-1);
   le=sqrt((JXYe(1,1)-JXYe(3,1))^2+(JXYe(1,2)-JXYe(3,2))^2);
   for i=1:6
       CDBe=CDBe+quan(i)*(fy(:,i)'*-q)*fy(:,i)*le/2;
   end
end
if bian==2
   q=[0,0,q1,0,0,q2,0,0,q3]';
   [fy,~,~,quan,~]=INT_S_kesi(JXYe,1);
   le=sqrt((JXYe(1,1)-JXYe(2,1))^2+(JXYe(1,2)-JXYe(2,2))^2);
   for i=1:6
       CDBe=CDBe+quan(i)*(fy(:,i)'*-q)*fy(:,i)*le/2;
   end
end
if bian==3
   q=[0,0,0,0,0,0,q3,q2,q1]';
   [fy,~,~,quan,~]=INT_S_ita(JXYe,1);
   le=sqrt((JXYe(3,1)-JXYe(1,1))^2+(JXYe(3,2)-JXYe(1,2))^2);
   for i=1:6
       CDBe=CDBe+quan(i)*(fy(:,i)'*-q)*fy(:,i)*le/2;
   end
end
if bian==4
   q=[q3,0,0,q2,0,0,q1,0,0]';
   [fy,~,~,quan,~]=INT_S_kesi(JXYe,-1);
   le=sqrt((JXYe(1,1)-JXYe(2,1))^2+(JXYe(1,2)-JXYe(2,2))^2);
   for i=1:6
       CDBe=CDBe+quan(i)*(fy(:,i)'*-q)*fy(:,i)*le/2;
   end
end
end

