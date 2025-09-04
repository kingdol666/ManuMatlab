function rectangle_grid(JMV,JXYV)
E=length(JMV(:,1));
NOD=length(JXYV(:,1));
axis equal
danyuanhao=1;
jiedianhao=1;
pinyix=abs(JXYV(1,1)-JXYV(2,1))/10;
pinyiy=abs(JXYV(JMV(1,1),2)-JXYV(JMV(1,7),2))/4;
hold on
JM=[JMV(:,1),JMV(:,3),JMV(:,9),JMV(:,7)];
for i=1:E
   for j=1:4
       x(j)=JXYV(JM(i,j),1);
       y(j)=JXYV(JM(i,j),2);
   end
   x(5)=x(1);
   y(5)=y(1);
   plot(x,y);
   if danyuanhao==1
       text(sum(x(1:4))/4-pinyix,sum(y(1:4))/4-pinyiy,['(',int2str(i),')'],'FontSize',10);
   end
end
if jiedianhao==1
    for i=1:NOD
        text(JXYV(i,1)-pinyix,JXYV(i,2),int2str(i),'FontSize',10);
    end
end 
axis off;
end

