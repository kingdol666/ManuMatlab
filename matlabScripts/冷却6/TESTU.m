function TESTU(LXY,DATA,T)
X=LXY(:,1);Y=LXY(:,2);
L=max(X)/2;
H=max(Y)/2;
figure1 = figure;
colormap('jet');
axes1 = axes('Parent',figure1);
% title({'C'},'Color','k','FontName','Times New Roman','FontSize',22,'FontWeight','bold');
axis(axes1,'image');
set(axes1,'BoxStyle','full','Layer','top',...
     'FontName','Times New Roman','FontSize',14,'FontWeight','bold');
% set(figure1 ,'visible','off');
set(gcf,'PaperUnits','centimeters','PaperPosition',[0 0 18 13])
colorbar('peer',axes1,'eastoutside','FontSize',14);
patch(X(LE)',Y(LE)',DATA(LE)','EdgeColor','none');
% patch(X(LE)',Y(LE)',DATA(LE)');%%%%Íø¸ñ
%line([0,L],[H,H],'Parent',axes1,'Color',[1 1 1]);
end

