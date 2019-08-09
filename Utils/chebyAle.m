% -----------------------------------------------------------------
% Author's page:
% https://sites.google.com/view/alessandrovilla/
% -----------------------------------------------------------------
function value = chebyAle(order,x,w,lowX,highX,lowT,highT,cross)

[row col]=size(x);
dimension=col;
value=zeros(row,1);

if dimension==1
    x(:,1)=chebyTransformAle(lowX(1),highX(1),x(:,1),false);
    for i=0:order
        value(:)=value(:)+w(i+1,1)*chebyTAle(i,x(:,1));
    end
end

if dimension==2 && cross==false
    x(:,1)=chebyTransformAle(lowX(1),highX(1),x(:,1),false);
    x(:,2)=chebyTransformAle(lowX(2),highX(2),x(:,2),false);
    for i=0:order
        value(:)=value(:)+w(i+1,1)*chebyTAle(i,x(:,1))+w(i+1,2)*chebyTAle(i,x(:,2));
    end
end

if dimension==2 && cross==true
    x(:,1)=chebyTransformAle(lowX(1),highX(1),x(:,1),false);
    x(:,2)=chebyTransformAle(lowX(2),highX(2),x(:,2),false);
    for i=0:order
        value(:)=value(:)+w(i+1,1)*chebyTAle(i,x(:,1))+w(i+1,2)*chebyTAle(i,x(:,2))+w(i+1,3)*chebyTAle(i,x(:,1).*x(:,2));
    end
end

if dimension==3
    x(:,1)=chebyTransformAle(lowX(1),highX(1),x(:,1),false);
    x(:,2)=chebyTransformAle(lowX(2),highX(2),x(:,2),false);
    x(:,3)=chebyTransformAle(lowX(3),highX(3),x(:,3),false);
    for i=0:order
        value(:)=value(:)+w(i+1,1)*chebyTAle(i,x(:,1))+w(i+1,2)*chebyTAle(i,x(:,2))+w(i+1,3)*chebyTAle(i,x(:,3));
    end
end

value(:)=chebyTransformAle(lowT,highT,value(:),true);

end

