% -----------------------------------------------------------------
% Author's page:
% https://sites.google.com/view/alessandrovilla/
% -----------------------------------------------------------------
function t = chebyTransformAle(low,high,x,fromUnitToGeneric)

    if fromUnitToGeneric
        t = (high-low)/2*(x+1)+low;
    else
        t = 2*(x-low)/(high-low)-1;
    end

end

