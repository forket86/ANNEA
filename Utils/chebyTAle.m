% -----------------------------------------------------------------
% Author's page:
% https://sites.google.com/view/alessandrovilla/
% -----------------------------------------------------------------
function T = chebyTAle(order,x)

switch order
   case 0
      T=ones(length(x),1);
   case 1
      T=x;
   case 2
      T=2*x.^2-1;
   case 3
      T=4*x.^3-3*x;
   case 4
      T=8*x.^4-8*x.^2+1;
   case 5
      T=16*x.^5-20*x.^3+5*x;
   case 6
      T=32*x.^6-48*x.^4+18*x.^2-1;
   case 7
      T=64*x.^7-112*x.^5+56*x.^3-7*x;
   case 8
      T=128*x.^8-256*x.^6+160*x.^4-32*x.^2+1;
   case 9
      T=256*x.^9-576*x.^7+432*x.^5-120*x.^3+9*x;
   case 10
      T=512*x.^10-1280*x.^8+1120*x.^6-400*x.^4+50*x.^2-1;
   case 11
      T=1024*x.^11-2816*x.^9+2816*x.^7-1232*x.^5+220*x.^3-11*x;
end

end

