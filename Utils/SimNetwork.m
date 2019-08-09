% -----------------------------------------------------------------
% Author's page:
% https://sites.google.com/view/alessandrovilla/
% -----------------------------------------------------------------
function output=SimNetwork(input,netparams)

in = (netparams.ymaxin-netparams.yminin)*(input-netparams.xminin)./(netparams.xmaxin-netparams.xminin) + netparams.yminin;
h = tansig(netparams.IW*in+netparams.B1);
y2n = netparams.LW*h + netparams.b2;
output=(netparams.xmaxout-netparams.xminout).*(y2n-netparams.yminout)./(netparams.ymaxout-netparams.yminout) + netparams.xminout;

end