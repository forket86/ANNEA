% -----------------------------------------------------------------
% Source (Please cite the paper if you use this algorithm in other applications):
% Machine Learning Projection Methods for Macro-finance Models
% https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3209934
%
% Author's page:
% https://sites.google.com/view/alessandrovilla/
% -----------------------------------------------------------------

function residuals = Euler(beta,alpha,delta,du,ctilde,phic,k_grd,z_grd,P_z)

for k_iter=1:length(k_grd)
    for z_iter=1:length(z_grd)
        
        k=k_grd(k_iter);
        z=z_grd(z_iter);
        c=ctilde(k,z,phic);
        
        knext=z.*k.^alpha+(1-delta)*k-c;
        
        cnext=@(znext) ctilde(knext*ones(length(znext),1),znext,phic);
        
        RHS=du(cnext(z_grd)).*(alpha*z_grd*knext^(alpha-1)+1-delta);
        ERHS=P_z(z_iter,:)*RHS;
        residuals(k_iter,z_iter)=du(c)-beta*ERHS;
    end
end

end