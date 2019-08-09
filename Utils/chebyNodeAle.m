function x=chebyNodeAle(n);

r	= max(1,n);
n	= (1:n)';
x	= cos( (pi*(n-0.5))/r );
