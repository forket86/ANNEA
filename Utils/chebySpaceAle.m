function grid=chebySpaceAle(low,high,size);

cna = chebyNodeAle(size);% Chebyshev nodes
grid=chebyTransformAle(low,high,cna,true);
grid = fliplr(grid');

end

