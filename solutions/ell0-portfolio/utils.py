import scipy.sparse as sparse

def sprandsym(n, density):
    M = sparse.random_array((n, n), density=density)
    L = sparse.tril(M)
    result = L + L.T - sparse.diags(M.diagonal())
    return result
