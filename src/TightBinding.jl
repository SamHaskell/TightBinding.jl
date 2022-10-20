module TightBinding

using LinearAlgebra

export dual, intralayerblock, intralayerhamiltonian, brillouinzone, occupancymatrix, meanfieldmatrix

function dual(u1, u2)
    
    """
    Generate the reciprocal lattice vectors dual to a pair of real-space lattice vectors.
    """

    U = [u1[1] u1[2];
         u2[1] u2[2]]
    V = 2*pi*inv(U)
    v1 = [V[1], V[2]]
    v2 = [V[3], V[4]]

    return v1, v2
end

function brillouinzone(g1, g2, N, half=true)
    
    """
    Generate a list of N x N k-vectors in the brillouin zone, with an option to do a half-BZ.
    N must be EVEN.
    Vectors are chosen to avoid BZ edges and corners to simplify integrating quantities over the BZ.
    """

    M = floor(Int64, N/2)

    dx, dy = g1/N, g2/N
    upper = M - 1
    
    if half == true
        lowerx = 0
        lowery = - M
    else
        lowerx = - M
        lowery = - M
    end

    return [(ix+0.5)*dx + (iy+0.5)*dy for ix in lowerx:upper, iy in lowery:upper]
end

function intralayerblock(K, J, Γ, uK, uJ, uΓ, uC, alpha)
    
    """
    Generate the 4x4 block U for a given direction alpha = 0, 1, 2 corresponding to x, y, z.
    """

    beta = (alpha+1)%3
    gamma = (alpha+2)%3

    i = alpha + 2 # +2 because julia indexes from 1, and we want lower right 3x3 of a larger 4x4.
    j = beta + 2
    k = gamma + 2
    
    U = [0 0 0 0;
         0 0 0 0;
         0 0 0 0;
         0 0 0 0]

    U[1, 1] = (K+J)*uK + 2Γ*uΓ + 2J*uJ
    U[i, i] = (K+J)*uC
    U[j, j] = J*uC
    U[k, k] = J*uC
    U[j, k] = Γ*uΓ
    U[k, j] = Γ*uΓ

    return U
end

function intralayerhamiltonian(block_list, momentum, neighbour_list)
    
    """
    Constructs the 8x8 k-dependant Hamiltonian block for a single KHG layer.
    """
    
    H = Array{ComplexF64}(undef, 4, 4)
    Z = zeros(ComplexF64, 4, 4)

    for i in 1:3
        H += 1im*block_list[i]*exp(1im*dot(momentum, neighbour_list[i]))
    end

    return -0.25*vcat(hcat(Z, H), hcat(H', Z))
end

function occupancymatrix(values, temperature=0, mu=0)

    """
    Calculates occupancy "matrix" for the bands at a given k.
    Can specify a temperature if you'd like, and can shift the fermi energy by changing mu.
    """

    return diagm([1/(exp((E-mu)/temperature)+1) for E in values])
end

function meanfieldmatrix(vectors, occupancy)

    """
    Given the occupancy matrix O of bands and matrix of eigenvectors P, 
    calculate the product `POP^{†}`.
    """

    return vectors*(occupancy*vectors')

end

end