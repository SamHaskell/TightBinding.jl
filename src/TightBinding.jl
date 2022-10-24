module TightBinding

using LinearAlgebra
using Documenter
using StaticArrays
using Distributed
using SharedArrays

export  dual, intralayerblock, intralayerhamiltonian, brillouinzone, 
        occupancymatrix, meanfieldmatrix, phasematrix, meanfieldpartial,
        meanfielditerator, meanfieldupdate, runtoconvergence, randomrepeat

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
    
    U = zeros(ComplexF64, 4, 4)

    U[1, 1] = (K+J)*uK + 2Γ*uΓ + 2J*uJ
    U[i, i] = (K+J)*uC
    U[j, j] = J*uC
    U[k, k] = J*uC
    U[j, k] = Γ*uC
    U[k, j] = Γ*uC

    return U
end

function intralayerhamiltonian(block_list, momentum, nearest_neighbours)
    
    """
    Constructs the 8x8 k-dependant Hamiltonian block for a single KHG layer.
    """
    
    H = zeros(ComplexF64, 4, 4)
    Z = zeros(ComplexF64, 4, 4)

    for i in 1:3
        H += 1im*block_list[i]*exp(1im*dot(momentum, nearest_neighbours[i]))
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

function phasematrix(momentum, translation)

    """
    Gives a diagonal 8x8 matrix U such that UAU^{†} applies the correct phase to the
    off-diagonal blocks of the layer Hamiltonian.
    """

    phase = exp(0.5im*dot(momentum, translation))
    phase_list = [phase, phase, phase, phase, phase', phase', phase', phase']
    return diagm(phase_list)
end

function meanfieldmatrix(vectors, occupancy)

    """
    Given the occupancy matrix Λ of bands and matrix of eigenvectors P, 
    calculate the product `PΛP^{†}`.
    """

    return vectors*(occupancy*vectors')
end

function meanfieldpartial(fields, phases)

    """
    Compute a matrix whose upper half triangle contains each of the k-momentum'th
    partial summands of our mean-fields.
    """

    mat = 1im*phases'*(fields*phases)

    return mat - transpose(mat)
end

function meanfielditerator(K, J, Γ, init_fields, brillouin_zone, nearest_neighbours, alpha, temperature = 0)
    
    inds = [(6, 2), (7, 3), (7, 2), (5, 1)]
    uK, uJ, uΓ, uC = [init_fields[x, y] for (x, y) in inds]

    block_list = [intralayerblock(K, J, Γ, uK, uJ, uΓ, uC, i-1) for i in 1:3]
    N = 2*Int(length(brillouin_zone))

    new_fields = zeros(ComplexF64, 8, 8)

    for k in brillouin_zone
        H = intralayerhamiltonian(block_list, k, nearest_neighbours)
        vals, vecs = eigen(H)
        occ = occupancymatrix(vals, temperature)
        fields = meanfieldmatrix(vecs, occ)
        phases = phasematrix(k, nearest_neighbours[alpha])
        new_fields += meanfieldpartial(fields, phases)
    end

    return new_fields/N
end

function meanfieldupdate(init_fields, new_fields, mixing = 1)

    """
    Uses successive over-relaxation/mixed updating to 
    generate new sets of mean-fields for each iteration.
    """

    return new_fields = (1-mixing)*init_fields .+ mixing*new_fields
end

function runtoconvergence(K, J, Γ, init_fields, brillouin_zone, nearest_neighbours, mixing = 1, tol = 1e-9, temperature = 0)

    """
    Given an initial set of mean-fields, run the iterator until all fields converge
    to the specified precision.
    """

    n_its = 0
    not_converged = true
    new_fields = init_fields
    while not_converged
        new_fields = meanfielditerator(K, J, Γ, init_fields, brillouin_zone, nearest_neighbours, 1, temperature)
        diff = abs.(real.(new_fields - init_fields))
        con = diff .> tol
        not_converged = any(con)
        n_its += 1
        init_fields = real.(meanfieldupdate(init_fields, new_fields, mixing))
    end
    println(n_its)
    return init_fields
end

function randomrepeat(K, J, Γ, brillouin_zone, nearest_neighbours, repeats = 10, mixing = 1, tol = 1e-9, temperature = 0)

    shape = (8, 8, repeats)
    init_field_arr = rand(Float64, shape).-0.5


end

end