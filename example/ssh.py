"""
A Tutorial for the Optical SSH Model Ground State Proper
"""
import numpy
import h5py
from renormalizer.model.model import Model, construct_j_matrix
from renormalizer.model.basis import BasisSimpleElectron, BasisSHO
from renormalizer.model.op import Op
from renormalizer.utils import Quantity
from renormalizer.mps import Mps, Mpo
from renormalizer.mps.gs import optimize_mps


class OpticalSSHModelGroundState():
    r"""
    Ground state properties for the optical SSH model.
    Hamiltonian:
    He = sum_{i} t a^\dagger_i a_{i+1}+a_{i+1}^\dagger a_i
    Hph = sum_i w_0 b^\dagger_i b_i
    Heph = sum_{i} g (a^\dagger_i+1 a_i + a^\dagger_i a_{i+1}) (X_{i+1} - X_i)
    X_i = b^\dagger_i + b_i
    """
    def __init__(self, params):
        # Model parameters
        self.mol_num = params['nsites']
        self.g = params['g']
        self.w0 = params['w0']
        self.nboson_max = params['nboson_max']
        self.bond_dim = params['bond_dim']
        self.nsweeps = params['nsweeps']
        self.periodic = params['periodic']
        self.t = params['t']

        # Construct hopping matrix
        j_matrix = construct_j_matrix(self.mol_num, Quantity(self.t), self.periodic)

        # Initialize model with basis and Hamiltonian
        self.model = self._construct_model(j_matrix)
    
    def _construct_model(self, j_matrix):
        # Construct basis
        basis = []
        for imol in range(self.mol_num):
            basis.append(BasisSimpleElectron(imol))
            basis.append(BasisSHO((imol, 0), self.w0, self.nboson_max))

        # Construct Hamiltonian terms
        ham = []
        # Electronic hopping terms
        for imol in range(self.mol_num):
            for jmol in range(self.mol_num):
                if j_matrix[imol, jmol] != 0:
                    ham.append(Op(r"a^\dagger a", [imol, jmol], j_matrix[imol, jmol]))
        
        # Bosonic terms
        for imol in range(self.mol_num):
            ham.append(Op("b^\dagger b", (imol, 0), self.w0))
        
        # Electron-phonon coupling
        ham.extend(self._construct_eph_terms())
        
        return Model(basis, ham)
    
    def _construct_eph_terms(self):
        eph_terms = []
        # Bulk terms
        for imol in range(self.mol_num-1):
            eph_terms.extend([
                Op(r"a^\dagger a", [imol, imol+1], self.g) * Op("b^\dagger+b", (imol+1, 0)),
                Op(r"a^\dagger a", [imol, imol+1], -self.g) * Op("b^\dagger+b", (imol, 0)),
                Op(r"a^\dagger a", [imol+1, imol], self.g) * Op("b^\dagger+b", (imol+1, 0)),
                Op(r"a^\dagger a", [imol+1, imol], -self.g) * Op("b^\dagger+b", (imol, 0))
            ])
        
        # Boundary terms (if periodic)
        if self.periodic:
            eph_terms.extend([
                Op(r"a^\dagger a", [self.mol_num-1, 0], self.g) * Op("b^\dagger+b", (0, 0)),
                Op(r"a^\dagger a", [self.mol_num-1, 0], -self.g) * Op("b^\dagger+b", (self.mol_num-1, 0)),
                Op(r"a^\dagger a", [0, self.mol_num-1], self.g) * Op("b^\dagger+b", (0, 0)),
                Op(r"a^\dagger a", [0, self.mol_num-1], -self.g) * Op("b^\dagger+b", (self.mol_num-1, 0))
            ])
        return eph_terms

    def get_gs_energy(self):
        # Initialize random MPS and construct MPO
        mps = Mps.random(self.model, 1, self.bond_dim, percent=1.0)
        mpo = Mpo(self.model)
        
        # Setup optimization procedure
        procedure = [
            [self.bond_dim//4, 0.4],
            [self.bond_dim//2, 0.2], 
            [3*self.bond_dim//4, 0.1]
        ] + [[self.bond_dim, 0]] * (self.nsweeps - 3)
        mps.optimize_config.procedure = procedure
        mps.optimize_config.method = "2site"
        
        # Optimize ground state
        energies, mps = optimize_mps(mps.copy(), mpo)
        
        # Calculate observables
        results = {
            'energies': energies,
            'edof_rdm': mps.calc_edof_rdm(),
            'phonon_occupations': mps.ph_occupations,
            'phonon_displacement': self.calc_phonon_displacement(mps),
            'ni_nj': self.calc_ni_nj(mps)
        }
        
        return results
    
    def calc_ni_nj(self, mps):
        """Calculate density-density correlation function."""
        ni_nj = numpy.zeros((self.mol_num, self.mol_num))
        for imol in range(self.mol_num):
            for jmol in range(self.mol_num):
                ni = Mpo(self.model, Op("a^\dagger a", [imol, imol]))
                nj = Mpo(self.model, Op("a^\dagger a", [jmol, jmol]))
                mpo = ni @ nj
                ni_nj[imol, jmol] = mps.expectation(mpo)
        return ni_nj
    
    def calc_phonon_displacement(self, mps):
        """Calculate phonon displacement for each site."""
        phonon_displacement = numpy.zeros(self.mol_num)
        for imol in range(self.mol_num):
            mpo = Mpo(self.model, Op("b^\dagger+b", (imol, 0)))
            phonon_displacement[imol] = mps.expectation(mpo)
        return phonon_displacement


if __name__ == '__main__':
    import sys 
    # Model parameters
    params = {
        "nsites": 2,
        'g': 0.7,
        'w0': 0.5,
        't': -1.0,
        'nboson_max': 4,
        'bond_dim': 16,
        'nsweeps': 10,
        'periodic': True
    }
    
    # Ground state calculation
    job = OpticalSSHModelGroundState(params)
    results = job.get_gs_energy()

    # Save results
    with h5py.File('gs.h5', 'w') as f:
        for key, value in results.items():
            f.create_dataset(key, data=value)
        f.create_dataset('gs_energy', data=min(results['energies']))