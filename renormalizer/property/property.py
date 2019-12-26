from typing import Union, List, Dict
from renormalizer.mps import Mpo, Mps, MpDm

class Property():
    '''
    class to store the properties mpos and expectation results.
    
    Property can be added in any TdMpsJob class (see example in thermalprop) to define
    the user-defined mpos whose expectation values should be calculated and
    stored during TD simulation.
    Note that when the result is complex number, use dump_type=".npz" in TdMpsJob.
    '''

    def __init__(self, prop_strs:List[str], prop_mpos:Dict[str, Mpo]):
        '''
        prop_strs: names of the properties
        prop_mpos: dict: mpos of the properties
        '''
        self.prop_strs = prop_strs
        self.prop_mpos = prop_mpos
        
        self.prop_res = {}
        for prop_str in prop_strs:
            self.prop_res[prop_str] = []
        

    def calc_properties(self, mpss: Union[Dict[str,Mps], Dict[str,MpDm],  Mps, MpDm], 
            mps_conjs:Union[Dict[str, Mps], Dict[str, MpDm], Mps, MpDm, None] =None):
        '''
        calculate all the properties with same mps or mps_conj
        or 
        calculate each property with different {prop_str:mps}, {prop_str:mps_conj}
        '''
        for prop_str in self.prop_strs:
            
            # pick ket
            if mpss.is_mps or mpss.is_mpdm:
                mps = mpss
            else:
                assert prop_str in mpss
                mps = mpss[prop_str]

            # pick bra
            if (mps_conjs is None) or (mps_conjs.is_mps or mps_conjs.is_mpdm):
                mps_conj = mps_conjs
            else:
                assert prop_str in mps_conjs
                mps_conj = mps_conjs[prop_str]

            # pick the mpo
            # todo: some properties are calculated by calling expectations
            # explicitly, while some others are calculated by calling specific
            # functions. It is better to unify these two cases
            if prop_str == "e_rdm":
                self.prop_res[prop_str].append(mps.calc_reduced_density_matrix())
            elif prop_str in self.prop_mpos.keys():
                mpo = self.prop_mpos[prop_str]
                self.prop_res[prop_str].append(mps.expectation(mpo, mps_conj))
            else:
                # on-the-fly constructed 
                raise NotImplementedError
            
            
            


