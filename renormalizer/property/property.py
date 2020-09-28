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


    def calc_properties_braketpair(self, mps):
        bra, ket = mps.bra_mps, mps.ket_mps
        for prop_str in self.prop_strs:
            mpo = self.prop_mpos[prop_str]
            if prop_str in ["x", "x^2", "n"]:
                # <bra|op|bra>, <ket|op|ket>
                res = []
                if isinstance(mpo, Mpo):
                    res.append(bra.expectation(mpo, None))
                    res.append(ket.expectation(mpo, None))
                elif isinstance(mpo, list):
                    # mpos
                    res.append(bra.expectations(mpo))
                    res.append(ket.expectations(mpo))

                self.prop_res[prop_str].append(res)
            else:
                # <bra |op|ket>
                self.prop_res[prop_str].append(ket.expectation(mpo, bra))


    def calc_properties(self, mps: Union[Mps, MpDm], 
            mps_conj:Union[Mps, MpDm, None] =None):
        '''
        calculate all the properties with same mps or mps_conj
        or 
        calculate each property with different {prop_str:mps}, {prop_str:mps_conj}
        '''
        for prop_str in self.prop_strs:
            
            # todo: 
            # 1. some properties are calculated by calling expectations
            # explicitly, while some others are calculated by calling specific
            # functions. It is better to unify these two cases
            # 2. different mps/mps_conj for different mpo
            if prop_str == "e_rdm":
                self.prop_res[prop_str].append(mps.calc_edof_rdm())
            elif prop_str in self.prop_mpos.keys():
                mpo = self.prop_mpos[prop_str]
                if isinstance(mpo, Mpo):
                    self.prop_res[prop_str].append(mps.expectation(mpo, mps_conj))
                elif isinstance(mpo, list):
                    # mpos
                    assert mps_conj is None
                    self.prop_res[prop_str].append(mps.expectations(mpo))
                else:
                    assert False
            else:
                # on-the-fly constructed 
                raise NotImplementedError
            
            
            


