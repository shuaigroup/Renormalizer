import hashlib
import os
import numpy as np
import pytest
from tools.backend_validation.protocol import (
    ProtocolError, array_descriptor, load_array, reconstruct, candidate_manifest,
    validate_manifest, freeze_manifest,
)


def test_numeric_roundtrip_strided_layout(tmp_path):
    backing = np.arange(12,dtype='float64')
    np.save(tmp_path/'a.npy', backing)
    desc = array_descriptor(tmp_path/'a.npy', shape=(2,3), strides=(48,16), offset=0)
    loaded = load_array(tmp_path, desc, max_elements=100, max_file_bytes=4096)
    view = reconstruct(loaded, desc)
    np.testing.assert_array_equal(view, [[0,2,4],[6,8,10]])
    assert view.strides == (48,16) and not view.flags.c_contiguous

@pytest.mark.parametrize('name', ['../a.npy','/tmp/a.npy','nested/a.npy'])
def test_paths_rejected(tmp_path, name):
    np.save(tmp_path/'a.npy',np.ones(2))
    desc=array_descriptor(tmp_path/'a.npy'); desc['file']=name
    with pytest.raises(ProtocolError):
        load_array(tmp_path,desc,max_elements=100,max_file_bytes=4096)

def test_symlink_header_and_budget_rejection(tmp_path):
    np.save(tmp_path/'a.npy',np.ones(2))
    desc=array_descriptor(tmp_path/'a.npy')
    os.symlink('a.npy',tmp_path/'link.npy')
    with pytest.raises(ProtocolError):
        load_array(tmp_path,{**desc,'file':'link.npy'},max_elements=100,max_file_bytes=4096)
    with pytest.raises(ProtocolError):
        load_array(tmp_path,desc,max_elements=1,max_file_bytes=4096)
    np.save(tmp_path/'bad.npy',np.array([object()]))
    data=(tmp_path/'bad.npy').read_bytes()
    bad={**desc,'file':'bad.npy','sha256':hashlib.sha256(data).hexdigest()}
    with pytest.raises(ProtocolError):
        load_array(tmp_path,bad,max_elements=100,max_file_bytes=4096)

def test_bounds_reject_before_view(tmp_path):
    np.save(tmp_path/'a.npy',np.arange(8.))
    desc=array_descriptor(tmp_path/'a.npy')
    for change in [{'offset':1000},{'strides':[-8]},{'shape':[100]}]:
        with pytest.raises(ProtocolError):
            reconstruct(np.arange(8.),{**desc,**change})

def test_manifest_projection_rejects_nested_extra_keys(tmp_path):
    np.save(tmp_path/'a.npy',np.eye(2)); np.save(tmp_path/'b.npy',np.eye(2))
    np.save(tmp_path/'reference.npy',np.eye(2))
    full=freeze_manifest(tmp_path,tmp_path,candidate_hash='a'*64,
                         source_hash='b'*64,environment_hash='c'*64)
    full["resource_policy"]["max_file_bytes"]=4096
    validate_manifest(full,max_elements=100,max_file_bytes=4096)
    visible=candidate_manifest(full)
    assert 'reference' not in visible and 'tolerances' not in visible
    assert 'directory' not in repr(visible)
    full['inputs'][0]['reference_path']='/secret'
    with pytest.raises(ProtocolError):
        candidate_manifest(full)

@pytest.mark.parametrize('change', [
    {'shape':[-1,2]}, {'shape':[1000001,0]}, {'dtype':'object'},
    {'dtype':['float64']}, {'offset':True}, {'strides':[3,8]},
])
def test_bad_descriptors_fail_before_numpy_load(tmp_path, monkeypatch, change):
    np.save(tmp_path/'a.npy',np.eye(2))
    desc=array_descriptor(tmp_path/'a.npy');desc.update(change)
    def forbidden(*args, **kwargs):
        pytest.fail('np.load called before descriptor validation')
    monkeypatch.setattr(np,'load',forbidden)
    with pytest.raises(ProtocolError):
        load_array(tmp_path,desc)


def test_output_shape_header_rejected_without_materialization(tmp_path,monkeypatch):
    np.save(tmp_path/'result.npy',np.ones((10,10)))
    monkeypatch.setattr(np,'load',lambda *args,**kwargs:pytest.fail('materialized bad output'))
    with pytest.raises(ProtocolError,match='shape_mismatch'):
        array_descriptor(tmp_path/'result.npy',expected_shape=(1,1))


def test_truncated_payload_and_nonfinite_json_are_rejected(tmp_path):
    np.save(tmp_path/'a.npy',np.eye(2))
    path=tmp_path/'a.npy';path.write_bytes(path.read_bytes()[:-3])
    with pytest.raises(ProtocolError): array_descriptor(path)
    from tools.backend_validation.runner import _json
    for data in ['{"x":NaN}','{"x":1,"x":2}']:
        with pytest.raises(ProtocolError): _json(data)
