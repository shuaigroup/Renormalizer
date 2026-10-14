"""Bounded JSON/NPY protocol. Trusted-only metadata never enters worker input."""
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid
import numpy as np
from .scoring import validate_tolerances

MAX_ELEMENTS = 1_000_000
MAX_FILE_BYTES = 32 * 1024 * 1024
DTYPES = frozenset(('float32', 'float64', 'complex64', 'complex128'))
DESCRIPTOR_KEYS = frozenset(('file','sha256','dtype','shape','storage_shape','strides','offset'))
CANDIDATE_FIELDS = ('schema_version','run_id','op','inputs','output','device','layout','mutation_policy')
FULL_FIELDS = frozenset(CANDIDATE_FIELDS + ('mode','input_directory','reference_directory',
    'reference','candidate_hash','source_hash','environment_hash','scorer_hash',
    'tolerance_hash','tolerances','resource_policy'))


class ProtocolError(ValueError):
    pass


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _exact(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ProtocolError('unexpected or missing keys')


def _filename(name):
    if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-]+\.npy',name):
        raise ProtocolError('invalid numeric filename')


def _hash(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):
        raise ProtocolError('invalid SHA256')


def _shape(shape, max_elements):
    if not isinstance(shape,list) or len(shape)>8 or any(type(x) is not int or x<0 or x>max_elements for x in shape):
        raise ProtocolError('invalid dimensions')
    if math.prod(shape)>max_elements:
        raise ProtocolError('element budget exceeded')


def validate_descriptor(desc, *, max_elements=MAX_ELEMENTS):
    _exact(desc, DESCRIPTOR_KEYS)
    _filename(desc['file']); _hash(desc['sha256'])
    if not isinstance(desc['dtype'],str) or desc['dtype'] not in DTYPES:
        raise ProtocolError('unsupported dtype')
    _shape(desc['shape'],max_elements); _shape(desc['storage_shape'],max_elements)
    strides, offset = desc['strides'], desc['offset']
    itemsize=np.dtype(desc['dtype']).itemsize
    if (not isinstance(strides,list) or len(strides)!=len(desc['shape']) or
        any(type(s) is not int or s<0 or s%itemsize for s in strides) or
        type(offset) is not int or offset<0 or offset%itemsize):
        raise ProtocolError('unsupported strides/offset')
    extent = offset + (sum((n-1)*s for n,s in zip(desc['shape'],strides))+itemsize
                       if math.prod(desc['shape']) else 0)
    if extent > math.prod(desc['storage_shape'])*itemsize:
        raise ProtocolError('view exceeds backing storage')


def read_file(directory, name, *, max_bytes):
    """Open a fixed basename through a directory FD; reject links/nonregular files."""
    if not isinstance(name,str) or Path(name).name != name or name in ('.','..'):
        raise ProtocolError('invalid filename')
    try:
        parent=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
            try:
                info=os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink!=1 or info.st_size>max_bytes:
                    raise ProtocolError('not a bounded regular file')
                with os.fdopen(fd,'rb',closefd=False) as stream:
                    data=stream.read(max_bytes+1)
                if len(data)>max_bytes:
                    raise ProtocolError('file budget exceeded')
                return data
            finally:
                os.close(fd)
        finally:
            os.close(parent)
    except OSError as error:
        raise ProtocolError('cannot open regular file without following links') from error


def _parse_npy(data, *, max_elements, expected_shape=None, expected_dtype=None, header_only=False):
    stream=io.BytesIO(data)
    try:
        version=np.lib.format.read_magic(stream)
        if version==(1,0):
            shape, fortran, dtype=np.lib.format.read_array_header_1_0(stream,max_header_size=4096)
        elif version==(2,0):
            shape, fortran, dtype=np.lib.format.read_array_header_2_0(stream,max_header_size=4096)
        else:
            raise ProtocolError('unsupported npy version')
        _shape(list(shape),max_elements)
        if str(dtype) not in DTYPES or fortran:
            raise ProtocolError('only numeric C-contiguous backing files are supported')
        if stream.tell()+math.prod(shape)*dtype.itemsize != len(data):
            raise ProtocolError('truncated or trailing numeric payload')
        if expected_shape is not None and tuple(expected_shape)!=shape:
            raise ProtocolError('shape_mismatch')
        if expected_dtype is not None and np.dtype(expected_dtype)!=dtype:
            raise ProtocolError('dtype_mismatch')
        if header_only:
            return shape,dtype
        return np.load(io.BytesIO(data),allow_pickle=False,max_header_size=4096)
    except ProtocolError:
        raise
    except (ValueError, EOFError, TypeError, OverflowError) as error:
        raise ProtocolError('invalid numeric file') from error


def load_array(directory, desc, *, max_elements=MAX_ELEMENTS,max_file_bytes=MAX_FILE_BYTES):
    validate_descriptor(desc,max_elements=max_elements)
    data=read_file(directory,desc['file'],max_bytes=max_file_bytes)
    if hashlib.sha256(data).hexdigest()!=desc['sha256']:
        raise ProtocolError('numeric file hash mismatch')
    array=_parse_npy(data,max_elements=max_elements,expected_shape=desc['storage_shape'],expected_dtype=desc['dtype'])
    if array.shape!=tuple(desc['storage_shape']) or str(array.dtype)!=desc['dtype']:
        raise ProtocolError('numeric header mismatch')
    return array


def reconstruct(backing, desc):
    validate_descriptor(desc)
    if (not isinstance(backing,np.ndarray) or not backing.flags.c_contiguous or
        str(backing.dtype)!=desc['dtype'] or backing.shape!=tuple(desc['storage_shape'])):
        raise ProtocolError('backing array mismatch')
    view=np.ndarray(tuple(desc['shape']),dtype=backing.dtype,buffer=backing,
                    offset=desc['offset'],strides=tuple(desc['strides']))
    view.flags.writeable=False
    return view


def array_descriptor(path, *, shape=None, strides=None, offset=0, expected_shape=None, expected_dtype=None):
    path=Path(path)
    _filename(path.name)
    data=read_file(path.parent,path.name,max_bytes=MAX_FILE_BYTES)
    storage_shape,dtype=_parse_npy(data,max_elements=MAX_ELEMENTS,expected_shape=expected_shape,
                                  expected_dtype=expected_dtype,header_only=True)
    canonical_strides=[dtype.itemsize*math.prod(storage_shape[i+1:]) for i in range(len(storage_shape))]
    if math.prod(storage_shape)==0:
        canonical_strides=[0]*len(storage_shape)
    desc=dict(file=path.name,sha256=hashlib.sha256(data).hexdigest(),dtype=str(dtype),
              shape=list(storage_shape if shape is None else shape),storage_shape=list(storage_shape),
              strides=list(canonical_strides if strides is None else strides),offset=offset)
    validate_descriptor(desc)
    return desc


def validate_manifest(manifest, *, max_elements=MAX_ELEMENTS,max_file_bytes=MAX_FILE_BYTES):
    _exact(manifest,FULL_FIELDS)
    if type(manifest['schema_version']) is not int or manifest['schema_version']!=1 or manifest['op']!='matmul':
        raise ProtocolError('unknown schema or operation')
    if not isinstance(manifest['run_id'],str) or not re.fullmatch('[0-9a-f]{32}',manifest['run_id']):
        raise ProtocolError('invalid run identifier')
    if manifest['mode'] not in ('reviewed','untrusted') or manifest['mutation_policy']!='forbid' or manifest['layout']!='explicit_strides':
        raise ProtocolError('unsupported execution contract')
    if manifest['device']!='cpu' and not re.fullmatch(r'cuda:[0-9]+',str(manifest['device'])):
        raise ProtocolError('unsupported device')
    for key in ('input_directory','reference_directory'):
        if not isinstance(manifest[key],str) or not Path(manifest[key]).is_absolute():
            raise ProtocolError('trusted directories must be absolute')
    for key in ('candidate_hash','source_hash','environment_hash','scorer_hash','tolerance_hash'):
        _hash(manifest[key])
    try:
        validate_tolerances(manifest['tolerances'])
    except ValueError as error:
        raise ProtocolError(str(error)) from error
    if canonical_hash(manifest['tolerances'])!=manifest['tolerance_hash']:
        raise ProtocolError('tolerance hash mismatch')
    inputs=manifest['inputs']
    if not isinstance(inputs,list) or len(inputs)!=2:
        raise ProtocolError('matmul needs two inputs')
    for desc in inputs+[manifest['reference']]:
        validate_descriptor(desc,max_elements=max_elements)
    if len({d['file'] for d in inputs})!=2:
        raise ProtocolError('input filenames must be distinct')
    a,b=inputs
    if len(a['shape'])!=2 or len(b['shape'])!=2 or a['shape'][1]!=b['shape'][0] or a['dtype']!=b['dtype']:
        raise ProtocolError('matmul shape/dtype mismatch')
    output=manifest['output']
    _exact(output,('file','shape','dtype'))
    if output!={'file':'result.npy','shape':[a['shape'][0],b['shape'][1]],'dtype':a['dtype']}:
        raise ProtocolError('output contract mismatch')
    _shape(output['shape'],max_elements)
    reference=manifest['reference']
    if reference['shape']!=output['shape'] or reference['dtype']!=output['dtype']:
        raise ProtocolError('reference contract mismatch')
    policy=manifest['resource_policy']
    _exact(policy,('timeout_seconds','max_memory_bytes','max_file_bytes'))
    if type(policy['timeout_seconds']) not in (int,float) or not math.isfinite(policy['timeout_seconds']) or not 0<policy['timeout_seconds']<=300:
        raise ProtocolError('invalid timeout')
    if type(policy['max_memory_bytes']) is not int or not 128*1024**2<=policy['max_memory_bytes']<=8*1024**3:
        raise ProtocolError('invalid address-space budget')
    if type(policy['max_file_bytes']) is not int or not 1024<=policy['max_file_bytes']<=max_file_bytes:
        raise ProtocolError('invalid file budget')
    return manifest


def candidate_manifest(full):
    validate_manifest(full)
    # Round trip also breaks references to scorer-owned mutable dictionaries.
    return json.loads(json.dumps({key:full[key] for key in CANDIDATE_FIELDS},allow_nan=False))


def freeze_manifest(input_directory, reference_directory, *, candidate_hash, source_hash,
                    environment_hash, mode='reviewed', device='cpu', tolerances=None):
    """Construct a trusted manifest from already frozen a/b/reference.npy files."""
    from . import scoring
    inputs=[array_descriptor(Path(input_directory)/f'{name}.npy') for name in ('a','b')]
    reference=array_descriptor(Path(reference_directory)/'reference.npy')
    tolerances=dict(atol_F=1e-12,rtol_F=1e-10,atol_max=1e-12,rtol_max=1e-10) if tolerances is None else dict(tolerances)
    result=dict(schema_version=1,run_id=uuid.uuid4().hex,op='matmul',inputs=inputs,
                output=dict(file='result.npy',shape=reference['shape'],dtype=reference['dtype']),
                device=device,layout='explicit_strides',mutation_policy='forbid',mode=mode,
                input_directory=str(Path(input_directory).absolute()),
                reference_directory=str(Path(reference_directory).absolute()),reference=reference,
                candidate_hash=candidate_hash,source_hash=source_hash,environment_hash=environment_hash,
                scorer_hash=hashlib.sha256(Path(scoring.__file__).read_bytes()).hexdigest(),
                tolerance_hash=canonical_hash(tolerances),tolerances=tolerances,
                resource_policy=dict(timeout_seconds=10.,max_memory_bytes=1024**3,max_file_bytes=MAX_FILE_BYTES))
    return validate_manifest(result)
