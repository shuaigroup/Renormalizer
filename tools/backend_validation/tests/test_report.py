import pytest
from tools.backend_validation.report import validate_report,write_report,HASHES


def record():
    return dict(schema_version=1,status='pass',domain='fixed control',mode='reviewed',resource_coverage='limited',unsupported=[],results={'correct':'pass'},**{k:'a'*64 for k in HASHES})

@pytest.mark.parametrize('key',HASHES)
def test_missing_identity_rejected(key):
    report=record();report.pop(key)
    with pytest.raises(ValueError):validate_report(report)


def test_false_completion_and_nonfinite_rejected(tmp_path):
    for update in [{'skipped':1},{'results':{}},{'device':'cuda:0'},{'claim':'performance','device_witness':True,'timings':{'t_e2e':1}},{'results':{'time':float('nan')}}]:
        with pytest.raises(ValueError):validate_report({**record(),**update})
    path=tmp_path/'reference.json';path.write_text('frozen')
    with pytest.raises(FileExistsError):write_report(path,record())
    assert path.read_text()=='frozen'


@pytest.mark.parametrize('cell', ['fail', 'blocked', {'status':'fail'},
    {'status':'pass','correctness':{'status':'fail'}}])
def test_contradictory_pass_rejected(cell):
    with pytest.raises(ValueError):
        validate_report({**record(),'results':{'correct':cell}})


def test_declared_negative_controls_have_exact_expected_keys():
    report={**record(),'results':{'correct':{'status':'pass'},'wrong':{'status':'fail'}},
            'expected_statuses':{'correct':'pass','wrong':'fail'}}
    assert validate_report(report) is report
    for expected in ({'correct':'pass'}, {'correct':'pass','wrong':'fail','extra':'pass'},
                     {'correct':'pass','wrong':'blocked'}):
        with pytest.raises(ValueError):validate_report({**report,'expected_statuses':expected})


def witness():
    return dict(device='cuda:0',dtype='float64',shape=[2,3],cupy_version='13.6.0')


@pytest.mark.parametrize('bad', [True, {}, {'device':'cuda:0'},
    {**witness(),'device':'cpu'}, {**witness(),'dtype':'object'},
    {**witness(),'shape':[True,3]}, {**witness(),'cupy_version':''}])
def test_gpu_witness_must_be_structured_at_every_result_level(bad):
    for update in ({'device':'cuda:0','device_witness':bad},
                   {'results':{'gpu':{'status':'pass','device':'cuda:0','device_witness':bad}}}):
        with pytest.raises(ValueError):validate_report({**record(),**update})


def test_nested_gpu_expected_controls_and_witness():
    gpu=dict(status='pass',device='cuda:0',device_witness=witness(),
             results={'correct':{'status':'pass'},'wrong':{'status':'fail'}},
             expected_statuses={'correct':'pass','wrong':'fail'})
    assert validate_report({**record(),'results':{'gpu':gpu}})['status']=='pass'
    with pytest.raises(ValueError):
        validate_report({**record(),'device':'cuda:1','device_witness':witness()})


def test_gpu_negative_control_list_contract():
    gpu=dict(status='pass',device_witness=witness(),correctness={'status':'pass'},
             negative_controls=[{'status':'fail','reason':'numerical_error'}])
    assert validate_report({**record(),'results':{'gpu':gpu}})['status']=='pass'
    for controls in ([],[{'status':'pass'}],[{'status':'blocked'}],['fail']):
        with pytest.raises(ValueError):
            validate_report({**record(),'results':{'gpu':{**gpu,'negative_controls':controls}}})
