import pytest

from tools.backend_validation.legacy_inventory import _git, inventory


@pytest.mark.skipif(_git('rev-parse', '--verify', '06430a9^{commit}').returncode != 0,
                    reason='reference commit 06430a9 not in the git history (shallow clone or source tree)')
def test_inventory_preserves_facade_and_conversion_consumers():
    result = inventory('06430a9')
    assert 'renormalizer/mps/oe_contract_wrap.py' in result['facade']
    assert 'renormalizer/mps/matrix.py' in result['matrix_helpers']
    assert len(result['source_commit']) == 40
    assert result['source'] == '06430a9'
