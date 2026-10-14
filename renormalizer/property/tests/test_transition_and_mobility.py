"""Independent transition-amplitude and host mobility-integration controls."""
import numpy as np
import pytest

from renormalizer import BasisHalfSpin, Model, Mps, Mpo
from renormalizer.mps.mps import BraKetPair
from renormalizer.property import Property
from renormalizer.transport.kubo import TransportKubo
from renormalizer.utils import Quantity
from renormalizer.utils.constant import mobility2au


@pytest.mark.parametrize('phase', [1., 1j, (1+1j)/np.sqrt(2)])
def test_property_transition_conjugates_physical_bra(phase, captured_backend):
    # Shared fixture honors the requested backend/device and restores captured
    # dispatch and host RNG; this test must not overwrite global selection.
    model = Model([BasisHalfSpin(0)], [])
    ket = Mps.hartree_product_state(model, {0: [1, 0]})
    # Use the supported complex conversion, independently of the constructor's
    # real allocation limitation, so failure isolates the property collector.
    bra = ket.to_complex()
    bra[0] = phase * bra[0].array
    operator = Mpo.identity(model)
    expected = np.vdot(bra.todense().ravel(), ket.todense().ravel())
    pair = BraKetPair(bra, ket, operator)
    assert pair.ft == pytest.approx(expected, abs=1e-14)
    assert ket.expectation(operator, bra=bra) == pytest.approx(expected, abs=1e-14)
    properties = Property(['transition'], {'transition': operator})
    properties.calc_properties_braketpair(pair)
    assert properties.prop_res['transition'][0] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize('times, correlation, area', [
    ([0., 1., 2.], [2.+0j, 2.+0j, 2.+0j], 4.),
    ([0., .5, 2.], [1.+8j, 3.-2j, 5.+9j], 7.),
])
def test_mobility_matches_independent_trapezoid_area(times, correlation, area):
    # Host-only observable integration: this case makes no device-execution
    # claim and needs no backend selection or random-number initialization.
    # Only the three observable fields used by calc_mobility are needed;
    # constructing a thermal-evolution job would add unrelated solver work.
    job = TransportKubo.__new__(TransportKubo)
    job.evolve_times = times
    job._auto_corr = correlation
    job.temperature = Quantity(2.)
    # Areas are hand-evaluated trapezoids; the second case also checks unequal
    # spacing and rejection of the imaginary correlation contribution.
    expected_au = area / job.temperature.as_au()
    actual_au, actual_mobility = job.calc_mobility()
    assert actual_au == pytest.approx(expected_au, rel=1e-13, abs=1e-14)
    assert actual_mobility == pytest.approx(expected_au/mobility2au, rel=1e-13)
