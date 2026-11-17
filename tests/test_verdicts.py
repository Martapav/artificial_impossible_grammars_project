"""Unit tests for the H′/L′ clause-(i) wh licensing verdicts.

The swap axis of the §F design: H′ licenses the third CAT1-position of the
base string (positional); L′ licenses the matrix object that HOSTS an embedded
clause, read from generation history (structural, proposal E3 2026-07-30).
Divergence = the two verdicts disagree on the same target. Category strings are
hand-picked worked examples of the divergent and coherent configurations; the
recovery functions are exercised against natural generation by
tests/test_structural_probes.py and tests/test_metadata_sidecar.py.
"""

import pytest

from grammars.verdicts import hprime_licenses, lprime_licenses, verdicts


# Each case: (name, categories, target_index, role, clause_id,
#             expected_hprime_license, expected_lprime_license, expected_divergence)
CASES = [
    # A bare transitive object hosts nothing, so under E3 (2026-07-30) L′ no
    # longer licenses it either: coherent not/not, where it used to be the mass
    # divergence case.
    ("bare_transitive_object_hosts_nothing",
     ["CAT1", "CAT4", "CAT3", "CAT1"], 3, "object", 0, False, False, False, False),

    # The same object, now HOSTING an embedded clause: L′ licenses (clause i''),
    # H′ does not (not third). This is E3's replacement mass divergence case.
    ("matrix_object_hosting_embedded",
     ["CAT1", "CAT4", "CAT3", "CAT1", "CAT9", "CAT1", "CAT3"], 3, "object", 0,
     False, True, True, True),

    # Matrix object at ordinal 3 (subject carries a PP complement):
    # both license — the agreement anchor.
    ("object_at_third_position_hosting",
     ["CAT1", "CAT6", "CAT1", "CAT3", "CAT1", "CAT9", "CAT1", "CAT3"], 4,
     "object", 0, True, True, False, True),

    # Embedded subject at ordinal 3 (subject-attached relative after a
    # subject PP comp is not needed — embedding supplies the ordinal):
    # H′ licenses (third position), L′ does not (not the matrix object).
    ("embedded_subject_at_third",
     ["CAT1", "CAT6", "CAT1", "CAT9", "CAT1", "CAT3", "CAT3"],
     4, "subject", 1, True, False, True, False),

    # PP complement at ordinal 3: H′ licenses, L′ does not.
    ("pp_comp_at_third",
     ["CAT1", "CAT3", "CAT1", "CAT6", "CAT1"], 4, "complement", 0,
     True, False, True, False),

    # Embedded object beyond ordinal 3: neither licenses (coherent not/not).
    ("embedded_object_past_third",
     ["CAT1", "CAT9", "CAT1", "CAT4", "CAT3", "CAT1", "CAT3", "CAT1"],
     7, "object", 1, False, False, False, False),
]


@pytest.mark.parametrize(
    "name,cats,idx,role,cid,h_lic,l_lic,div,hosts",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_verdict_cases(name, cats, idx, role, cid, h_lic, l_lic, div, hosts):
    assert hprime_licenses(cats, idx) is h_lic
    assert lprime_licenses(role, cid, hosts) is l_lic
    v = verdicts(cats, idx, role, cid, hosts)
    assert v["hprime_verdict"] == ("license" if h_lic else "not")
    assert v["lprime_verdict"] == ("license" if l_lic else "not")
    assert v["divergence"] is div


def test_hprime_cat1pron_counts_as_position():
    # A CAT1PRON shifts the ordinal count exactly like a CAT1 (D7).
    cats = ["CAT1", "CAT1PRON", "CAT3", "CAT1"]
    assert hprime_licenses(cats, 3) is True          # third position
    assert hprime_licenses(cats, 1) is False         # second position
    # Inflection-bearing words other than nominals never count.
    cats = ["CAT1", "CAT4", "CAT2", "CAT3", "CAT3AUX", "CAT1"]
    assert hprime_licenses(cats, 5) is False         # only 2nd CAT1-position


def test_hprime_fewer_than_three_positions_licenses_nothing():
    assert hprime_licenses(["CAT1", "CAT3", "CAT1"], 2) is False
    assert hprime_licenses(["CAT1", "CAT3"], 0) is False


def test_lprime_matrix_object_hosting_an_embedded_clause_only():
    """Clause (i'') is E3 (2026-07-30): matrix object AND it hosts a clause."""
    assert lprime_licenses("object", 0, True) is True
    assert lprime_licenses("object", 0, False) is False   # hosts nothing
    assert lprime_licenses("object", 1, True) is False    # embedded object
    assert lprime_licenses("subject", 0, True) is False   # subject host
    assert lprime_licenses("complement", 0, True) is False


def test_target_index_out_of_range():
    with pytest.raises(IndexError):
        hprime_licenses(["CAT1", "CAT6"], 5)


def test_target_must_be_nominal():
    with pytest.raises(ValueError):
        hprime_licenses(["CAT1", "CAT3", "CAT1"], 1)
