"""Installation smoke test for the phase-zero package scaffold."""


def test_package_import():
    import empygrad

    assert empygrad.__version__
