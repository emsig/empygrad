"""Installation smoke test for the package."""


def test_package_import():
    import empygrad

    assert empygrad.__version__
