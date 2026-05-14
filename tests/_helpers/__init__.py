"""Test helpers shared across v2.6+ test files.

Underscore-prefixed (no ``test_`` prefix) so pytest does NOT
collect this directory. Sibling test files import from here, e.g.:

    from tests._helpers.incremental import (
        assert_batch_incremental_equivalent,
    )
"""
