from npfm.upstream import permitted_license,github_repository,keep_source

def test_licenses_are_explicit_and_repository_is_bounded():
    assert permitted_license('MIT; weights separately licensed')
    assert permitted_license('代码MIT；数据与权重独立核验')
    assert permitted_license('GPL-3.0')
    assert permitted_license('GPLv3；权重另行核验')
    assert not permitted_license('CC BY-NC-SA 4.0')
    assert not permitted_license('unknown')
    assert github_repository('https://github.com/rdkit/rdkit/tree/master')=='rdkit/rdkit'
    assert github_repository('https://example.com/rdkit/rdkit') is None

def test_snapshot_excludes_data_and_weights_preserves_code():
    assert keep_source('repo/src/algorithm.f90',5000)
    assert keep_source('repo/kernels/kernel.cu',5000)
    assert keep_source('repo/LICENSE',5000)
    assert not keep_source('repo/data/measurements.txt',5000)
    assert not keep_source('repo/model.pt',5000)
    assert not keep_source('repo/train.ipynb',5000)
