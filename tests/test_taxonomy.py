from classifier.taxonomy import load_taxonomy


def test_taxonomy_loads():
    taxonomy = load_taxonomy("taxonomy/assets/matador-c1-taxonomy.json")
    assert taxonomy.num_nodes > 0
    assert taxonomy.leaves
    assert taxonomy.edge_index().shape[0] == 2

