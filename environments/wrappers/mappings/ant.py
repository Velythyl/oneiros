from environments.wrappers.mappings.vector_index_rearrange import _Mapping


class Ant(_Mapping):
    act: dict = {
        0: 6,
        1: 7,
        2: 0,
        3: 1,
        4: 2,
        5: 3,
        6: 4,
        7: 5
    }

    obs: dict = {
        0: 0,
        1: 2,
        2: 3,
        3: 4,
        4: 1,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
        10: 10,
        11: 11,
        12: 12,
        13: 13,
        14: 14,
        15: 15,
        16: 16,
        17: 17,
        18: 18,
        19: 19,
        20: 20,
        21: 21,
        22: 22,
        23: 23,
        24: 24,
        25: 25,
        26: 26
    }
