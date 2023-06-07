def set_color(label: int) -> tuple:
    """
    Adds color depending on the class
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color_table = {
        0: (85, 45, 255), # person
        1: (7, 127, 15),  # bicycle
        2: (255, 149, 0), # Car
        3: (0, 204, 255), # Motobike
        5: (0, 149, 255), # Bus
        7: (222, 82, 175) # truck
    }

    if label in color_table.keys():
        color = color_table[label]
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    
    return color