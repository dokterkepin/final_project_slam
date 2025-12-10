import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

def draw_map(ax):
    # outer scope
    ax.plot([2, 39], [2, 2], "k", linewidth=2)
    ax.plot([2, 39], [18, 18], "k", linewidth=2)

    # left border
    ax.plot([2, 2], [2, 8.5], "k", linewidth=2)
    ax.plot([2, 2], [11.5, 18], "k", linewidth=2)
    ax.plot([0, 2], [8.5, 8.5], "k", linewidth=2)
    ax.plot([0, 2], [11.5, 11.5], "k", linewidth=2)
    # right border
    ax.plot([39, 39], [2, 8.5], "k", linewidth=2)
    ax.plot([39, 39], [11.5, 18], "k",linewidth=2)
    ax.plot([40, 39], [8.5, 8.5], "k", linewidth=2)
    ax.plot([40, 39], [11.5, 11.5], "k", linewidth=2)

    # inner scope
    # A
    ax.plot([5, 10], [6.5, 6.5], "k", linewidth=2)
    ax.plot([5, 10], [8.5, 8.5], "k", linewidth=2)
    ax.plot([5, 5], [6.5, 8.5],"k", linewidth=2)
    ax.plot([10, 10], [6.5, 7], "k", linewidth=2)
    ax.plot([10, 10], [8, 8.5], "k", linewidth=2)
    ax.plot([10, 13], [8, 8], "k", linewidth=2)
    ax.plot([10, 13], [7, 7], "k", linewidth=2)
    ax.plot([13, 13], [7, 8], "k", linewidth=2)
    # B
    ax.plot([5, 10], [13.5, 13.5], "k", linewidth=2)
    ax.plot([5, 10], [11.5, 11.5], "k", linewidth=2)
    ax.plot([5, 5], [11.5, 13.5], "k", linewidth=2)
    ax.plot([10, 10], [11.5, 12], "k", linewidth=2)
    ax.plot([10, 10], [13.5, 13], "k", linewidth=2)
    ax.plot([10, 13], [12, 12], "k", linewidth=2)
    ax.plot([10, 13], [13, 13], "k", linewidth=2)
    ax.plot([13, 13], [12, 13], "k", linewidth=2)

    # Rectangle
    rectangle1 = Rectangle((15, 12.5), 5, 2.5, facecolor="none", edgecolor="black", linewidth=2)
    rectangle2 = Rectangle((15, 5), 5, 2.5, facecolor="none", edgecolor="black", linewidth=2)
    rectangle3 = Rectangle((15, 9.5), 3, 1, facecolor="none", edgecolor="black", linewidth=2)
    ax.add_patch(rectangle1)
    ax.add_patch(rectangle2)
    ax.add_patch(rectangle3)

    # C
    ax.plot([35, 35], [5, 15], "k", linewidth=2)
    ax.plot([23, 35], [15, 15], "k", linewidth=2)
    ax.plot([23, 25], [12.5, 12.5], "k", linewidth=2)
    ax.plot([23, 23], [12.5, 15], "k", linewidth=2)
    ax.plot([25, 25], [7.5, 12.5], "k", linewidth=2)
    ax.plot([23, 25], [7.5, 7.5], "k", linewidth=2)
    ax.plot([23, 35], [5, 5], "k", linewidth=2)
    ax.plot([23, 23], [5, 7.5], "k", linewidth=2)

WALLS = [
    # Outer scope 
    (2, 2, 39, 2),        # Bottom wall
    (2, 18, 39, 18),      # Top wall

    # Left border
    (2, 2, 2, 8.5),
    (2, 11.5, 2, 18),
    (0, 8.5, 2, 8.5),
    (0, 11.5, 2, 11.5),

    # Right border
    (39, 2, 39, 8.5),
    (39, 11.5, 39, 18),
    (40, 8.5, 39, 8.5),
    (40, 11.5, 39, 11.5),

    # Room A (Bottom Left)
    (5, 6.5, 10, 6.5),
    (5, 8.5, 10, 8.5),
    (5, 6.5, 5, 8.5),
    (10, 6.5, 10, 7),
    # Continuing Room A
    (10, 8, 10, 8.5),
    (10, 8, 13, 8),
    (10, 7, 13, 7),
    (13, 7, 13, 8),

    # Room B (Top Left) 
    (5, 13.5, 10, 13.5),
    (5, 11.5, 10, 11.5),
    (5, 11.5, 5, 13.5),
    (10, 11.5, 10, 12),
    (10, 13.5, 10, 13),
    (10, 12, 13, 12),
    (10, 13, 13, 13),
    (13, 12, 13, 13),

    # Rectangles (Middle) 
    # Rectangle 1 (Top Middle): (15, 12.5), w=5, h=2.5
    (15, 12.5, 20, 12.5),
    (15, 15, 20, 15),
    (15, 12.5, 15, 15),
    (20, 12.5, 20, 15),

    # Rectangle 2 (Bottom Middle): (15, 5), w=5, h=2.5
    (15, 5, 20, 5),
    (15, 7.5, 20, 7.5),
    (15, 5, 15, 7.5),
    (20, 5, 20, 7.5),

    # Rectangle 3 (Center Small): (15, 9.5), w=3, h=1
    (15, 9.5, 18, 9.5),
    (15, 10.5, 18, 10.5),
    (15, 9.5, 15, 10.5),
    (18, 9.5, 18, 10.5),

    # Room C (Large Right Wall)
    (35, 5, 35, 15),
    (23, 15, 35, 15),
    (23, 12.5, 25, 12.5),
    (23, 12.5, 23, 15),
    (25, 7.5, 25, 12.5),
    (23, 7.5, 25, 7.5),
    (23, 5, 35, 5),
    (23, 5, 23, 7.5)
]