import matplotlib.pyplot as plt
import numpy as np


def draw_young_tableau_with_numbers(numbers):
    """
    Draw a Young tableau with numbers in the cells given the numbers.
    
    Parameters:
    numbers (list of list of int): Nested list where each inner list represents the numbers in the corresponding row.
    """
    # Calculate the shape from the numbers
    shape = [len(row) for row in numbers]
    
    # Determine the size of the grid
    max_row_length = max(shape)
    num_rows = len(shape)
    
    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Draw the boxes and numbers
    for i, row_length in enumerate(shape):
        for j in range(row_length):
            # Draw the box
            rect = plt.Rectangle((j, -i-1), 1, 1, fill=None, edgecolor='black')
            ax.add_patch(rect)
            
            # Add the number
            ax.text(j + 0.5, -i - 0.5, str(numbers[i][j]), ha='center', va='center', fontsize=40)
    
    # Set the limits and aspect ratio
    ax.set_xlim(0, max_row_length+1)
    ax.set_ylim(-num_rows-1, 0)
    ax.set_aspect('equal')
    
    # Remove axes
    ax.axis('off')
    
    # Show the plot
    plt.savefig('young_tableau.png')

# Example usage:
# The numbers are specified as a nested list.
draw_young_tableau_with_numbers([[1, 2, 3, 4], [5, 6, 7], [8, 10]])
