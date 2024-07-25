import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation
import random
import os

def create_block_sequence(num_blocks=6):
    """
    Creates a sequence of connected blocks in 3D space, biased to run along an axis.
    
    Args:
    num_blocks (int): Number of blocks in the sequence.
    
    Returns:
    list: A list of tuples, each containing the position and color of a block.
    """
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    sequence = []
    
    # Choose a primary axis
    primary_axis = np.random.choice(['x', 'y', 'z'])
    if primary_axis == 'x':
        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    elif primary_axis == 'y':
        directions = [(0, 1, 0), (0, -1, 0), (1, 0, 0), (-1, 0, 0), (0, 0, 1), (0, 0, -1)]
    else:  # z-axis
        directions = [(0, 0, 1), (0, 0, -1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
    
    # Place the first block at the origin
    position = np.array([0, 0, 0])
    color = np.random.choice(colors)
    sequence.append((position, color))
    
    occupied_positions = set([tuple(position)])
    
    # Add remaining blocks
    for _ in range(1, num_blocks):
        color = np.random.choice(colors)
        
        valid_position_found = False
        while not valid_position_found:
            # Choose a random existing block to attach to
            attach_to = sequence[np.random.randint(len(sequence))][0]
            
            # Bias towards the primary axis
            if np.random.random() < 0.7:  # 70% chance to choose primary axis
                direction = directions[0] if np.random.random() < 0.5 else directions[1]
            else:
                direction = directions[np.random.randint(2, len(directions))]
            
            new_position = tuple(attach_to + direction)
            if new_position not in occupied_positions:
                sequence.append((np.array(new_position), color))
                occupied_positions.add(new_position)
                valid_position_found = True
    
    return sequence

def create_cube(position):
    """
    Creates the vertices and faces of a cube at the given position.
    
    Args:
    position (tuple): The (x, y, z) coordinates of the cube's origin.
    
    Returns:
    list: A list of faces, where each face is a list of vertex coordinates.
    """
    x, y, z = position
    vertices = [
        [x, y, z], [x+1, y, z], [x+1, y+1, z], [x, y+1, z],
        [x, y, z+1], [x+1, y, z+1], [x+1, y+1, z+1], [x, y+1, z+1]
    ]
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[3], vertices[7], vertices[4]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[3], vertices[2], vertices[6], vertices[7]]
    ]
    return faces

def rotate_structure(structure, angles):
    """
    Rotates the entire structure around multiple axes.
    
    Args:
    structure (list): List of faces representing the structure.
    angles (tuple): Rotation angles in degrees for (x, y, z) axes.
    
    Returns:
    list: Rotated structure.
    """
    rotation = Rotation.from_euler('xyz', angles, degrees=True)
    center = np.mean([np.mean(face, axis=0) for face in structure], axis=0)
    rotated_structure = []
    for face in structure:
        rotated_face = rotation.apply(face - center) + center
        rotated_structure.append(rotated_face)
    return rotated_structure

def plot_structure(structure, colors, ax):
    """
    Plots the 3D structure on the given axes.
    
    Args:
    structure (list): List of faces representing the structure.
    colors (list): List of colors for each face.
    ax (Axes3D): The 3D axes to plot on.
    """
    collection = Poly3DCollection(structure, facecolors=colors, edgecolors='k', linewidths=0.5)
    ax.add_collection3d(collection)
    
    # Set equal aspect ratio for all axes
    all_points = np.array([point for face in structure for point in face])
    max_range = np.array([all_points[:, 0].ptp(), all_points[:, 1].ptp(), all_points[:, 2].ptp()]).max()
    mid_x, mid_y, mid_z = np.mean(all_points, axis=0)
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    ax.set_box_aspect((1, 1, 1))

def create_structure(sequence):
    """
    Creates a 3D structure from a sequence of blocks.
    
    Args:
    sequence (list): List of (position, color) tuples representing blocks.
    
    Returns:
    tuple: (structure, colors) where structure is a list of faces and colors is a list of face colors.
    """
    structure = []
    colors = []
    for position, color in sequence:
        cube = create_cube(position)
        structure.extend(cube)
        colors.extend([color] * 6)  # 6 faces per cube
    return structure, colors

def save_individual_image(name, structure, colors, trial_num, base_angle):
    """
    Saves an individual image for a single structure.
    
    Args:
    name (str): Name of the image (e.g., 'target', 'answer', 'distractor1').
    structure (list): List of faces representing the structure.
    colors (list): List of colors for each face.
    trial_num (int): The current trial number.
    base_angle (int): The base rotation angle for this trial.
    """
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    plot_structure(structure, colors, ax)
    ax.set_axis_off()
    plt.tight_layout()
    
    # Create 'individual_images' folder if it doesn't exist
    if not os.path.exists('individual_images'):
        os.makedirs('individual_images')
    
    plt.savefig(f'individual_images/trial_{trial_num}_angle_{base_angle}_{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_combined_image(trial_num, base_angle):
    """
    Generates a combined image for each trial, showing the answer and distractors
    at the top and the target below.
    
    Args:
    trial_num (int): The current trial number.
    base_angle (int): The base rotation angle for this trial.
    """
    names = ['target', 'answer', 'distractor1', 'distractor2', 'distractor3']
    
    # Create the main sequence for target and answer
    main_sequence = create_block_sequence()
    main_structure, main_colors = create_structure(main_sequence)
    
    # Create distractor sequences with the same color pattern
    distractor_sequences = [create_block_sequence() for _ in range(3)]
    distractor_structures = [create_structure(seq)[0] for seq in distractor_sequences]
    
    # Generate random rotation angles for each axis
    answer_angles = np.random.uniform(-base_angle, base_angle, 3)
    distractor_angles = [np.random.uniform(-base_angle, base_angle, 3) for _ in range(3)]
    
    # Create a figure with a 3x2 grid
    fig = plt.figure(figsize=(18, 12))
    
    # Randomize the order of answer and distractors
    top_order = ['answer', 'distractor1', 'distractor2', 'distractor3']
    random.shuffle(top_order)
    
    # Plot the top row (answer and distractors)
    for i, name in enumerate(top_order):
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        
        if name == 'answer':
            rotation_angles = tuple(answer_angles)
            structure = main_structure
            colors = main_colors
        else:
            distractor_index = int(name[-1]) - 1
            rotation_angles = tuple(distractor_angles[distractor_index])
            structure = distractor_structures[distractor_index]
            colors = main_colors[:len(structure)]
        
        rotated_structure = rotate_structure(structure, rotation_angles)
        plot_structure(rotated_structure, colors, ax)
        ax.set_axis_off()
        
        # Save individual image
        save_individual_image(name, rotated_structure, colors, trial_num, base_angle)
    
    # Plot the target at the bottom center
    ax = fig.add_subplot(2, 4, (6, 7), projection='3d')
    plot_structure(main_structure, main_colors, ax)
    ax.set_axis_off()
    
    # Save individual target image
    save_individual_image('target', main_structure, main_colors, trial_num, base_angle)
    
    plt.tight_layout()
    plt.savefig(f'trial_{trial_num}_angle_{base_angle}_combined.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print rotation angles for reference
    print(f"Trial {trial_num} rotations:")
    print(f"Answer: {tuple(answer_angles)}")
    print(f"Distractors: {[tuple(angles) for angles in distractor_angles]}")
    print(f"Top row order: {top_order}")
    print()

# Generate and save combined images for 4 trials for each specified angle
base_angles = [30, 60, 90, 120, 150, 180]
for base_angle in base_angles:
    for trial in range(1, 5):
        create_combined_image(trial, base_angle)

print("Combined and individual images generated successfully!")