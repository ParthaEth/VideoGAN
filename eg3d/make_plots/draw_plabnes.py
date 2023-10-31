import numpy as np
import matplotlib.pyplot as plt

# Define the range for the planes
range_limit = 5

# Create points for the XY plane
xy_plane = np.array([[0, 0, 0], [range_limit, 0, 0], [0, range_limit, 0]])
x, y, z = xy_plane.T

# Create points for the YZ plane
yz_plane = np.array([[0, 0, 0], [0, range_limit, 0], [0, 0, range_limit]])
x2, y2, z2 = yz_plane.T

# Create points for the XZ plane
xz_plane = np.array([[0, 0, 0], [range_limit, 0, 0], [0, 0, range_limit]])
x3, y3, z3 = xz_plane.T

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the XY plane
ax.plot(x, y, z, 'r-')

# Plot the YZ plane
ax.plot(x2, y2, z2, 'g-')

# Plot the XZ plane
ax.plot(x3, y3, z3, 'b-')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the title
plt.title('XY, YZ, and XZ Planes Passing Through Origin')

# Set the limits for the plot
ax.set_xlim(-range_limit, range_limit)
ax.set_ylim(-range_limit, range_limit)
ax.set_zlim(-range_limit, range_limit)

# Show the plot
plt.show()
