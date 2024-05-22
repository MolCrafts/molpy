# Box

Simulation box is a fundamental concept in molecular simulations. It is a parallelepiped that contains all the atoms, and also used to define the periodic boundary conditions. Orthogonal box is the most common type, which edges are perpendicular to each other. Triclinic box's edges are not perpendicular to each other. The representation of those two boxes can be unified by one class, `Box`. To make tutorial more clear, we split the discussion into two parts.

The core of `Box` is a matrix represents 3 edges vectors of the box. For orthogonal box, the matrix is a diagonal matrix,

$$
\begin{bmatrix}
xhi-xlo & 0 & 0 \\
0 & yhi-ylo & 0 \\
0 & 0 & zhi-zlo
\end{bmatrix}
$$

where $(xlo, ylo, zlo)$ is lower bound and "origin" of the box. 

A triclinic box is defined with 3 arbitrary edges vectors $\mathbf A$, $\mathbf B$, and $\mathbf C$, so long as they are non-zero, distinct, and not co-planar. In addition, they must define a right-handed system, such that ($\mathbf A$ cross $\mathbf B$) points in the direction of C. Note that a left-handed system can be converted to a right-handed system by simply swapping the order of any pair of the $\mathbf A$, $\mathbf B$, $\mathbf C$ vectors. The box can be done in two ways.

A *general* triclinic box is specified by an origin (xlo, ylo, zlo) and arbitrary edge vectors A = (ax,ay,az), B = (bx,by,bz), and C = (cx,cy,cz). So there are 12 parameters in total.

A *restricted* triclinic box also has an origin (xlo,ylo,zlo), but its edge vectors are of the following restricted form: A = (xhi-xlo,0,0), B = (xy,yhi-ylo,0), C = (xz,yz,zhi-zlo). So there are 9 parameters in total. The *restricted* requires $\mathbf A$ to be along the x-axis, B to be in the xy plane with a y-component in the +y direction, and C to have its z-component in the +z direction. Note that a restricted triclinic box is right-handed by construction since (A cross B) points in the direction of C.

The xy,xz,yz values can be zero or positive or negative. They are called “tilt factors” because they are the amount of displacement applied to edges of faces of an orthogonal box to change it into a restricted triclinic parallelepiped.

## Transformation from general to restricted triclinic boxes

Let $\mathbf A$, $\mathbf B$, $\mathbf C$ be the right-handed edge vectors of a general triclinic simulation box. $\mathbf a$, $\mathbf b$, $\mathbf c$ for a restricted triclinic box are a 3d rotation of $\mathbf A$, $\mathbf B$, and $\mathbf C$ and can be computed as follows:

$$
(\mathbf a, \mathbf b, \mathbf c) = \begin{bmatrix}
a_x, b_x, c_x \\
0, b_y, c_y \\
0, 0, c_z
\end{bmatrix}
$$

$$
\begin{align*}
a_x =& A \\
b_x =& B \cdot \hat{\mathbf A} = B \cos \gamma \\
b_y =& |\hat{\mathbf A} \times{\mathbf B}| = B \sin \gamma = \sqrt{B^2 - b_x^2} \\
c_x =& \mathbf C \cdot \hat{\mathbf A} = C \cos \beta \\
c_y =& \mathbf C \cdot \widehat{(\mathbf A \times \mathbf B)} \times \hat{\mathbf A} = \frac{\mathbf B \cdot \mathbf C - b_x c_x}{b_y} \\
c_z =& |\mathbf C \cdot \widehat{(\mathbf A \times \mathbf B)}| = \sqrt{C^2 - c_x^2 - c_y^2}
\end{align*}
$$

where $A = |\mathbf A|$ indicates the scalar length of $\mathbf A$. The hat symbol ( $\hat{\ }$ ) indicates the corresponding unit vector. $\beta$ and $\gamma$ are angles between the $\mathbf A$ $\mathbf B$ and $\mathbf A$ $\mathbf C$ vectors.

For consistency, the same rotation applied to the triclinic box edge vectors can also be applied to atom positions, velocities, and other vector quantities. This can be conveniently achieved by first converting to fractional coordinates in the general triclinic coordinates and then converting to coordinates in the restricted triclinic basis. The transformation is given by the following equation:

$$
\mathbf x = (\mathbf a, \mathbf b, \mathbf c) \cdot \frac{1}{V} \begin{bmatrix}
\mathbf B \times \mathbf C \\
\mathbf C \times \mathbf A \\
\mathbf A \times \mathbf B
\end{bmatrix}
\cdot \mathbf X
$$

where V is the volume of the box (same in either basis), X is the fractional vector in the general triclinic basis and x is the resulting vector in the restricted triclinic basis.

## Crystallographic general triclinic representation of a simulation box

General triclinic crystal structures are often defined using three lattice constants a, b, and c, and three angles $\alpha$, $\beta$, and $\gamma$. Note that in this nomenclature, the a, b, and c lattice constants are the scalar lengths of the edge vectors a, b, and c defined above. The relationship between these 6 quantities (a, b, c, $\alpha$, $\beta$, $\gamma$) and the LAMMPS restricted triclinic box sizes (lx,ly,lz) = (xhi-xlo,yhi-ylo,zhi-zlo) and tilt factors (xy,xz,yz) is as follows: