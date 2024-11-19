# Lab 5: Omnidirectional Vision


## 2. 3D reconstruction from 3D calibrated stereo using fish-eyes

### Kannala-Brandt projection

Implementa la proyección de puntos 3D en el sistema de coordenadas de la cámara al plano de imagen de una cámara con lente ojo de pez utilizando el modelo de proyección de Kannala-Brandt. Tiene en cuenta la distorsión radial significativa introducida por las lentes ojo de pez.

```python
def kannala_brandt_projection(X, K, D):
```

K contiene: 
- `alpha_x` y `alpha_y`: Los factores de escala (focales) en las direcciones x e y
- `c_x` y `c_y`: El centro óptico o punto principal, que indica el desplazamiento en píxeles del centro de la imagen.

X: Matriz de puntos 3D, cada columna representa un punto [x,y,z] (coordenadas de cada punto)

D: Array que contiene coeficientes de distorsión radial

Calcular R: Radio de cada ounto al eje óptico (Z)

Calcular theta: Ángulo que forma el punto 3D con el eje óptico (z)

Aplicar distorisión radial d_theta: Polinomio modela cómo las lentes ojo de pez deforman las líneas rectas.

Calcular phi: Ángulo azimutal en el plano XY, que representa la dirección angular del punto 3D respecto al eje x

u,v: Coordenadas proyectadas del punto en el plano de la imagen

### Kannala-Brandt unprojection


```python
def kannala_brandt_unprojection_roots(u, K, D):
```

Implementa la desproyección de puntos desde coordenadas 2D en el plano de imagen (con distorsión ojo de pez) al espacio 3D, utilizando el modelo de Kannala-Brandt. 


K: Matriz intrínseca de la cámara

Llevar a coordenadas normalizadas en el plano de la cámara, pero todavía no tienen en cuenta la distorsión radial.

Calcular el radio y el ángulo azimutal

Resolver el polinomio para encontrar theta

Seleccionar las raíces reales y seleccionar la mejor

Calcular las direcciones en espacio 3D


### Triangulation Algorithm using planes

Se utiliza para encontrar un punto X n el espacio 3D que corresponde a sus proyecciones x1 y x2 en dos cámaras diferentes. 

Representar los rayos de proyección de las cámaras mediante dos planos:

- Plano simétrico
- Plano perpendicular

```python
def triangulate_point(directions1, directions2, T_wc1, T_wc2, T_c1c2):
```

1. Definir el Rayo con Dos Planos

Un rayo 3D v que pasa por un punto en el espacio puede ser descrito por dos planos
- Plano simétricos respecto al eje Z
- Plano perpendicular al rayo

2. Transformación de los planos

Los planos definidos en la cámara 1 se transforman a la cámara 2 mediante la matriz de tranformación extrínseca T_21.

3. Construcción de la matriz A

Para el punto 3D X esté en ambos rayos (ambas cámaras) debe satisfacer `AX=0`

4. Resolver sistemas

`U,S,V = sdv(A)` (última columna de V es solución para X en coordenadas homogéneas)

5. Validación del sistema: Rango es aproximadamente 3


