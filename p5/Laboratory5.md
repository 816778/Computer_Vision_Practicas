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

Cámara proyecta un punto 3D `X=(X,Y,Z,1)^T` en el espacio 2D `u=(u,v)^T`:  `u=KPX`
```python
def triangulate_points(uv1, uv2, T_wc1, T_wc2, K1, K2):
```
1. Desproyectar un punto u en la dirección de un rayo 3D

2. Calcular la transformación relativa entre dos cámaras
```python
T_c1c2 = np.linalg.inv(T_wc1) @ T_wc2
```
3. Triangulación de puntos individuales 

4. Conversión de coordenadas. Los puntos 3D se calculan inicialmente en el sistema de la cámara 2. Finalmente, se transforman al sistema de coordenadas de la cámara 1
```python
# Returns the point in C2 coordinates
points_3d[:, i] = triangulate_point(T_c1c2, v1, v2)
# To C1 coordinates
return (T_wc2 @ points_3d)[:3, :]
```


```python
def triangulate_point(T_c1c2, v1, v2):
```

1. Definir el Rayo con Dos Planos

Un rayo 3D v que pasa por un punto en el espacio puede ser descrito por dos planos
- Plano simétricos asociado al vector v (pasa por el origen y es perpecdicular a dirección derivada del vector)
- Plano perpendicular al rayo
```
Pi_sym_1, Pi_perp_1 = define_planes(v1)
Pi_sym_2, Pi_perp_2 = define_planes(v2)
```

2. Transformación de los planos

Los planos definidos en la cámara 1 se transforman a la cámara 2 mediante la matriz de tranformación extrínseca T_c1c2.

```python
Pi_sym_2_1 = T_c1c2.T @ Pi_sym_1
Pi_perp_2_1 = T_c1c2.T @ Pi_perp_1
```

3. Construcción de la matriz A

Para el punto 3D X esté en ambos rayos (ambas cámaras) debe satisfacer `AX=0`
```python
A = np.vstack((Pi_sym_2_1.T, Pi_perp_2_1.T, Pi_sym_2.T, Pi_perp_2.T))
```
4. Resolver sistemas

`U,S,V = sdv(A)` (última columna de V es solución para X en coordenadas homogéneas)

5. Validación del sistema: Rango es aproximadamente 3


### Bundle Adjustment Fisheye
```python
def residual_bundle_adjustment_fisheye(params, K_1, K_2, D1_k_array, D2_k_array, xData, T_wc1, T_wc2):
```
Calcula los residuos de reproyección para un conjunto de puntos 3D y transformaciones entre cámaras
1. Extraer parámetros del estado optimizable
2. Reconstruir la transformación TwAwB
3. Transformaciones para cámaras asociadas a wA y wB
```python
# Transformaciones del sistema mundial al sistema de cámaras asociadas a wA
# Cámaras 1 y 2 en el sistema wA están descritas directamente en el sistema mundial
T_wAc1 = T_wc1
T_wAc2 = T_wc2
# El sistema wB no está alineado con el sistema mundial (w). Su posición y orientación están definidas relativamente al sistema wA.
# T_wAwB permite convertir coordenadas del sistema wA al sistema wB
# T_wc1: Describe dónde está  c1 de wA en el sistema mundial.
# T_wAwB: Ajusta esta posición para reflejar el hecho de que estamos ahora en el sistema wB.
T_wBc1 = T_wAwB @ T_wc1
T_wBc2 = T_wAwB @ T_wc2
```
4. Proyectar puntos 3D a coordenadas de imagen
5. Calcular los residuos de reproyección

```python
def run_bundle_adjustment_fisheye(T_wAwB_seed, K_1, K_2, D1_k_array, D2_k_array, X_w, xData, T_wc1, T_wc2):
```

**Objetivo**: ptimizar los parámetros de la transformación relativa TwAwB y las posiciones de los puntos 3D

1. Inicialización de la transformación relativa TwAwB
2. Optimización con Bundle Adjustment
3. Extraer parámetros optimizados