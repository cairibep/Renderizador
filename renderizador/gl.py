#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: Andre e Caio
Disciplina: Computação Gráfica
Data: 17/08
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante

    model_matrix = np.identity(4)
    view_matrix = np.identity(4)
    projection_matrix = np.identity(4)
    transform_stack = []

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width * 2 # para supersampling 2x2
        GL.height = height * 2
        GL.near = near
        GL.far = far

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polypoint2D
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é a
        # coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista e assuma que sempre vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polypoint2D
        # você pode assumir inicialmente o desenho dos pontos com a cor emissiva (emissiveColor).

        emissive = colors.get("emissiveColor", [1.0, 1.0, 1.0])
        color = [int(255 * c) for c in emissive]

        # Para cada par (x, y) em point
        for i in range(0, len(point), 2):
            gpu.GPU.draw_pixel([int(point[i]), int(point[i+1])], gpu.GPU.RGB8, color)
        
    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Polyline2D
        # Nessa função você receberá os pontos de uma linha no parâmetro lineSegments, esses
        # pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o valor da
        # coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto. Já point[2] é
        # a coordenada x do segundo ponto e assim por diante. Assuma a quantidade de pontos
        # pelo tamanho da lista. A quantidade mínima de pontos são 2 (4 valores), porém a
        # função pode receber mais pontos para desenhar vários segmentos. Assuma que sempre
        # vira uma quantidade par de valores.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Polyline2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).

        emissive = colors.get("emissiveColor", [1.0, 1.0, 1.0])
        color = [int(255 * c) for c in emissive]

        # Para cada segmento de linha (ponto i até ponto i+1)
        for i in range(0, len(lineSegments) - 2, 2):
            x0 = int(lineSegments[i])
            y0 = int(lineSegments[i+1])
            x1 = int(lineSegments[i+2])
            y1 = int(lineSegments[i+3])

            # Algoritmo simples de linha (DDA)
            dx = x1 - x0
            dy = y1 - y0
            steps = max(abs(dx), abs(dy))
            if steps == 0:
                if 0 <= x0 < GL.width and 0 <= y0 < GL.height:
                    gpu.GPU.draw_pixel([x0, y0], gpu.GPU.RGB8, color)
                continue
            x_inc = dx / steps
            y_inc = dy / steps
            x = x0
            y = y0
            for _ in range(steps + 1):
                px = int(round(x))
                py = int(round(y))
                if 0 <= px < GL.width and 0 <= py < GL.height:
                    gpu.GPU.draw_pixel([px, py], gpu.GPU.RGB8, color)
                x += x_inc
                y += y_inc

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#Circle2D
        # Nessa função você receberá um valor de raio e deverá desenhar o contorno de
        # um círculo.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o Circle2D
        # você pode assumir o desenho das linhas com a cor emissiva (emissiveColor).
        
        emissive_color = colors.get('emissiveColor', [1.0, 1.0, 1.0])
        r = int(emissive_color[0] * 255)
        g = int(emissive_color[1] * 255)
        b = int(emissive_color[2] * 255)
        
        # Converte radius para inteiro
        radius = int(radius)
        
        # Centro do círculo (centro da tela)
        center_x = GL.width // 2
        center_y = GL.height // 2
        
        # Ajuste na tela
        max_radius = min(center_x, center_y) - 1
        if radius > max_radius:
            radius = max_radius
        
        # Segmentos
        num_segments = 360
        angle_step = 2 * math.pi / num_segments
        
        for i in range(num_segments):
            # Pontos com as coordenadas
            angle1 = i * angle_step
            angle2 = (i + 1) * angle_step
            
            x1 = center_x + radius * math.cos(angle1)
            y1 = center_y + radius * math.sin(angle1)
            x2 = center_x + radius * math.cos(angle2)
            y2 = center_y + radius * math.sin(angle2)
            
            dx = x2 - x1
            dy = y2 - y1
            steps = max(abs(dx), abs(dy))
            
            if steps == 0:
                continue
                
            x_inc = dx / steps
            y_inc = dy / steps
            x = x1
            y = y1
            
            for _ in range(int(steps) + 1):
                px = int(round(x))
                py = int(round(y))
                if 0 <= px < GL.width and 0 <= py < GL.height:
                    gpu.GPU.draw_pixel([px, py], gpu.GPU.RGB8, [r, g, b])
                x += x_inc
                y += y_inc

    @staticmethod
    def triangleSet2D(vertices, colors):
        """Função usada para renderizar TriangleSet2D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry2D.html#TriangleSet2D
        # Nessa função você receberá os vertices de um triângulo no parâmetro vertices,
        # esses pontos são uma lista de pontos x, y sempre na ordem. Assim point[0] é o
        # valor da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto.
        # Já point[2] é a coordenada x do segundo ponto e assim por diante. Assuma que a
        # quantidade de pontos é sempre multiplo de 3, ou seja, 6 valores ou 12 valores, etc.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, para o TriangleSet2D
        # você pode assumir inicialmente o desenho das linhas com a cor emissiva (emissiveColor).
        
        emissive_color = colors.get('emissiveColor', [1.0, 1.0, 1.0])
        r = int(emissive_color[0] * 255)
        g = int(emissive_color[1] * 255)
        b = int(emissive_color[2] * 255)
        
        # Processa triângulos de 3 em 3 pontos
        for i in range(0, len(vertices), 6):
            if i + 5 < len(vertices):
                p0 = (int(vertices[i] * 2), int(vertices[i + 1] * 2)) # para supersampling 2x2
                p1 = (int(vertices[i + 2] * 2), int(vertices[i + 3] * 2))
                p2 = (int(vertices[i + 4] * 2), int(vertices[i + 5] * 2))
                
                # Desenha o triângulo usando o algoritmo de preenchimento
                GL.draw_triangle((p0, p1, p2), [[r, g, b], [r, g, b], [r, g, b]], [1.0, 1.0, 1.0])

    @staticmethod
    def draw_triangle(pts, colors, z_values):
        (x0, y0), (x1, y1), (x2, y2) = pts
        c0, c1, c2 = colors
        z0, z1, z2 = z_values

        min_x = max(0, min(x0, x1, x2))
        max_x = min(GL.width - 1, max(x0, x1, x2))
        min_y = max(0, min(y0, y1, y2))
        max_y = min(GL.height - 1, max(y0, y1, y2))

        # Área total para coordenadas baricêntricas
        def edge_fn(x0, y0, x1, y1, x2, y2):
            return (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)

        area = edge_fn(x0, y0, x1, y1, x2, y2)

        if area == 0:
            return  # triângulo degenerado

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                w0 = edge_fn(x1, y1, x2, y2, x, y)
                w1 = edge_fn(x2, y2, x0, y0, x, y)
                w2 = edge_fn(x0, y0, x1, y1, x, y)

                if (w0 >= 0 and w1 >= 0 and w2 >= 0) or (w0 <= 0 and w1 <= 0 and w2 <= 0):
                    alpha = w0 / area
                    beta = w1 / area
                    gamma = w2 / area

                    one_over_z = alpha / z0 + beta / z1 + gamma / z2 # corrigir a interpolação com profundidade (Z)
                    z = 1 / one_over_z

                    def interp_channel(i):
                        v0 = c0[i] / z0
                        v1 = c1[i] / z1
                        v2 = c2[i] / z2
                        return int(z * (alpha * v0 + beta * v1 + gamma * v2))

                    r = interp_channel(0)
                    g = interp_channel(1)
                    b = interp_channel(2)

                    z_buffer_value = gpu.GPU.read_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F)

                    if z < z_buffer_value:
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, [r, g, b])
                        gpu.GPU.draw_pixel([x, y], gpu.GPU.DEPTH_COMPONENT32F, [z])

    @staticmethod
    def triangleSet(point, colors):
        """Função usada para renderizar TriangleSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleSet
        # Nessa função você receberá pontos no parâmetro point, esses pontos são uma lista
        # de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x do
        # primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e
        # assim por diante.
        # No TriangleSet os triângulos são informados individualmente, assim os três
        # primeiros pontos definem um triângulo, os três próximos pontos definem um novo
        # triângulo, e assim por diante.
        # O parâmetro colors é um dicionário com os tipos cores possíveis, você pode assumir
        # inicialmente, para o TriangleSet, o desenho das linhas com a cor emissiva
        # (emissiveColor), conforme implementar novos materias você deverá suportar outros
        # tipos de cores.
        emissive = colors.get("emissiveColor", [1.0, 1.0, 1.0])
        color = [int(255 * c) for c in emissive]
        for i in range(0, len(point), 9):
            screen_pts = []
            screen_colors = []
            z_values = []
            for j in range(3):
                idx = i + j*3
                p = np.array([point[idx], point[idx+1], point[idx+2], 1])
                p = GL.model_matrix @ p
                p = GL.view_matrix @ p
                p = GL.projection_matrix @ p
                p /= p[3]

                z_values.append(p[2])  # interpolação com Z

                screen_matrix = np.array([
                    [GL.width / 2,      0,               0, GL.width / 2],
                    [0,        -GL.height / 2,           0, GL.height / 2],
                    [0,                 0,               1,             0],
                    [0,                 0,               0,             1],
                ])
                p_screen = screen_matrix @ p
                x = int(p_screen[0])
                y = int(p_screen[1])
                screen_pts.append((x, y))

                emissive = colors.get("emissiveColor", [1.0, 1.0, 1.0])
                screen_colors.append([int(255 * c) for c in emissive])

            GL.draw_triangle(screen_pts, screen_colors, z_values)

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        # Na função de viewpoint você receberá a posição, orientação e campo de visão da
        # câmera virtual. Use esses dados para poder calcular e criar a matriz de projeção
        # perspectiva para poder aplicar nos pontos dos objetos geométricos.
        eye = np.array(position)
        at = np.array([0, 0, -1])
        up = np.array([0, 1, 0])

        # aplica orientacao ao at e up
        x, y, z, angle = orientation
        R = GL.rotation_matrix_quaternion_axis_angle([x, y, z], angle)
        at = R @ at
        
        up = R @ up
        at = eye + at

        w = at - eye
        w = w / np.linalg.norm(w)
        u = np.cross(w, up)
        u = u / np.linalg.norm(u)
        v = np.cross(u, w)
        v = v / np.linalg.norm(v)

        # monta a matriz camera-to-world
        m = np.identity(4)
        m[0, :3] = [u[0], v[0], -w[0]]
        m[1, :3] = [u[1], v[1], -w[1]]
        m[2, :3] = [u[2], v[2], -w[2]]
        m[:3, 3] = eye
        # inverte para obter world-to-camera
        GL.view_matrix = np.linalg.inv(m) # matriz lookat

        aspect_ratio = GL.width / GL.height
        fovy = fieldOfView
        near = GL.near
        far = GL.far

        top = near * math.tan(fovy / 2)
        right = top * aspect_ratio

        GL.projection_matrix = np.array([
            [near/right, 0, 0, 0],
            [0, near/top, 0, 0],
            [0, 0, -(far+near)/(far-near), -(2*far*near)/(far-near)],
            [0, 0, -1, 0]
        ])

    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_in será chamada quando se entrar em um nó X3D do tipo Transform
        # do grafo de cena. Os valores passados são a escala em um vetor [x, y, z]
        # indicando a escala em cada direção, a translação [x, y, z] nas respectivas
        # coordenadas e finalmente a rotação por [x, y, z, t] sendo definida pela rotação
        # do objeto ao redor do eixo x, y, z por t radianos, seguindo a regra da mão direita.
        # ESSES NÃO SÃO OS VALORES DE QUATÉRNIOS AS CONTAS AINDA PRECISAM SER FEITAS.
        # Quando se entrar em um nó transform se deverá salvar a matriz de transformação dos
        # modelos do mundo para depois potencialmente usar em outras chamadas. 
        # Quando começar a usar Transforms dentre de outros Transforms, mais a frente no curso
        # Você precisará usar alguma estrutura de dados pilha para organizar as matrizes.

        GL.transform_stack.append(GL.model_matrix.copy())

        t = np.identity(4)
        if translation:
            tx, ty, tz = translation
            t = np.array([
                [1, 0, 0, tx],
                [0, 1, 0, ty],
                [0, 0, 1, tz],
                [0, 0, 0, 1]
            ])

        s = np.identity(4)
        if scale:
            sx, sy, sz = scale
            s = np.array([
                [sx, 0, 0, 0],
                [0, sy, 0, 0],
                [0, 0, sz, 0],
                [0, 0, 0, 1]
            ])
            
        r = np.identity(4)
        if rotation: # Rotação usando quatérnios
            x, y, z, angle = rotation
            rotation_matrix = GL.rotation_matrix_quaternion_axis_angle([x, y, z], angle)
            r = np.eye(4)
            r[:3, :3] = rotation_matrix

        GL.model_matrix = GL.model_matrix @ t @ r @ s # usa a matriz de transformação atual e multiplica pela nova transformação

    @staticmethod
    def rotation_matrix_quaternion_axis_angle(axis, theta):
        """Cria a matriz de rotação 3x3 a partir de cálculos com eixo e ângulo envolvendo quatérnios."""
        axis = np.array(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        half_theta = theta / 2
        q_r = np.cos(half_theta)
        q_xyz = axis * np.sin(half_theta)
        q = np.array([q_r, *q_xyz])
        qr, qi, qj, qk = q
        return np.array([
            [1 - 2*(qj**2 + qk**2),   2*(qi*qj - qk*qr),     2*(qi*qk + qj*qr)],
            [2*(qi*qj + qk*qr),       1 - 2*(qi**2 + qk**2), 2*(qj*qk - qi*qr)],
            [2*(qi*qk - qj*qr),       2*(qj*qk + qi*qr),     1 - 2*(qi**2 + qj**2)]
        ])

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        # A função transform_out será chamada quando se sair em um nó X3D do tipo Transform do
        # grafo de cena. Não são passados valores, porém quando se sai de um nó transform se
        # deverá recuperar a matriz de transformação dos modelos do mundo da estrutura de
        # pilha implementada.

        if GL.transform_stack:
            GL.model_matrix = GL.transform_stack.pop()

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#TriangleStripSet
        # A função triangleStripSet é usada para desenhar tiras de triângulos interconectados,
        # você receberá as coordenadas dos pontos no parâmetro point, esses pontos são uma
        # lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor da coordenada x
        # do primeiro ponto, point[1] o valor y do primeiro ponto, point[2] o valor z da
        # coordenada z do primeiro ponto. Já point[3] é a coordenada x do segundo ponto e assim
        # por diante. No TriangleStripSet a quantidade de vértices a serem usados é informado
        # em uma lista chamada stripCount (perceba que é uma lista). Ligue os vértices na ordem,
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        color_per_vertex = True  # Por padrão, assume que há cor por vértice
        emissive = colors.get("emissiveColor", [1.0, 1.0, 1.0])
        default_rgb = [int(255 * c) for c in emissive]

        index = 0
        for count in stripCount:
            for i in range(count - 2): 
                screen_pts = []
                screen_colors = []
                z_values = []

                for j in range(3):  
                    idx = index + i + j
                    x, y, z = point[3*idx], point[3*idx+1], point[3*idx+2]
                    v = np.array([x, y, z, 1])
                    v = GL.model_matrix @ v
                    v = GL.view_matrix @ v
                    v = GL.projection_matrix @ v
                    v /= v[3]

                    z_values.append(v[2])

                    screen_matrix = np.array([
                        [GL.width / 2, 0, 0, GL.width / 2],
                        [0, -GL.height / 2, 0, GL.height / 2],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]
                    ])
                    v = screen_matrix @ v
                    screen_pts.append((int(v[0]), int(v[1])))

                    screen_colors.append(default_rgb)

                if i % 2 == 1:
                    screen_pts[1], screen_pts[2] = screen_pts[2], screen_pts[1]
                    screen_colors[1], screen_colors[2] = screen_colors[2], screen_colors[1]
                    z_values[1], z_values[2] = z_values[2], z_values[1]

                GL.draw_triangle(screen_pts, screen_colors, z_values)
            index += count

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/rendering.html#IndexedTriangleStripSet
        # A função indexedTriangleStripSet é usada para desenhar tiras de triângulos
        # interconectados, você receberá as coordenadas dos pontos no parâmetro point, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim point[0] é o valor
        # da coordenada x do primeiro ponto, point[1] o valor y do primeiro ponto, point[2]
        # o valor z da coordenada z do primeiro ponto. Já point[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedTriangleStripSet uma lista informando
        # como conectar os vértices é informada em index, o valor -1 indica que a lista
        # acabou. A ordem de conexão será de 3 em 3 pulando um índice. Por exemplo: o
        # primeiro triângulo será com os vértices 0, 1 e 2, depois serão os vértices 1, 2 e 3,
        # depois 2, 3 e 4, e assim por diante. Cuidado com a orientação dos vértices, ou seja,
        # todos no sentido horário ou todos no sentido anti-horário, conforme especificado.

        emissive = colors.get("emissiveColor", [1.0, 1.0, 1.0])
        default_rgb = [int(255 * c) for c in emissive]
        strip = []

        for i in index:
            if i == -1:
                for j in range(len(strip) - 2):
                    screen_pts = []
                    screen_colors = []
                    z_values = []

                    for k in range(3):
                        idx = strip[j + k]
                        x, y, z = point[3*idx], point[3*idx+1], point[3*idx+2]
                        v = np.array([x, y, z, 1])
                        v = GL.model_matrix @ v
                        v = GL.view_matrix @ v
                        v = GL.projection_matrix @ v
                        v /= v[3]

                        z_values.append(v[2])

                        screen_matrix = np.array([
                            [GL.width / 2, 0, 0, GL.width / 2],
                            [0, -GL.height / 2, 0, GL.height / 2],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
                        ])
                        v = screen_matrix @ v
                        screen_pts.append((int(v[0]), int(v[1])))
                        screen_colors.append(default_rgb)

                    if j % 2 == 1:
                        screen_pts[1], screen_pts[2] = screen_pts[2], screen_pts[1]
                        screen_colors[1], screen_colors[2] = screen_colors[2], screen_colors[1]
                        z_values[1], z_values[2] = z_values[2], z_values[1]

                    GL.draw_triangle(screen_pts, screen_colors, z_values)
                strip = []
            else:
                strip.append(i)

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex, texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#IndexedFaceSet
        # A função indexedFaceSet é usada para desenhar malhas de triângulos. Ela funciona de
        # forma muito simular a IndexedTriangleStripSet porém com mais recursos.
        # Você receberá as coordenadas dos pontos no parâmetro cord, esses
        # pontos são uma lista de pontos x, y, e z sempre na ordem. Assim coord[0] é o valor
        # da coordenada x do primeiro ponto, coord[1] o valor y do primeiro ponto, coord[2]
        # o valor z da coordenada z do primeiro ponto. Já coord[3] é a coordenada x do
        # segundo ponto e assim por diante. No IndexedFaceSet uma lista de vértices é informada
        # em coordIndex, o valor -1 indica que a lista acabou.
        # A ordem de conexão não possui uma ordem oficial, mas em geral se o primeiro ponto com os dois
        # seguintes e depois este mesmo primeiro ponto com o terçeiro e quarto ponto. Por exemplo: numa
        # sequencia 0, 1, 2, 3, 4, -1 o primeiro triângulo será com os vértices 0, 1 e 2, depois serão
        # os vértices 0, 2 e 3, e depois 0, 3 e 4, e assim por diante, até chegar no final da lista.
        # Adicionalmente essa implementação do IndexedFace aceita cores por vértices, assim
        # se a flag colorPerVertex estiver habilitada, os vértices também possuirão cores
        # que servem para definir a cor interna dos poligonos, para isso faça um cálculo
        # baricêntrico de que cor deverá ter aquela posição. Da mesma forma se pode definir uma
        # textura para o poligono, para isso, use as coordenadas de textura e depois aplique a
        # cor da textura conforme a posição do mapeamento. Dentro da classe GPU já está
        # implementadado um método para a leitura de imagens.
        face = []
        for i in coordIndex:
            if i == -1:
                for j in range(1, len(face) - 1):  # triângulos: 0,1,2 ; 0,2,3 ; ...
                    triangle = [face[0], face[j], face[j + 1]]
                    screen_pts = []
                    screen_colors = []
                    z_values = []

                    for k, idx in enumerate(triangle):
                        x, y, z = coord[3*idx], coord[3*idx+1], coord[3*idx+2] # transformação do vértice
                        p = np.array([x, y, z, 1])
                        p = GL.model_matrix @ p
                        p = GL.view_matrix @ p
                        p = GL.projection_matrix @ p
                        p /= p[3]

                        z_values.append(p[2])  # correção de perspectiva

                        screen_matrix = np.array([
                            [GL.width / 2, 0, 0, GL.width / 2],
                            [0, -GL.height / 2, 0, GL.height / 2],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]
                        ])
                        p_screen = screen_matrix @ p
                        screen_pts.append((int(p_screen[0]), int(p_screen[1])))

                        if colorPerVertex and color:
                            if colorIndex:
                                try:
                                    c_idx = coordIndex.index(idx)
                                    c_idx = colorIndex[c_idx] if c_idx < len(colorIndex) else idx
                                except ValueError:
                                    c_idx = idx
                            else:
                                c_idx = idx

                            if 3 * c_idx + 2 < len(color):
                                rgb = color[3 * c_idx : 3 * c_idx + 3]
                                if len(rgb) < 3:
                                    rgb += [0] * (3 - len(rgb))
                            else:
                                rgb = colors.get("emissiveColor", [1.0, 1.0, 1.0])
                        else:
                            rgb = colors.get("emissiveColor", [1.0, 1.0, 1.0])

                        rgb = [int(255 * c) for c in rgb]
                        screen_colors.append(rgb)

                    GL.draw_triangle(screen_pts, screen_colors, z_values)
                face = []
            else:
                face.append(i)

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Box : size = {0}".format(size)) # imprime no terminal pontos
        print("Box : colors = {0}".format(colors)) # imprime no terminal as cores

        # Exemplo de desenho de um pixel branco na coordenada 10, 10
        gpu.GPU.draw_pixel([10, 10], gpu.GPU.RGB8, [255, 255, 255])  # altera pixel

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Sphere
        # A função sphere é usada para desenhar esferas na cena. O esfera é centrada no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da esfera que está sendo criada. Para desenha essa esfera você vai
        # precisar tesselar ela em triângulos, para isso encontre os vértices e defina
        # os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Sphere : radius = {0}".format(radius)) # imprime no terminal o raio da esfera
        print("Sphere : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/navigation.html#NavigationInfo
        # O campo do headlight especifica se um navegador deve acender um luz direcional que
        # sempre aponta na direção que o usuário está olhando. Definir este campo como TRUE
        # faz com que o visualizador forneça sempre uma luz do ponto de vista do usuário.
        # A luz headlight deve ser direcional, ter intensidade = 1, cor = (1 1 1),
        # ambientIntensity = 0,0 e direção = (0 0 −1).

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("NavigationInfo : headlight = {0}".format(headlight)) # imprime no terminal

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#DirectionalLight
        # Define uma fonte de luz direcional que ilumina ao longo de raios paralelos
        # em um determinado vetor tridimensional. Possui os campos básicos ambientIntensity,
        # cor, intensidade. O campo de direção especifica o vetor de direção da iluminação
        # que emana da fonte de luz no sistema de coordenadas local. A luz é emitida ao
        # longo de raios paralelos de uma distância infinita.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""
