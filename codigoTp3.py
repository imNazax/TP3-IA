# Prototipo de red de Hopfield para identificación de patrones simples con ruido
# Basado en una matriz de 10x10 (100 neuronas)

import numpy as np
import matplotlib.pyplot as plt

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        # Aplicar regla de Hebb
        for pattern in patterns:
            pattern = np.array(pattern)
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)  # No autoconexiones

    def recover(self, pattern, steps=10):
        pattern = np.array(pattern)
        for _ in range(steps):
            pattern = np.sign(self.weights @ pattern)
        return pattern

    def mostrar_patron(self, pattern, titulo=""):
        grid = np.array(pattern).reshape((10, 10))
        plt.imshow(grid, cmap="Greys")
        plt.title(titulo)
        plt.axis("off")
        plt.show()

def binario_a_bipolar(p):
    return [1 if x == 1 else -1 for x in p]

def bipolar_a_binario(p):
    return [1 if x == 1 else 0 for x in p]

def mutar_patron(patron, porcentaje_ruido):
    patron_mutado = patron.copy()
    total = len(patron)
    cantidad = int(total * porcentaje_ruido)
    indices = np.random.choice(total, cantidad, replace=False)
    for i in indices:
        patron_mutado[i] *= -1  # Invertir bit bipolar
    return patron_mutado

# ------------------- EJECUCIÓN -------------------
if __name__ == "__main__":
    # Definir un patrón simple (ej. damero 10x10)
    patron_original = [1 if (i + j) % 2 == 0 else -1 for i in range(10) for j in range(10)]

    # Inicializar red y entrenar
    red = HopfieldNetwork(100)
    red.train([patron_original])

    # Mutar patrón (ruido del 30%)
    patron_con_ruido = mutar_patron(patron_original, 0.3)

    # Recuperar patrón
    patron_recuperado = red.recover(patron_con_ruido)

    # Mostrar resultados
    red.mostrar_patron(patron_original, "Patrón Original")
    red.mostrar_patron(patron_con_ruido, "Patrón con Ruido")
    red.mostrar_patron(patron_recuperado, "Patrón Recuperado")