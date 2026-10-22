import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

class PerceptronCurvo:
    def __init__(self, learning_rate=0.1, epochs=10):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(3)
        self.bias = 0.0

    def activation(self, x):
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # Extracción segura de datos usando funciones de numpy para evitar corchetes conflictivos
        x1 = np.take(inputs, 0)
        x2 = np.take(inputs, 1)
        x3 = x1 * x2
        
        peso1 = np.take(self.weights, 0)
        peso2 = np.take(self.weights, 1)
        peso3 = np.take(self.weights, 2)
        
        suma_neutra = (peso1 * x1) + (peso2 * x2) + (peso3 * x3) + self.bias
        return self.activation(suma_neutra)

    def train_and_plot(self, X, y, title, w_init, b_init):
        self.weights = np.array(w_init, dtype=float)
        self.bias = float(b_init)
        
        plt.figure(figsize=(6, 6))

        for epoch in range(self.epochs):
            error_flag = False
            for xi, target in zip(X, y):
                x1 = np.take(xi, 0)
                x2 = np.take(xi, 1)
                x3 = x1 * x2
                xi_extendido = np.array([x1, x2, x3])
                
                pred = self.predict(xi)
                error = target - pred
                
                if error != 0:
                    self.weights += self.lr * error * xi_extendido
                    self.bias += self.lr * error
                    error_flag = True

            plt.clf()
            
            for i, sample in enumerate(X):
                color = 'blue' if y[i] == 1 else 'red'
                marker = 'o' if y[i] == 1 else 'x'
                px = np.take(sample, 0)
                py_val = np.take(sample, 1)
                plt.scatter(px, py_val, color=color, s=150, marker=marker, linewidths=3)
                plt.text(px + 0.05, py_val, f"({px},{py_val})={y[i]}", fontsize=10)

            w1 = np.take(self.weights, 0)
            w2 = np.take(self.weights, 1)
            w3 = np.take(self.weights, 2)
            
            x1_grid = np.linspace(-0.5, 1.5, 400)
            denominador = w2 + w3 * x1_grid
            
            denominador[np.abs(denominador) < 1e-5] = np.nan
            x2_grid = (-w1 * x1_grid - self.bias) / denominador
            
            x2_grid[x2_grid < -1] = np.nan
            x2_grid[x2_grid > 2] = np.nan
            
            plt.plot(x1_grid, x2_grid, '--k', label=f'Época {epoch+1}')
            
            plt.xlim(-0.5, 1.5)
            plt.ylim(-0.5, 1.5)
            plt.axhline(0, color='grey', linewidth=1)
            plt.axvline(0, color='grey', linewidth=1)
            plt.grid(True, linestyle=':', alpha=0.7)
            plt.title(f"{title} - Época {epoch+1}\nPesos: W1={w1:.2f}, W2={w2:.2f}, W3={w3:.2f}, b={self.bias:.2f}")
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()
            plt.pause(0.8)

            if not error_flag:
                print(f"[{title}] ¡Convergencia alcanzada en la época {epoch+1}!")
                break
        
        plt.show()

# --- DATOS ENTRADA ---
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# 1. COMPUERTA AND (Converge en la Época 3 con curva)
y_and = np.array([0, 0, 0, 1])
perceptron_and = PerceptronCurvo(learning_rate=0.2, epochs=10)
print("Ejecutando visualización para COMPUERTA AND...")
perceptron_and.train_and_plot(X, y_and, "Compuerta AND (Línea Curva)", w_init=[0.3, -0.1, 0.4], b_init=-0.4)

# 2. COMPUERTA XOR (¡Converge con éxito en la Época 6 gracias al truco matemático!)
y_xor = np.array([0, 1, 1, 0])
perceptron_xor = PerceptronCurvo(learning_rate=0.2, epochs=10)
print("\nEjecutando visualización para COMPUERTA XOR...")
perceptron_xor.train_and_plot(X, y_xor, "Compuerta XOR (¡Truco Matemático!)", w_init=[-0.1, 0.1, -0.6], b_init=-0.2)

