import numpy as np
import matplotlib.pyplot as plt
import math


mi = 1/16 #define o valor de mi
N = [100, 200, 400, 800, 1600] #define os tamanhos de subintervalos
h = np.zeros(len(N))

E1 = np.zeros(len(N))
E2 = np.zeros(len(N))
E3 = np.zeros(len(N))

slopes_E1 = []
slopes_E2 = []
slopes_E3 = []


for i in range(5):
    h[i] = (3 * np.pi) / (N[i]) #define o tamanho do subintervalo para cada N 

for i in range(5):
    erro = np.zeros(N[i])
    F = np.zeros(N[i]+1)
    x = np.linspace(0, 3 * np.pi, N[i]+1)
    u = np.cos(x)#chute inicial

    def funcao_discretizada(u):
        u[0] = 1
        x[0] = 0
        
        #definindo os valores de x
        for j in range(1, N[i]+1):
            x[j] = x[j-1] + h[i]
        #funcao discretizada que é usada para i entre 1 e N-1
        for j in range(1, N[i]):
            F[j] = (u[j+1] - 2*u[j] + u[j-1]) / (h[i]**2) + mi * (u[j]**2 - 1) * ((u[j+1] - u[j-1]) / (2 * h[i])) + u[j] - mi * np.sin(x[j])**3
        #funcao para i igual a 0
        F[0] =  u[0] - 1
        #funcao para i igual a N
        F[N[i]] = (-2*u[N[i]] + 2*u[N[i]-1] - 2*h[i]*(u[N[i]]+1)) / (h[i]**2) - mi * ((u[N[i]]**2) - 1) * (u[N[i]]+1) + u[N[i]] - mi * np.sin(x[N[i]])**3
        
        return F

    def jacobiana(u):
        J = np.zeros((N[i]+1, N[i]+1))
        #Definindo os valores da jacobiana para as possicoes do meio
        #e da ultima linha tirando as posicoes [N,N-1] e [N,N]. Para isso fizemos
        #as derivadas em relacao a i-1, i e i+1 da funcao discretizada
        for j in range(1, N[i]):
            J[j, j-1] = 1 / (h[i]**2) - mi * (u[j]**2 - 1) / (2 * h[i])
            J[j, j] = -2 / (h[i]**2) + mi * (u[j]) * ((u[j+1] - u[j-1]) / (h[i])) + 1
            J[j, j+1] = 1 / (h[i]**2) + mi * (u[j]**2 - 1) / (2 * h[i])
        #derivada da F[0]
        J[0, 0] = 1
        #Derivada de F[N] em relacao a i-1 e a i
        J[N[i], N[i]] = (-2-2*h[i]) / h[i]**2 - mi * (3*u[N[i]]**2 + 2*u[N[i]] - 1) + 1
        J[N[i], N[i]-1] = 2 / h[i]**2
        
        return J

    def newton_method(F, J, u0, tol=1e-10, max_iter=100):
        x = u0
        cont = 0
        
        for iter in range(max_iter):
            # Calcular o vetor das equações não-lineares
            f =  funcao_discretizada(x)
            
            # Calcular a matriz Jacobiana
            Jx = jacobiana(x)
            
            # Resolver o sistema linear J(x) * delta = -F
            delta = np.linalg.solve(Jx, -f)
            
            # Atualizar a solução
            x = x + delta
            
            # Verificar o critério de parada
            if np.linalg.norm(delta) < tol:
                break
            cont += 1
            if cont > 100:
                print("nao convergiu")
                break
        
        return x, iter + 1
    
    #define o vetor solucao para cada valor de N
    solucao, num_interacoes = newton_method(funcao_discretizada(u) ,jacobiana(u), u, tol=1e-6, max_iter=100)
    
    #Encontra as normas 1, 2 e infinita dos erros 
    E1[i] = np.linalg.norm(np.cos(x) - solucao, ord = 1)/np.linalg.norm(np.cos(x), ord=1)
    E2[i] = np.linalg.norm(np.cos(x) - solucao)/np.linalg.norm(np.cos(x))
    E3[i] = np.linalg.norm(np.cos(x) - solucao, np.inf)

#Calcula a inclinacao da reta dos erros entre dois h's
for i in range(len(h) - 1):
    slope = (np.log(E1[i+1]) - np.log(E1[i])) / (np.log(h[i+1]) - np.log(h[i]))
    slopes_E1.append(slope)


# Printa a inclinacao
print("inclinacao para cada par de pontos para E1:")
print("-------------------------------")
print("|    h1    |    h2    |  Slope |")
print("-------------------------------")
for i in range(len(slopes_E1)):
    print(f"|  {h[i]:.4f}  |  {h[i+1]:.4f}  |  {slopes_E1[i]:.4f}  |")
print("-------------------------------")

for i in range(len(h) - 1):
    slope = (np.log(E1[i+1]) - np.log(E1[i])) / (np.log(h[i+1]) - np.log(h[i]))
    slopes_E2.append(slope)


# Print the table
print("inclinacao para cada par de pontos para E2:")
print("-------------------------------")
print("|    h1    |    h2    |  Slope |")
print("-------------------------------")
for i in range(len(slopes_E2)):
    print(f"|  {h[i]:.4f}  |  {h[i+1]:.4f}  |  {slopes_E2[i]:.4f}  |")
print("-------------------------------")

for i in range(len(h) - 1):
    slope = (np.log(E1[i+1]) - np.log(E1[i])) / (np.log(h[i+1]) - np.log(h[i]))
    slopes_E3.append(slope)


# Print the table
print("inclinacao para cada par de pontos para E3:")
print("-------------------------------")
print("|    h1    |    h2    |  Slope |")
print("-------------------------------")
for i in range(len(slopes_E3)):
    print(f"|  {h[i]:.4f}  |  {h[i+1]:.4f}  |  {slopes_E3[i]:.4f}  |")
print("-------------------------------")

#Plota as retas das normas
plt.loglog(h, E1, marker='o', linestyle='-', color='red', label='Norma 1')
plt.loglog(h, E2, marker='o', linestyle='-', color='blue', label='Norma 2')
plt.loglog(h, E3, marker='o', linestyle='-', color='orange', label='Norma Infinita')
plt.title('Erro em função do espaçamento (Escala Logarítmica)')
plt.xlabel('h')
plt.ylabel('Erro')
plt.legend()  # Adiciona a legenda
plt.grid(True)
plt.show()
