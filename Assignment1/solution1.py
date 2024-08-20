import numpy as np
import matplotlib.pyplot as plt

mi_valores = [0, 1/16, 1/8, 1/4, 1/2, 1, 6/5] #definindo os valores para mi
N = 100 #definindo o número de subintervalos
h = (3*np.pi) / N #definindo o tamanho do subintervalo
x = np.linspace(0, 3*np.pi, N+1) #defindo o intervalo de x
u0 = np.cos(x) #definindo o chute inicial

def funcao_discretizada(u):
    F = np.zeros(N+1)
    #funcao discretizada que é usada para i entre 1 e N-1
    for i in range(1, N):
        F[i] = (u[i+1] - 2*u[i] + u[i-1]) / (h**2) + mi * (u[i]**2 - 1) * ((u[i+1] - u[i-1])/(2*h)) + u[i]
    #funcao para i igual a 0
    F[0] = u[0] - 1
    #funcao para i igual a N
    F[N] = (-2*u[N] + 2*u[N-1] - 2*h*(u[N]+1)) / (h**2) - mi * ((u[N]**2) - 1)*(u[N] + 1) + u[N]
    
    return F

def jacobiana(u):
    J = np.zeros((N+1,N+1))
    #Definindo os valores da jacobiana para as possicoes do meio
    #e da ultima linha tirando as posicoes [N,N-1] e [N,N]. Para isso fizemos
    #as derivadas em relacao a i-1, i e i+1 da funcao discretizada
    for i in range(1, N):
        J[i, i-1] = 1 / (h**2) - mi * (u[i]**2 - 1) / (2 * h)
        J[i, i] = -2 / (h**2) + mi * (u[i])*((u[i+1] - u[i-1])/(h)) + 1
        J[i, i+1] = 1 / (h**2) + mi * (u[i]**2 - 1) / (2 * h)
    #derivada da F[0]
    J[0,0] = 1
    #Derivada de F[N] em relacao a i-1 e a i
    J[N,N] = (-2-2*h)/(h**2) - mi * (3*u[N]**2 + 2*u[N] - 1) + 1
    J[N,N-1] = 2/(h**2)
    
    return J

#Calcula o método de newton utilizando ass funcoes criadas acima
def newton_method(F, J, u0, mi, tol=1e-6, max_iter=500):
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
            if cont >= 1000:
                print("nao convergiu")
                break
    
    #printa o numero de iteracoes necessárias para a convergencia   
    print("Convergiu em ", cont,"interacoes")
    
    return x

#Plota as solucoes para cada valor de mi
plt.figure(figsize=(10, 8))
mi_valores2=[0, 1/16, 1/8, 1/4, 1/2]
for mi in mi_valores2:
    #Calcula a solucao da funcao para cada mi em mi_valores2
    solucao = newton_method(funcao_discretizada, jacobiana, u0, mi)
    plt.plot(x, solucao, label=f'mu = {mi}')

plt.title('Solução da equação de Van der Pol para diferentes valores de mu')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True)
plt.show()

#Plota as solucoes para cada valor de mi
plt.figure(figsize=(10, 8))
mi_valores3=[1, 6/5]
for mi in mi_valores3:
    #Calcula a solucao da funcao para cada mi em mi_valores2
    solucao = newton_method(funcao_discretizada, jacobiana, u0, mi)
    plt.plot(x, solucao, label=f'mu = {mi}')

plt.title('Solução da equação de Van der Pol para diferentes valores de mu')
plt.xlabel('x')
plt.ylabel('u(x)')
plt.legend()
plt.grid(True)
plt.show()
