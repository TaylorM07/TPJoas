# importação das bibliotecas necessárias

import os

# pybrain
from pybrain.datasets.supervised import SupervisedDataSet 
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer

# gráficos 
import matplotlib.pyplot as plt
import numpy as np

# tratamento de imagem
from PIL import Image

# obter os nomes dos arquivos dentro de uma pasta
def getFiles( path ):
    files = []
    _files = []
    for past, subpath, file in os.walk( path ):
        _files = file
    for file in _files:
        if file.endswith(".jpg"):
            files.append( f'{path}\\{file}' )
    return files

# redimensionar imagem para 40x40
def resize( path ):
    img = Image.open( path )
    img = img.resize( (60,60) )
    return img

# função para converter imagem em matriz
def getImage( path ):
    img = resize( path )
    pixels = img.load()
    largura = img.size[0]
    altura = img.size[1]
    data = []
    pixel = []
    for i in range( largura ):
        for j in range( altura ):
            pixel = pixels[i,j]
            data.append( pixel[0] )
            data.append( pixel[1] )
            data.append( pixel[2] )
    # exif_data = img._getexif()
    # exif_data
    return data

# função para obter o size da imagem rgb
def getSize( path ):
    img = resize( path)
    largura = img.size[0]
    altura = img.size[1]
    return largura * altura * 3


# função para carregar os dados de treinamento
def getData( path ):
    #Open file
    file = open( path, "r" )
    
    data = []    
    
    for linha in file:        # obtem cada linha do arquivo
      linha = linha.rstrip()  # remove caracteres de controle, \n
      digitos = linha.split(" ")  # pega os dígitos
      for numero in digitos:   # para cada número da linha
        data.append( numero )  # add ao vetor de dados  
    
    file.close()
    return data


tamanho = getSize( 'treino\\lubri1.jpg' )

# num de neuronio de acordo com o tamanho da rede
neuronios = int(tamanho/8)
print('neuronios: {}'.format(neuronios))

# configurando a rede neural artificial e o dataSet de treinamento
network = buildNetwork( tamanho, neuronios,  neuronios, 1 )    # define network 
dataSet = SupervisedDataSet( tamanho, 1 )  # define dataSet

'''
Exemplo da lista de arquivos de treinamento:
arquivos = ['1.txt', '1a.txt', '1b.txt', '1c.txt',
            '1d.txt', '1e.txt', '1f.txt']
'''  
arquivos = getFiles( 'treino' )

# Exemplo de resposta 
# resposta = [ [1], [1],[0], [0] ] 
resposta = [] 

for arquivo in arquivos:

    #verificar nome do arquivo removendo o prefixo
    nome = arquivo.split('\\')[1]
    
    if(nome.startswith('n')):
        resposta.append([0])
    else:
        resposta.append([1])


i = 0
for arquivo in arquivos:           # para cada arquivo de treinamento
    data =  getImage( arquivo )  # pegue os dados do arquivo
    dataSet.addSample( data, resposta[i] )  # add dados no dataSet
    i = i + 1


# trainer
trainer = BackpropTrainer( network, dataSet )
error = 1
iteration = 0
outputs = []
file = open("resultimg.txt", "w") # arquivo para guardar os resultados

while error > 0.00001: # 10 ^ -3
    error = trainer.train()
    outputs.append( error )
    iteration += 1    
    print ( iteration, error )
    file.write( str(error)+"\n" )

file.close()

# Fase de teste
# arquivos = ['teste\\n_lub_1.jpg','teste\\n_lub_2.jpg', 'teste\\lub_1.jpg']
arquivos = ['testes\\naolub1.jpg','testes\\naolub2.jpg', 'testes\\naolub3']
for arquivo in arquivos:
    data =  getImage( arquivo )
    print ( network.activate( data ) )


# plot graph
plt.ioff()
plt.plot( outputs )
plt.xlabel('Iterações')
plt.ylabel('Erro Quadrático')
plt.show()