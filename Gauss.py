import numpy as np
from PIL import Image
from numba import cuda, float32
import time
from google.colab import drive
import matplotlib.pyplot as plt

drive.mount('/content/drive')
image_source="/content/drive/MyDrive/painting.jpg"
Image.MAX_IMAGE_PIXELS = None
image=Image.open(image_source)
image_rgb=np.array(image)
input_array=np.array(image_rgb)

#:atrice predefinie du tp
matrice_tp=np.array([
    [1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]
],dtype=np.uint8)
kernel_tp=matrice_tp/matrice_tp.sum()
#kernel
@cuda.jit
def my_kernel(input,output,kernel,kernel_size):
    x,y=cuda.grid(2)
    longueur,largeur= input.shape
    demi_largeur_noeud= kernel_size//2
    if x<longueur and y<largeur:
        acc=0
        for i in range(kernel_size):
            for j in range(kernel_size):
                xx = min(max(x+i-demi_largeur_noeud,0),longueur- 1)
                yy = min(max(y+j-demi_largeur_noeud,0),largeur - 1)
                acc= acc+input[xx,yy] * kernel[i,j]
        output[x,y]=acc


def gaussian_blur(input_array,gaussian_kernel):
    longueur,largeur=input_array.shape[:2]
    kernel_size=gaussian_kernel.shape[0]
    d_kernel=cuda.to_device(gaussian_kernel)
    channels=[input_array[:,:,i].astype(np.uint8)for i in range(3)]
    output=[np.zeros_like(channel)for channel in channels]
    TB=(8,8)
    grid_x=(longueur+TB[0]-1)//TB[0]
    grid_y=(largeur+TB[1]-1)//TB[1]
    blockspergrid=(grid_x,grid_y)
    for i,channel in enumerate(channels):
        d_input=cuda.to_device(channel)
        d_output=cuda.to_device(output[i])
        my_kernel[blockspergrid,TB](d_input,d_output,d_kernel,kernel_size)
        output[i]=d_output.copy_to_host()
    return np.dstack((output[0],output[1],output[2])).astype(np.uint8)
#floute l image en utilisant le kernel predefinie dur le TP
def floute(image_source,image_destination):
    image=Image.open(image_source)
    input_array=np.array(image)
    blurred_image=gaussian_blur(input_array,kernel_tp)
    blurred_image_pil=Image.fromarray(blurred_image)
    blurred_image_pil.save(image_destination)
#floute l image en utilisant la formule Guaussienne
def FlouteGaussienCustom(image_source,image_destination,largeur,sigma):
    image=Image.open(image_source)
    input_array=np.array(image)
    demi_largeur_noeud=largeur//2
    coords=np.arange(-demi_largeur_noeud,demi_largeur_noeud+1,dtype=np.float32)
    x,y=np.meshgrid(coords,coords)
    kernel_matrix=np.exp(-(x**2+y**2)/(2*sigma**2))/(2*np.pi*sigma**2)
    gaussian_kernel_custom=kernel_matrix/kernel_matrix.sum()
    blurred_image=gaussian_blur(input_array,gaussian_kernel_custom)
    blurred_image_pil=Image.fromarray(blurred_image)
    blurred_image_pil.save(image_destination)
#test de performance avec affichage de graphique
def Bench(image_source):
    image = Image.open(image_source)
    input_array = np.array(image)
    block_sizes = [(8,8),(16,16),(32,32)]
    kernel_sizes = [5,7,9]  
    results_thread_blocks = []
    results_kernel_sizes = []

    for threadsperblock in block_sizes:
        longueur,largeur = input_array.shape[:2]
        times=[]
        blockspergrid_x=(longueur + threadsperblock[0]-1)//threadsperblock[0]
        blockspergrid_y=(largeur + threadsperblock[1]-1)//threadsperblock[1]
        blockspergrid=(blockspergrid_x,blockspergrid_y)

        for _ in range(11):
            start_time = time.time()
            blurred_image = gaussian_blur(input_array, kernel_tp)
            cuda.synchronize()
            times.append(time.time() - start_time)

        avg_time = np.mean(times[1:])  
        results_thread_blocks.append(avg_time)
        print(f"Taille du bloc {threadsperblock}: temps moy {avg_time:.5f} s")

    for k in kernel_sizes:
        image_temp_path= f"/content/drive/MyDrive/temp_blurred_{k}x{k}.jpg"
        start_time=time.time()
        FlouteGaussienCustom(image_source, image_temp_path,k,2)
        avg_time=time.time()-start_time
        results_kernel_sizes.append(avg_time)
        print(f"Taille du noyau {k}x{k}: temps moy {avg_time:.5f} s")

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    tailles_bloc = ['8x8','16x16','32x32']
    plt.plot(tailles_bloc, results_thread_blocks, marker='o')
    plt.xlabel('Taille du bloc')
    plt.ylabel('Temps moyen (secondes)')
    plt.title('Performance en fonction de la taille du bloc de thread')
    plt.subplot(1,2,2)
    tailles_noyau = [f"{k}x{k}" for k in kernel_sizes]
    plt.plot(tailles_noyau, results_kernel_sizes, marker='o')
    plt.xlabel('Taille du noyau gaussien')
    plt.ylabel('Temps moyen (secondes)')
    plt.title('Performance en fonction de la taille du kernel gaussien')
    plt.tight_layout()
    plt.show()



image_destination="/content/drive/MyDrive/blurred_painting.jpg"
custom_image_destination="/content/drive/MyDrive/custom_blurred_painting.jpg"
floute(image_source,image_destination)
FlouteGaussienCustom(image_source,custom_image_destination,7,2)
Bench(image_source)