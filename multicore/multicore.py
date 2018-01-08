
# coding: utf-8

# # Multi-Scale Gaussian Normalization for Solar Image Processing
# ### Based on Huw Morgan · Miloslav Druckmüller 
# ### Solar Phys (2014) 289:2945–2955
# 
# This code apply a gaussian normalization algorithm to the megamovie data using the technque described in Morgan and Druckmüller (2014).
# 
# The code works as follows:
# 1. Using a CVS with all the metadata from BiQuery several variables are extracted, namely:
#     - detected_circle_radius = element[6]
#     - detected_circle_center_y = element[7]
#     - detected_circle_center_x = element[8]
#     - image_format = element[20]
#     - storage_link = element[11]
#     - image_time = element[15]
# 2. Using the storage link it download the associated file as a temporary file
# 3. Depending on the image format, the code extract the image information and create an array with the image
# 4. Crop the image using the x-y coordinates of the center and the shortest distance to the array limit
# 5. Apply the gaussian normalization algorithm
# 6. Create a figure with the resulting arrays
# 
# The code uses the module *ipyparallel*, using the number of cores specified by the user in *jupyter*
# 
# ### Authors:
# *Juan Carlos Martinez Oliveros*
# 
# *Saida Milena Diaz*
# 
# ### Date
# Nov 30, 2017 *version 1*
# 
# Dec 01, 2017 *version 1, revision for loop position changed, deleting variables to free memory*


#import ipyparallel as ipp
import multiprocessing as mp
import csv
import matplotlib
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
from matplotlib.patches import Circle
from PIL import Image


import astropy.wcs
from astropy.coordinates import EarthLocation
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, Gaussian2DKernel, Tophat2DKernel, Gaussian1DKernel

import rawpy
from IPython.display import clear_output
from IPython import display
import time,os,sys

import datetime

#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'

def MGN_filter(data):
    w=[5,10,20,40,80,120]
    k=3.5 # Binarization incresed
    
    a0=data.min()
    a1=data.max()
    gamma=3.2
    h=1.7
    
    C_g=((data-a0)/(a1-a0))**(1./gamma)

    C_prima=[]
    print('complete 1')
    
    for i in w:
        print(i)
        gaussian_1D_kernel = Gaussian1DKernel(i)

        #First convolve
        data_gauss_convolve1=[]
        for i in range(len(data)):
            data_gauss_convolve1.append(convolve(data[i], gaussian_1D_kernel))
        data_gauss_convolve2=np.transpose(np.array(data_gauss_convolve1))

        data_gauss_convolve3=[]        
        for i in range(len(data_gauss_convolve2)):
            data_gauss_convolve3.append(convolve(data_gauss_convolve2[i], gaussian_1D_kernel))

        data_gauss_convolve_final=np.transpose(np.array(data_gauss_convolve3))

        diff_data_gauss_convolve=data-data_gauss_convolve_final		
        A=diff_data_gauss_convolve**2

        del data_gauss_convolve1,data_gauss_convolve2,data_gauss_convolve3,data_gauss_convolve_final

        ##Second convolve
        data_gauss_convolve1=[]
        for i in range(len(A)):
            data_gauss_convolve1.append(convolve(A[i], gaussian_1D_kernel))
        data_gauss_convolve2=np.transpose(np.array(data_gauss_convolve1))
        data_gauss_convolve3=[]

        for i in range(len(data_gauss_convolve2)):
            data_gauss_convolve3.append(convolve(data_gauss_convolve2[i], gaussian_1D_kernel))
        diff_data_gauss_convolve_final=np.transpose(np.array(data_gauss_convolve3))

        sigma=np.sqrt(diff_data_gauss_convolve_final)

        del data_gauss_convolve1,data_gauss_convolve2,data_gauss_convolve3,data_gauss_convolve_final

        c=diff_data_gauss_convolve/sigma
        c_prima=np.arctan(k*c)
        C_prima.append(c_prima)

    print('complete 2')

    C_prima_mean=np.mean(np.array(C_prima),axis=0)
    final_data=h*C_g+C_prima_mean
    
    raise Exception("FAIL")

    return final_data


#def worker(detected_circle_radius,detected_circle_center_y,detected_circle_center_x,image_time,im):
def worker(numero, element):
    """worker function"""
    tstart=time.time()
    detected_circle_radius = element[6]
    detected_circle_center_y = element[7]
    detected_circle_center_x = element[8]
    image_format = element[20]
    storage_file = element[5]
    storage_link = element[11]
    #image_time = datetime.datetime.strptime(element[15], '%Y-%m-%d %H:%M:%S')
    image_time = datetime.datetime.strptime(element[15], '%Y-%m-%d %H:%M:%S.%f %Z')

    outfile = 'output_test_hugh/mmp_filter_%04d%02d%02d_%02d%02d%02d.jpg'%(image_time.year,image_time.month,
                                                                          image_time.day,image_time.hour,
                                                                          image_time.minute,image_time.second)
    if (not os.path.isfile(outfile)):
        if detected_circle_radius != None:
            print('%04d %s %s %s %s'%(numero, detected_circle_radius, detected_circle_center_y, detected_circle_center_x,image_format), storage_file)

            if image_format == 'JPEG':
                f = '%s.jpg'%storage_file
                urllib.request.urlretrieve(storage_link, f)

                # Read in the image data

                # read in the image and flip it so that it's correct
        #            im_rgb = np.flipud(matplotlib.image.imread(f))

                im_rgb = matplotlib.image.imread(f)
                # remove color info
                im = np.average(im_rgb, axis=2)

                im_rgb = None
                
            elif image_format == 'CR2':
                f = '%s.cr2'%storage_file
                urllib.request.urlretrieve(storage_link, f)

                # Read in the image data

                raw = rawpy.imread(f)
                rgb = raw.postprocess()
                img = Image.fromarray(rgb)
                im = np.array(img.convert('L'))

                raw = None
                rgb = None
                img = None

            elif image_format == 'NEF':
                f = '%s.nef'%storage_file
                urllib.request.urlretrieve(storage_link, f)

                # Read in the image data

                raw = rawpy.imread(f)
                rgb = raw.postprocess()
                img = Image.fromarray(rgb)
                im = np.array(img.convert('L'))

                raw = None
                rgb = None
                img = None

            else:
                img_format = 'UNKNOWN'

            if image_format != 'UNKNOWN':

                im_raw_y=np.shape(im)[0]
                im_raw_x=np.shape(im)[1]

                y_circle_center=int(float(detected_circle_center_y))
                x_circle_center=int(float(detected_circle_center_x))

                a=min([abs(im_raw_y-y_circle_center), y_circle_center,abs(im_raw_x-x_circle_center), x_circle_center])


                y_upp=y_circle_center+a
                y_low=y_circle_center-a
                x_low=x_circle_center-a
                x_upp=x_circle_center+a

                crop_image=im[y_low:y_upp-1,x_low:x_upp]

                im_raw_cy=np.shape(crop_image)[0]/2.
                im_raw_cx=np.shape(crop_image)[1]/2.

                filter_new_image=MGN_filter(crop_image)
                np.flipud(filter_new_image)

                fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(9, 9))
                circ = Circle([int(float(detected_circle_center_x)), 
                               int(float(detected_circle_center_y))], 
                              radius=float(detected_circle_radius), facecolor='none', edgecolor='red', linewidth=2)
                ax[0,0].imshow(im, interpolation='bicubic', origin='lower')#, cmap=plt.cm.Greys)
                ax[0,0].add_patch(circ)
                ax[0,0].set_title("Original Image - Hough transform")

                ax[0,1].imshow(crop_image, interpolation='bicubic', origin='lower')
                ax[0,1].set_title('Cropped Image')

                ax[1,0].imshow(crop_image, interpolation='bicubic', cmap=plt.cm.Greys, origin='lower')
                ax[1,0].set_title('Cropped Image BW')

                ax[1,1].imshow(filter_new_image, interpolation='bicubic', cmap=plt.cm.Greys, origin='lower')
                ax[1,1].set_title('Filtered Image')

                plt.suptitle('%s'%image_time)

                plt.savefig(outfile, format='jpg', dpi=300)
                plt.close() 

                try:
                    os.remove(f)   

                except:
                    print('An error occured.')

            else:
                print(image_format)                                
        else:
            tmp=None
    else:
        if os.path.isfile(outfile):
            print('Image already processed')
            print('%04d %s %s %s %s'%(numero, detected_circle_radius, detected_circle_center_y, detected_circle_center_x,image_format), storage_file)
        else:
            print('No Hough filter solution (Google)')
            print('%04d %s %s %s %s'%(numero, detected_circle_radius, detected_circle_center_y, detected_circle_center_x,image_format), storage_file)

    tend=time.time()
    t1=tend-tstart
    print('Processing Image= %0.5f sec'%(t1))
    return None

def maprec(obj, fun):
    if isinstance(obj, list):
        return [maprec(x, fun) for x in obj]
    return fun(obj)

################################################################################
if __name__ == '__main__':
    tstart_2=time.time()
    if not os.path.isdir('output_test_hugh'):
        os.makedirs('output_test_hugh')

    if len(sys.argv) > 1:
        ncpus = int(sys.argv[1])
    else:
        ncpus = mp.cpu_count()

    #Opening the list from BigQuery
#    with open('results_20171121_115549.csv', 'r') as f:  # <<JC Test
    
    with open('results-20171130-102459.csv', 'r') as f:
        reader = csv.reader(f)
        your_list = list(reader)
    f.close()
    
    #Defining a new list that can be used in a interation loop
    new_list = maprec(your_list, lambda x: None if x is '' else x)        
    
    # Running Gaussian Filtering code
    print("Starting multiprocessing v2 with %d workers"%(ncpus))

    pool = mp.Pool(processes=ncpus)

    try:
        jobs = [pool.apply_async(worker, args=(numero, element)) for numero,element in enumerate(new_list)]
        results = [r.get() for r in jobs]    # This line actually runs the jobs
        pool.close()
        pool.join()

    # re-raise the rest
    except Exception:
        print("Exception in worker:")
        traceback.print_exc()
        raise

    tend_2=time.time()
    t2=tend_2-tstart_2
    print('Multiprocessing = %0.5f sec'%(t2))

    im = None

