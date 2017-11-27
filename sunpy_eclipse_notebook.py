
# coding: utf-8

# # Processing your Eclipse Photo with SunPy

# Requirements
# * Python 3
# * matplotlib
# * scipy
# * sunpy >= 0.8
# * skimage
# * exifread
# * astropy

# In[ ]:


from datetime import datetime

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

import scipy.ndimage as ndimage
from skimage.transform import hough_circle, hough_circle_peaks

import astropy.wcs
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u

import sunpy
import sunpy.map
import sunpy.coordinates

import exifread # to read information from the image
import rawpy

#get_ipython().magic('matplotlib inline')
#get_ipython().magic("config InlineBackend.figure_format = 'svg'")


# In[ ]:


sunpy.system_info()


# Replace the following with your photo filename. Here we are using an eclipse photo originally taken by Henke Bril taken in Salloum, Egypt on March 29, 2006. We've hacked the EXIF info to make it seem like it was taken for this eclipse. We will update this with a real 2017 Eclipse photo as soon as we have one!

# In[ ]:


#f = '../sample-photos/total_solar_eclipse2017.jpg'
f = '../Camera_raw/DSC_0076.NEF'


# ## First let's try to get some metadata from the file

# In[ ]:


tags = exifread.process_file(open(f, 'rb'))


# In[ ]:


# the following functions will help us get GPS data from the EXIF data if it exists
def _convert_to_degress(value):
    """
    Helper function to convert the GPS coordinates stored in the EXIF to degress in float format
    :param value:
    :type value: exifread.utils.Ratio
    :rtype: float
    """
    d = float(value.values[0].num) / float(value.values[0].den)
    m = float(value.values[1].num) / float(value.values[1].den)
    s = float(value.values[2].num) / float(value.values[2].den)

    return d + (m / 60.0) + (s / 3600.0)
    
def get_exif_location(exif_data):
    """
    Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)
    """
    lat = None
    lon = None

    gps_latitude = exif_data.get('GPS GPSLatitude', None)
    gps_latitude_ref = exif_data.get('GPS GPSLatitudeRef', None)
    gps_longitude = exif_data.get('GPS GPSLongitude', None)
    gps_longitude_ref = exif_data.get('GPS GPSLongitudeRef', None)

    if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
        lat = _convert_to_degress(gps_latitude)
        if gps_latitude_ref.values[0] != 'N':
            lat = 0 - lat

        lon = _convert_to_degress(gps_longitude)
        if gps_longitude_ref.values[0] != 'E':
            lon = 0 - lon

    return lat, lon


# In[ ]:


if "EXIF ExposureTime" in tags:
    exposure_tag = tags['EXIF ExposureTime']
    exposure_time = exposure_tag.values[0].num / exposure_tag.values[0].den * u.s
    print("Exposure time of {0} found!".format(exposure_time))
if "Image Artist" in tags:
    author_str = tags['Image Artist'].values
    print("Author name of {0} found!".format(author_str))
else:
	author_str = 'JC Martinez Oliveros' 
	print("Author name of {0} found!".format(author_str))
if "EXIF DateTimeOriginal" in tags:
    datetime_str = tags['EXIF DateTimeOriginal'].values.replace(' ', ':').split(':')
    time = datetime(int(datetime_str[0]), int(datetime_str[1]), 
                    int(datetime_str[2]), int(datetime_str[3]),
                    int(datetime_str[4]), int(datetime_str[5]))
    print("Image time of {0} found!".format(str(time)))
if "Image Model" in tags:
    camera_model_str = tags['Image Model'].values
    print("Camera model of {0} found!".format(camera_model_str))
    
lat, lon = get_exif_location(tags)

lat = 44.892984
lon = -123.020692

if ((lat != None) and (lon != None)):
    gps = [lat, lon] * u.deg
    print("Found GPS location of {0}, {1} found!".format(gps[0], gps[1]))


# The following variables need to be defined. If they were not found in the EXIF data please define them below

# In[ ]:


# exposure_time = 0.1 * u.s
# camera_model_str = 'Canon 70D'
# author_str = 'Julius Berkowski'
time = datetime(2017, 8, 21, 17, 17, 51) # don't forget to convert your time to UTC!
#gps = [44.37197, -116.87393] * u.deg # latitude, longitude of Mann Creek, Idaho


# ## Read in the image data

# In[ ]:


# read in the image and flip it so that it's correct
#im_rgb = np.flipud(matplotlib.image.imread(f))
raw = rawpy.imread(f)
im_rgb = np.flipud(raw.postprocess())
# remove color info
im = np.average(im_rgb, axis=2)


# In[ ]:


plt.imshow(im, origin='lower')


# # Get info from the image

# We need the following information from the image
# * the location of the center of the Sun/Moon and 
# * the scale of the picture which we get from the size of the Sun/Moon in pixels

# In[ ]:


blur_im = ndimage.gaussian_filter(im, 8)
mask = blur_im > blur_im.mean() * 3
plt.imshow(mask)


# In[ ]:


label_im, nb_labels = ndimage.label(mask)
plt.imshow(label_im)


# In[ ]:


slice_x, slice_y = ndimage.find_objects(label_im==1)[0]
roi = blur_im[slice_x, slice_y]
plt.imshow(roi)


# In[ ]:


sx = ndimage.sobel(roi, axis=0, mode='constant')
sy = ndimage.sobel(roi, axis=1, mode='constant')
sob = np.hypot(sx, sy)
plt.imshow(sob > (sob.mean() * 5))


# In[ ]:


from skimage.transform import hough_circle, hough_circle_peaks

hough_radii = np.arange(np.floor(np.mean(sob.shape)/4), np.ceil(np.mean(sob.shape)/2), 10)
hough_res = hough_circle(sob > (sob.mean() * 5), hough_radii)

# Select the most prominent circle
accums, cy, cx, radii = hough_circle_peaks(hough_res, hough_radii, total_num_peaks=1)
print(cx, cy, radii)


# In[ ]:


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
circ = Circle([cy, cx], radius=radii, facecolor='none', edgecolor='red', linewidth=2)
ax.imshow(sob)
ax.add_patch(circ)
plt.show()


# In[ ]:


fig, ax = plt.subplots(ncols=3, nrows=1, figsize=(9.5, 6))
ax[0].imshow(im[slice_x, slice_y])
ax[0].set_title('Original')
ax[1].imshow(sob > (sob.mean() * 5))
ax[1].set_title('Derivative')
circ = Circle([cy, cx], radius=radii, facecolor='none', edgecolor='red', linewidth=2, label='Hough fit')
ax[2].imshow(im[slice_x, slice_y])
ax[2].add_patch(circ)
ax[2].set_title('Original with fit')
plt.legend()


# Now let's check it with the original image

# In[ ]:


fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(10, 4))
fudge_shift_x = 0 * u.pix # update this in case the fit needs to be shifted in x
fudget_shift_y = 0 * u.pix # update this in case the fit needs to be shifted in y
im_cx = (cx + slice_x.start) * u.pix + fudge_shift_x
im_cy = (cy + slice_y.start) * u.pix + fudget_shift_y
im_radius = radii * u.pix
circ = Circle([im_cy.value, im_cx.value], radius=im_radius.value, facecolor='none', edgecolor='red', linewidth=2)
ax.imshow(im)
ax.add_patch(circ)
plt.show()


# # Getting information about the Sun

# Let's now get the distance between the Earth and the Sun at the time the photo was taken

# In[ ]:


dsun = sunpy.coordinates.get_sunearth_distance(time.isoformat())
print(dsun)


# The size of the Sun in the sky is then

# In[ ]:


rsun_obs = np.arctan(sunpy.sun.constants.radius / dsun).to('arcsec')
print(rsun_obs)


# The image plate scale is then

# In[ ]:


plate_scale = rsun_obs / im_radius
print(plate_scale)


# We also need the solar rotation angle

# In[ ]:


loc = EarthLocation(lat=gps[0], lon=gps[1])
fudge_angle = 0.0 * u.deg # update this in case your camera was not perfectly level.
solar_rotation_angle = sunpy.coordinates.get_sun_orientation(loc, time) + fudge_angle


# In[ ]:


hgln_obs = 0 * u.deg # sunpy.coordinates.get_sun_L0(time)
hglt_obs = sunpy.coordinates.get_sun_B0(time)
print("{0} {1}".format(hglt_obs, hgln_obs))


# # Define your WCS object and header

# In[ ]:


w = astropy.wcs.WCS(naxis=2)
w.wcs.crpix = [im_cy[0].value, im_cx[0].value]
w.wcs.cdelt = np.ones(2) * plate_scale.to('arcsec/pix').value
w.wcs.crval = [0, 0]
w.wcs.ctype = ['TAN', 'TAN']
w.wcs.cunit = ['arcsec', 'arcsec']
w.wcs.dateobs = time.isoformat()


# Using this object we can now create the sunpy map header

# In[ ]:


header = dict(w.to_header())
header.update({'CROTA2': solar_rotation_angle.to('deg').value})
header.update({'DSUN_OBS': dsun.to('m').value})
header.update({'HGLN_OBS': hgln_obs.to('deg').value})
header.update({'HGLT_OBS': hglt_obs.to('deg').value})
header.update({'CTYPE1': 'HPLN-TAN'})
header.update({'CTYPE2': 'HPLT-TAN'})
header.update({'RSUN': dsun.to('m').value})
header.update({'RSUN_OBS': np.arctan(sunpy.sun.constants.radius / dsun).to('arcsec').value})


# Now add metadata about the photograph.

# In[ ]:


header.update({'AUTHOR': author_str})
header.update({'EXPTIME': exposure_time.to('s').value})
header.update({'TELESCOP': camera_model_str})
header.update({'INSTRUME': camera_model_str})
header.update({'DETECTOR': camera_model_str})


# In[ ]:


header


# # Creating SunPy Map

# In[ ]:


m = sunpy.map.Map((im, header))
m


# In[ ]:


fig = plt.figure(figsize=(10,10))
ax = plt.subplot(projection=m)
m.plot(axes=ax)
m.draw_grid(axes=ax)
m.draw_limb(axes=ax)
plt.show()


# # Overplot the location of Regulus

# In[ ]:


regulus = SkyCoord(ra='10h08m22.311s', dec='11d58m01.95s', distance=79.3 * u.lightyear, frame='icrs').transform_to(m.coordinate_frame)


# In[ ]:


regulus


# In[ ]:


fig = plt.figure(figsize=(9,9))
ax = plt.subplot(projection=m)
m.plot(axes=ax)
ax.plot_coord(regulus, '*w', label='Regulus')
m.draw_grid(axes=ax)
m.draw_limb(axes=ax)
plt.legend()
plt.show()


# We can see that the predicted location of regulus does not match which tells us that there a slight error in the angle. This is likely because the camera was not level with the horizon. Therefore we adjust to get it right.

# In[ ]:


fudge_angle = 2.55 * u.deg # update this in case your camera was not perfectly level.
solar_rotation_angle = sunpy.coordinates.get_sun_orientation(loc, time) + fudge_angle
header.update({'CROTA2': solar_rotation_angle.to('deg').value})
m = sunpy.map.Map((im, header))


# In[ ]:


fig = plt.figure(figsize=(9,9))
ax = plt.subplot(projection=m)
m.plot(axes=ax)
ax.plot_coord(regulus, 'o', markeredgewidth=0.5, markeredgecolor='w', 
              markerfacecolor='None', label='Regulus')
m.draw_grid(axes=ax)
m.draw_limb(axes=ax)
plt.legend()
plt.show()


# More tweaking could be done here to get thing right. We will leave that as an exercise for the reader!

# # Plot an SDO AIA Image of the Sun on your photo

# First we need to download the images

# In[ ]:


from sunpy.net import Fido, attrs as a
# Replace the time below with the time in UT of the eclipse
t = a.Time('2017-08-21 17:27:13', "2017-08-21 17:45:13")
aia_result = Fido.search(t, a.Instrument('AIA'), a.Wavelength(171*u.Angstrom))
aia_result


# In[ ]:


files = Fido.fetch(aia_result[0,0])


# In[ ]:


files


# In[ ]:


aia_map = sunpy.map.Map(files[0])
aia_map.plot()


# To overplot the images we must align them, this can be done with `rotate`.

# In[ ]:


am2 = aia_map.rotate(rmatrix=np.linalg.inv(m.rotation_matrix),
                     recenter=True, order=3, scale=(aia_map.scale[0]/m.scale[0]))


# We then must calculate the extent of the AIA image in terms of pixels in the eclipse image.

# In[ ]:


xmin, ymin = (u.Quantity(m.reference_pixel) - u.Quantity(am2.reference_pixel)).to_value(u.pix)


# In[ ]:


xmax = am2.data.shape[1] + xmin
ymax = am2.data.shape[0] + ymin


# In[ ]:


extent = (xmin, xmax, ymin, ymax)


# Finally we mask out the pixels with a low value in the eclipse image (to make the disk transparent).

# In[ ]:


m.data[m.data < 30] = np.nan


# In[ ]:


fig = plt.figure(figsize=(10,15))
ax = plt.subplot(projection=m)

# Set the axes background to black.
ax.set_facecolor('k')

# Plot the AIA image.
am2.plot(extent=extent)
# Plot the eclipse image
m.plot()

# Draw heliographic and helioprojective grids
m.draw_grid()
ax.coords.grid(color='white', alpha=1, linestyle='dotted',linewidth=2)

plt.show()
# In[ ]:




