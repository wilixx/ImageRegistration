
ImageRegistration
============

This repository is to retain substance for Medical Image Registration.

<p>There are many tools available for doing the `image registration`. For simplicity, I will pay more attention on the ITK related facilities. Since the ITK is notably complicated and require the knowledge of C++ templated programming, there are a lot of efforts to do. The main aim of this period is to get the work done with any flexible means.

ThereT are a few key element we should know about Image registration process.


  * Transformations
  * Metrics
  * Optimazers
  * Interpolaters
  * Sampler

Note that ---An object-oriented framework of `image registration` is as following pseudo codes snippet.

~~~python
import Registration
# since we have difined a class named Registration
reg = Registration()
param={}

reg.setTransform()
reg.setMetric()
reg.setSampler()
reg.setInterpolator()

reg.setFixedImage()
reg.setMovingImage()

reg.exe()

#Done
~~~



