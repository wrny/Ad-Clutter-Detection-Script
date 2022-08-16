![](https://raw.githubusercontent.com/wrny/Ad-Clutter-Detection-Script/master/sample_outputs/other_examples/Detected_image_1660327704_screenshot.jpg.png?token=GHSAT0AAAAAABXANJW7C3G5YZVNM24E6TLUYX3YUDQ)

# Ad-Clutter-Detection-Script

This tool takes a machine learning approach to identify banner ads on browser screenshots. It was originally built to help identify sites with "ad clutter", meaning more than three ads visible on a page at a time. 

The Script:
* Loads the model into Tensorflow
* Scans it using the appropriate threshold (default is 90% sureity, but can be changed)
* Identifies banners + marks them inside of a rectangle, and writes them out to an output folder.
* The script also writes the time, file, number of ads detected and center point of all of the ads into a text doc.

## How to Use It?
1. Install the script per the instructions below.
2. Load your screenshots into the "dataset" folder via scp or plain old drag-n-drop. There will be a placeholder file called text_doc.txt which can be ignored.
3. Run python3 AdDetector.py

From there, you'll see processed images in the output_folder directory, along with a text doc named 'data_sheet.txt' which is where the results of the model. 

## Does it work?
Yes! Sort of. It has trouble with:

1. Takeovers with Skins and/or a main popup. The lightbox-style pop-up is missed, the skins will usually have one skin detected, but not always both.
2. Really large ad formats (think 970x500). These might get counted as one ad, two ads, or missed entirely.
3. Mobile Interstitals (320x480) hardly ever are detected.
4. Ads without clearly defined borders or very thin borders, common with shady programmatic ads and Google default ads.
5. Non English pages for whatever reason give this tool a lot of trouble.
6. Often when there is a menu that is shaped like a banner ad (happens often!) The program misinterprets those menus as banners.
7. Likewise elements from news or shopping sites that look like 300x250s or 300x600s will ID'ed as ads. Does this mean websites are making their elements look like ads? It happens a lot!

Other than that, it works like a dream. 😅

### Example of working well:
[![1](https://raw.githubusercontent.com/wrny/Ad-Clutter-Detection-Script/master/sample_outputs/good_examples/Detected_image_1660474931_allrecipes.com_2022-08-14-07-58-48_5_.png.png?token=GHSAT0AAAAAABXANJW7PHAH27NOLSKOM4QYYX3YTCQ "2")](https://raw.githubusercontent.com/wrny/Ad-Clutter-Detection-Script/master/sample_outputs/good_examples/Detected_image_1660474931_allrecipes.com_2022-08-14-07-58-48_5_.png.png?token=GHSAT0AAAAAABXANJW7PHAH27NOLSKOM4QYYX3YTCQ "1")

### Example of working badly:
![1](https://raw.githubusercontent.com/wrny/Ad-Clutter-Detection-Script/master/sample_outputs/bad_examples/Detected_image_1660474931_airbnb.com_2022-08-14-00-29-23_1_.png.png?token=GHSAT0AAAAAABXANJW64XHYGKHQ4FTMLT3IYX3YU5Q "1")

### Example of working so-so:
![1](https://raw.githubusercontent.com/wrny/Ad-Clutter-Detection-Script/master/sample_outputs/other_examples/Detected_image_1660328383_screenshot.jpg.png?token=GHSAT0AAAAAABXANJW6HDZBWVF4BGIX5JWGYX3YXYA "1")

### How to improve it:
* Reducing the threshold will reduce the amount of false positives, but it will cause the tool to miss some fairly obvious ads.
* Feeding the model more screenshots with more clearly defined ads.

There was definitely a trade-off between length of solving (this takes a few seconds) vs accuracy. This was designed to look at millions of pages, so time was a factor.

### Requirements :
* Only works with Python 3.7 or earlier. At Python 3.8+, Google removed support for Tensorflow version 1.x and this uses tensorflow==1.13.1. I get around this by downloading Anaconda and creating a 3.7 virtual environment.
* From my testing, it requires at least 2vCPU, 2GB of memory and 8 GB of disk space. 
* Works best on a Windows server, IMHO, but does work on Linux, you follow the below steps + fiddle with the root permissions in order for the script to work.

**Installation Guide for Windows**:

`Download + Install Git for Windows https://git-scm.com/download/win`
`Download + Install Anaconda from here https://www.anaconda.com/` (will take a while, ~10 mins?)
`git clone https://github.com/wrny/Ad-Clutter-Detection-Script`
`cd Ad-Clutter-Detection-Script`
`Open Anaconda Command Prompt from the Win icon in the lower-left corner of the screen`
`conda create -n py37 python=3.7`
`conda activate py37`
`conda install pip`
`python -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.13.1-py3-none-any.whl` (I know it says for Mac. Ignore that, the link still works)
`conda install tensorflow==1.13.1`
`conda install matplotlib`
`pip install opencv-python`

**Installation for Linux (here Ubuntu 20.04 LTS)**:

`sudo apt update`
`sudo apt upgrade`
`sudo git clone https://github.com/wrny/Ad-Clutter-Detection-Script`
`Go to anaconda.org from a normal web browser + copy the downloaded file link`
`wget downloaded file link`
`bash <<DOWNLOADED ANACONDA FILE>>`
`Yes to install Install + reboot once installation is done`
`conda create -n py37 python=3.7`
`conda activate py37`
`conda install pip` 
`python -m pip install --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.13.1-py3-none-any.whl` (I know it says for Mac. Ignore that, the link still works)
`conda install tensorflow==1.13.1`
`conda install matplotlib`
`pip install opencv-python`
`sudo apt install libgl1-mesa-glx`
