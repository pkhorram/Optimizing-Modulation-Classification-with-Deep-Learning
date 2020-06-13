import tarfile
tar = tarfile.open("./RML2016.10a.tar.bz2", "r:bz2")  
tar.extractall()
# Extract this "RML2016.10b.tar.bz2" if you want to run on bigger dataset

