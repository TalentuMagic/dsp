import simple_image_download.simple_image_download as sid

images = sid.Downloader()

keywords = ['people', 'dogs', 'cats', 'landscape', 'flowers']

for keyword in keywords:
    images.download(keyword, 10)
