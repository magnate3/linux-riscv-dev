# This portion is part of my test code
from PIL import Image
import io
byteImgIO = io.BytesIO()
byteImg = Image.open("../data/img1.jpg")
byteImg.save(byteImgIO, "PNG")
byteImgIO.seek(0) #add 
byteImg = byteImgIO.read()


# Non test code
dataBytesIO = io.BytesIO(byteImg)
Image.open(dataBytesIO)
