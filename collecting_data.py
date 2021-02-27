import os
import pytesseract
from PIL import Image
from torchvision import transforms

img = Image.open(r'E:\botay\диплом\train_sample\npa\0002.jpg')

def find_lightest_rows(img, threshold):
    line_luminances = [0] * img.height

    for y in range(img.height):
        for x in range(img.width):
            line_luminances[y] += img.getpixel((x, y))

    line_luminances = [x for x in enumerate(line_luminances)]
    line_luminances.sort(key=lambda x: -x[1])
    lightest_row_luminance = line_luminances[0][1]
    lightest_rows = []
    for row, lum in line_luminances:
        if(lum > lightest_row_luminance * threshold):
            lightest_rows.append(row)
    lightest_rows.sort()
    return lightest_rows

def colour_lightest_rows(img):
    rows = find_lightest_rows(img, 0.99)
    for row in rows:
        for x in range(img.width):
            img.load()[x,row] = 0
    img.show()

def find_lightest_columns(img, threshold):
    line_luminances = [0] * img.width

    for x in range(img.width):
        for y in range(img.height):
            line_luminances[x] += img.getpixel((x, y))

    line_luminances = [y for y in enumerate(line_luminances)]
    line_luminances.sort(key=lambda y: -y[1])
    lightest_row_luminance = line_luminances[0][1]
    lightest_rows = []
    for row, lum in line_luminances:
        if(lum > lightest_row_luminance * threshold):
            lightest_rows.append(row)
    lightest_rows.sort()
    return lightest_rows
        
def colour_lightest_columns(img):
    rows = find_lightest_columns(img, 0.97)
    for row in rows:
        for x in range(img.height):
            img.load()[row,x] = 0
    img.show()
    
#img = Image.open(r'E:\botay\диплом\data\doc_0000_slice_0001.jpg')
#colour_lightest_rows(img)

def long_slicer(image_path, out_name, outdir):
    """slice an image into parts slice_size tall"""
    img = Image.open(image_path)
    width, height = img.size
    upper = 0
    left = 0
    rows = find_lightest_rows(img, 0.999)
    count = 1
    for i in range(len(rows)-1):
        if rows[i+1]-rows[i]<30 or rows[i+1]-rows[i]>40:
            continue
        else:
            upper = rows[i]
            lower = rows[i+1]
            #set the bounding box! The important bit
            bbox = (left, upper, width, lower)
            working_slice = img.crop(bbox)
            #save the slice
            working_slice.save(os.path.join(outdir, out_name + "_slice_" + str(count).rjust(4, '0') + ".jpg"))
            count += 1

path_npa = (r'E:\botay\диплом\train_sample\npa')
strings_save_path = (r'E:\botay\диплом\data_for_test')

def collect_strings(docs_path, strings_save_path, stop):
    count = 0
    images = os.listdir(docs_path)
    for doc in images:
        image_path = os.path.join(docs_path, doc)
        long_slicer(image_path, 'doc_' + str(count).rjust(4, '0'), strings_save_path)
        count += 1
        saved = os.listdir(strings_save_path)
        if (len(saved) > stop): 
            for trash in saved[stop:]:
                os.remove(os.path.join(strings_save_path, trash))
            break
    
collect_strings(path_npa, strings_save_path, 50)


        


