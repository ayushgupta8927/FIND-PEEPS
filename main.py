from flask import Flask, send_file, render_template, request, redirect
import os
import zipfile
import cv2
import sys
from flask.helpers import url_for
import numpy
import time
from PIL import Image, ImageOps

from matplotlib import pyplot
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from mtcnn.mtcnn import MTCNN

# draw the image for the Detected Faces, alongside their Features pointed out:::

app = Flask(__name__)
app.config["Image_UPLOADS"] = os.getcwd() + "/static/"
app.config["FACE_IMAGES"] = os.getcwd() + "/static/face_output/"


@app.context_processor
def handle_context():
    '''Inject object into jinja2 templates.'''
    return dict(os=os)


@app.route('/')
def main():
    return render_template('main.html', theImageSource='myCameraLogo.png', theWidth=500, theHeight=300)


@app.route('/upload', methods=['GET'])
def redirection():
    return render_template('main.html')


@app.route('/upload', methods=['POST'])
def upload():
    if(request.method == 'POST'):
        mySrc = request.files['mySrc']
        mySrc.save(os.path.join(app.config["Image_UPLOADS"], mySrc.filename))
        myWidth = request.form.get('myWidth')
        myHeight = request.form.get('myHeight')

        os.chdir(str(app.config["Image_UPLOADS"]))
        print(os.getcwd())

        try:
            os.remove('Operated.jpg')
        except:
            pass

        filename = mySrc.filename
        # load image from file
        pixels = pyplot.imread(filename)
        # create the detector, using default weights
        # detect faces in the image
        detector = MTCNN()

        def draw_image_with_boxes(filename, result_list):
            data = pyplot.imread(filename)
            pyplot.imshow(data)
            ax = pyplot.gca()
            # plot each box
            for result in result_list:
                x, y, width, height = result['box']
                # create the shape
                rect = Rectangle((x, y), width, height,
                                 fill=False, color='orange')
                # draw the box
                ax.add_patch(rect)
                # draw the dots
                for key, value in result['keypoints'].items():
                    # create and draw dot
                    dot = Circle(value, radius=2, color='blue')
                    ax.add_patch(dot)
            pyplot.savefig('Operated.jpg')

        def remove_files():
            os.chdir(str(app.config["Image_UPLOADS"]))
            path = os.path.join(os.getcwd(), 'face_output')
            for f in os.listdir(path):
                os.remove(os.path.join(path, f))

        def draw_faces(filename, result_list):
            data = pyplot.imread(filename)
            im = Image.open(filename)
            remove_files()
            for i in range(len(result_list)):
                x1, y1, width, height = result_list[i]['box']
                x2, y2 = x1+width, y1+height
                pyplot.subplot(1, len(result_list), i+1)
                pyplot.axis('off')
                face_file = "faces" + str(i) + ".jpg"
                im = im.crop((x1, y1, x2, y2))
                pyplot.imsave(os.path.join(
                    os.getcwd(), 'face_output', face_file), data[y1:y2, x1:x2])

        faces = detector.detect_faces(pixels)
        # display faces on the original image

        draw_image_with_boxes(filename, faces)
        draw_faces(filename, faces)

        return render_template('results.html', theImageSource='Operated.jpg')


@app.route('/download-files')
def return_files():
    os.remove("Facefiles.zip")
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'face_output')
    print(UPLOAD_FOLDER)
    zipfolder = zipfile.ZipFile(
        'Facefiles.zip', 'w', compression=zipfile.ZIP_STORED)  # Compression type

    # zip all the files which are inside in the folder
    for f in os.listdir(app.config["FACE_IMAGES"]):
        zipfolder.write(app.config["FACE_IMAGES"] + f)
    zipfolder.close()

    return send_file('static/Facefiles.zip',
                     mimetype='zip',
                     attachment_filename='Facefiles.zip',
                     as_attachment=True)

    # Delete the zip file if not needed


if(__name__ == '__main__'):
    app.run(debug=True)
