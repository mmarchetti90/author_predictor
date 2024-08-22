# Residual Convolutional Neural Network for the prediction of authors based on books' text.

The model can be used to predict the author of a book based on their writing "style" (see below for better description).
Also, if an author was not used for building the model (i.e. does not belong to any prediction class), then the model can be used to have an idea of what other author has the most similar "style".
The model works great especially for authors that have a distinguished and consistent writing style (e.g. there's only one Tolkien!).

/// ---------------------------------------- ///

## Overview:

### Book embedding:

Using a process akin to frequency chaos game representation, books (in txt format) are converted to images as follows:

* A book is read and undesired characters are removed (e.g. dates, special characters).\
  This simplifies the text, but preserves words and sentences.

* The book's text is broken into sentences and each sentence into its constituent words.

* Each letter of the filtered alphabet is assigned a position on a square matrix, equally spaced along its 4 sides.

* Images are then generated, each using a different random subset of sentences.\
  Each sentence is broken into words, and each word is positioned on the matrix by using frequency chaos game representation.\
  The initial position of a word is initialized as (0, 0) (for the first word in a sentence) or as the final position of the previous word.\
  Each word is broken into its constituent letters, then the position of the word on the matrix is updated as the mid point between the current position and the coordinates of the next letter.\
  Once the final position is found, the corresponding matrix "pixel" is updated with a +1 count.
  Note that in this way, images information is dependent on how long sentences are and on their structure (i.e. what words the author choses and what is their order). This in turn can be considered as the "style" of the author, which is then captured by the neural network.

* Lastly, images are normalized to 0-255 grayscale and saved.

<p align="center">
  <img width="1852" height="1795" src="https://github.com/mmarchetti90/author_predictor/blob/main/data/book_images_example.png">
</p>

### Residual Convolutional Neural Network:

* The model reads in book images and returns a prediction of authorship.

* See data/model_layers.txt for details.

* Training was tested on a M1 MacBook Pro using the following parameters:
    * ~ 100 images per author
    * 127 authors total (mostly sci-fi/fantasy fiction)
    * 100 images per batch
    * 0.2 validation split

* Example of training/validation loss:

<p align="center">
  <img width="1229" height="469" src="https://github.com/mmarchetti90/author_predictor/blob/main/data/model_loss.png">
</p>

* Resulting testing accuracy was 98.66%.

* The model was further tested by predicting the authors of 5 books not used for training, but whose authors do belong to the model output classes. For example, the was trained using the "Lord of the Rings" trilogy by J.R.R. Tolkien and correctly predicted with 89% certainty the authorship of "The Children of Húrin", which was a book not used for training.\
  The model can also be used on books whose authors do not belong to the model. In this case, the prediction will match authors with the most similar "style" to the actual author.

/// ---------------------------------------- ///

## DEPENDENCIES:

Python 3.7.17+ &

	matplotlib
	numpy
	pandas
	PIL

Tensorflow 2.10.0+
