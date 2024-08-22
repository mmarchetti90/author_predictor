#!/usr/bin/env python3

### ---------------------------------------- ###

class book_to_image:

    """
    Class for turning books into image sets using a process akin to frequency chaos game
    representation.

    Parameters
    ----------
    books_path : string
        Path to directory containing books in txt format.
        Note that each book must be named in the following format:
        "<title> - <author>.txt"

    Attributes
    ----------
    books : pd.DataFrame
        Pandas data-frame with info on the books located at books_path.

    half_image_size : int
        Half the size of the output images.

    image_size : int
        Size of the output images.
    
    sentences_per_image : int
        Number of sentences to be used to generate each image.
    
    alphabet : str
        Characters to be used for embedding.
        i.e. abcdefghijklmnopqrstuvwxyz, '"
    
    sentence_breakers : str
        String with characters that define the end of a sentence
        i.e. .;!?
    
    int_to_char : dict
        Dictionary to convert indexes to their respective character in the alphabet.
    
    char_to_int : dict
        Dictionary to convert characters to their respective index in the alphabet.
    
    letter_coords : dict
        2D coordinates of letters in alphabet used for embedding.
    
    embeddings_manifest : pd.DataFrame
        Pandas data-frame of generated images info.

    Methods
    -------
    list_books()
        Static method that parses the list of books at books_path and generates a data frame.

    generate_images()
        Generates image embeddings for each book.
        Wrapper for book_embedding().

    book_embedding()
        Reads in a book and generates image embeddings.

    generate_image_identifier()
        Static method that creates a unique identifier for each image.

    make_letter_vectors()
        Static method to generate the reference coordinates for frequency chaos game
        representation.

    get_midpoint()
        Static method used to calculate frequency chaos game representation coordinates.
    """
    
    ### ------------------------------------ ###
    ### INIT                                 ###
    ### ------------------------------------ ###
    
    def __init__(self, books_path='./'):
        
        self.books = self.list_books(books_path)
    
    ### ------------------------------------ ###
    
    @staticmethod
    def list_books(path):

        """
        Static method that parses the list of books at books_path and generates a data frame.

        Parameters
        ----------
        books_path : string
            Path to directory containing books in txt format.
            Note that each book must be named in the following format:
            "<title> - <author>.txt"
        """
        
        manifest = {col : [] for col in ['author', 'title', 'path']}
        
        for file in listdir(path):
            
            if file.endswith('.txt'):
                
                *title, author = file.replace('.txt', '').split(' - ')
                title = ' - '.join(title)
                
                manifest['author'].append(author)
                manifest['title'].append(title)
                manifest['path'].append(f'{path}/{file}')
                
        manifest = pd.DataFrame(manifest)
        
        return manifest
    
    ### ------------------------------------ ###
    ### IMAGE EMBEDDING                      ###
    ### ------------------------------------ ###
    
    def generate_images(self, image_size=128, images_per_author=100, sentences_per_image=250, output_dir_name='book_images', group_by_author=True):
        
        """
        Generates image embeddings for each book.
        Wrapper for book_embedding().

        Parameters
        ----------
        image_size : int, optional
            Size of output images in pixels.
            Default = 128

        images_per_author : int, optional
            Number of images to generate for each author.
            If group_by_author is set to false, then it indicates the number of images to generate
            for each book.
            Default=100

        sentences_per_image : int, optional
            Number of sentences to use to generate each image.
            Note that this parameter directly impacts the amount of information in each image.
            Default=250

        output_dir_name : string, optional
            Name of autput directory to be created.
            Default='book_images'

        group_by_author : bool, optional
            If True, the output images will be grouped by author, i.e. all images related to a
            specific author will be in a subfolder named with the author's name. Set to true to
            generate training/testing images.
            If False, the output images will be grouped by book title, i.e. all images related to a
            specific book will be in a subfolder named with the book's title. Set to true to
            generate images of books you want to use for prediction.
            Default=True
        """

        self.half_image_size = int(image_size / 2)
        self.image_size = int(self.half_image_size * 2)
        
        self.sentences_per_image = sentences_per_image
        
        # Init output path
        output_path, n = output_dir_name, 0
        while exists(output_path):
            
            n += 1
            output_path = f'{output_dir_name}_{n}'
        
        mkdir(output_path)
        
        # Init alphabet
        self.alphabet = "abcdefghijklmnopqrstuvwxyz, '" + '"'
        self.sentence_breakers = '.;!?'
        self.int_to_char = {idx : letter for letter,idx in enumerate(self.alphabet)}
        self.char_to_int = {letter : idx for letter,idx in enumerate(self.alphabet)}
        
        # Define letters matrix
        self.letter_coords = self.make_letter_vectors(self.alphabet, self.half_image_size)
        
        # Init embeddings manifest
        embeddings_manifest = {col : [] for col in ['author', 'title', 'image_id']}
        
        # Embedd books to images
        n_authors = len(set(self.books.author))
        for n,author in enumerate(set(self.books.author)):
            
            if (n + 1) % 10 == 0:
                
                print(f'Processed {n + 1} / {n_authors} authors', end='\r')
            
            books_sub = self.books.loc[self.books.author == author]
            
            author = author.replace(' ', '_')

            if group_by_author:

                sub_dir = author
            
                mkdir(f'{output_path}/{sub_dir}')
                
                images_per_book = int(max(1, images_per_author / books_sub.shape[0]))

            else:

                images_per_book = images_per_author
        
            for n,(_, title, path) in books_sub.iterrows():

                if not group_by_author:

                    sub_dir = title.replace(' ', '_')

                    mkdir(f'{output_path}/{sub_dir}')
                
                # Embedd
                book_embeddings = self.book_embedding(path, images_per_book)
                
                # Log each image, while generating unique identifiers
                for be in book_embeddings:
                    
                    # Generate a unique identifier
                    identifier = self.generate_image_identifier()
                    while identifier in embeddings_manifest['image_id']:
                        
                        identifier = self.generate_image_identifier()
                    
                    # Log image
                    embeddings_manifest['author'].append(author)
                    embeddings_manifest['title'].append(title)
                    embeddings_manifest['image_id'].append(identifier)
                    
                    # Save image
                    be = Image.fromarray(be).convert('L')
                    be.save(f'{output_path}/{sub_dir}/{identifier}.png')
            
        embeddings_manifest = pd.DataFrame(embeddings_manifest)
        
        embeddings_manifest.sort_values(by=['author', 'title'], inplace=True)
        
        embeddings_manifest.to_csv(f'{output_path}/images_manifest.tsv', sep='\t', header=True, index=False)
        
        self.embeddings_manifest = embeddings_manifest
        
    ### ------------------------------------ ###
    
    def book_embedding(self, path, images_per_book=100):

        """
        Reads in a book and generates image embeddings.
        Note that, being wrapped by self.generate_images, this method will take several parameters
        from the wrapper.

        Parameters
        ----------
        path : string
            Path to input txt book.

        images_per_book : int, optional
            Number of images to generate for the book.
            Default=100
        """
        
        # Read book
        book_data = open(path).read().lower()
        
        # Remove unwanted characters
        bad_chars = [c for c in list(set(book_data)) if c not in self.alphabet + self.sentence_breakers and c != ' ']
        for bc in bad_chars:
            
            book_data = book_data.replace(bc, '')
        
        # Replace sentence_breakers with '.'
        for sb in self.sentence_breakers:
            
            book_data = book_data.replace(sb, '.')

        # Split into sentences
        book_data = book_data.split('.')
        
        # Remove top and bottom 5% to account for preface, glossaries, etc
        n_cut = int(len(book_data) * 0.05)
        book_data = book_data[n_cut : - n_cut]
        
        # Remove sentences with less than 10 characters
        book_data = [bd for bd in book_data if len(bd) > 10]
        
        # Generate images
        usable_book_size = max(1, len(book_data) - self.sentences_per_image)
        start_word_indexes = np.random.choice(range(usable_book_size), replace=False, size=min(images_per_book, usable_book_size))
        
        embeddings = []
        
        for idx in start_word_indexes:

            # Subset book
            book_data_sub = book_data[idx : idx + self.sentences_per_image]

            # Init embedding
            book_data_embedding = np.zeros((self.image_size, self.image_size))
    
            for sentence in book_data_sub:
                
                coords = (0, 0)
                for word in sentence.split(' '):
                    
                    if not len(word):
                        
                        continue
                    
                    for n,letter in enumerate(word):
                        
                        coords = self.get_midpoint(self.letter_coords[letter], coords)
                    
                    coords = (max(0, min(self.image_size - 1, coords[0] + self.half_image_size)),
                              max(0, min(self.image_size - 1, coords[1] + self.half_image_size)))
                    
                    book_data_embedding[coords] += 1
        
            book_data_embedding = np.ceil(255 * book_data_embedding / book_data_embedding.max())
            book_data_embedding[book_data_embedding > 255] = 255
            book_data_embedding = book_data_embedding.astype('uint8')
        
            embeddings.append(book_data_embedding)
        
        return embeddings

    ### ------------------------------------ ###
    
    @staticmethod
    def generate_image_identifier(size=16):

        """
        Static method that creates a unique identifier for each image.

        Parameters
        ----------

        size : int, optional
            Length of the unique alphanumeric identifier
            Default = 16
        """
        
        identifier = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'), replace=True, size=size))
        
        return identifier
    
    ### ------------------------------------ ###
    
    @staticmethod
    def make_letter_vectors(alphabet, half_fig_size=64):

        """
        Static method to generate the reference coordinates for frequency chaos game
        representation.

        Parameters
        ----------

        alphabet : str
            String with all the characters making up the alphabet to be used for representation.

        half_fig_size : int, optional
            Half the size of the output images in pixels.
            Default = 64
        """
        
        angle_step = 360 / len(alphabet)
        letter_coords = {}
        for n,letter in enumerate(alphabet):
            
            angle = angle_step * n
            x, y = cos(radians(angle)), sin(radians(angle))
            m = y / x
            if angle >= 315 or angle <= 45:
                
                projected_x = 1
                projected_y = m * projected_x
            
            elif angle <= 135:
                
                projected_y = 1
                projected_x = projected_y / m
            
            elif angle <= 225:
                
                projected_x = -1
                projected_y = m * projected_x
            
            else:
                
                projected_y = - 1
                projected_x = projected_y / m
                
            letter_coords[letter] = (projected_x, projected_y)
        
        letter_coords = {letter : (round(half_fig_size * y), round(half_fig_size * x)) for letter,(y, x) in letter_coords.items()}
        
        return letter_coords
    
    ### ------------------------------------ ###
    
    @staticmethod
    def get_midpoint(a, b):

        """
        Static method used to calculate frequency chaos game representation coordinates.

        Parameters
        ----------

        a : tuple
            X and Y coordinates of the origin point.

        b : tuple
            X and Y coordinates of the destination point.
        """
        
        y = (a[0] - b[0]) // 2
        x = (a[1] - b[1]) // 2
        
        return (y, x)
    
### ------------------MAIN------------------ ###

import numpy as np
import pandas as pd

from math import cos,sin,radians
from os import listdir, mkdir
from os.path import exists
from PIL import Image
