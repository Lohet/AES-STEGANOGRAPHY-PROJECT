# stepic - Python image steganography
# Copyright (C) 2007 Lenny Domnitser
# Copyright (C) 2018,2020 Scott Kitterman
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

'''Python image steganography

Stepic hides arbitrary data inside Pillow images.

Stepic uses the Python Image Library (Pillow)
(apt: python3-pil, web: <https://pypi.org/project/Pillow/>).
'''

__author__ = 'Lenny Domnitser <http://domnit.org/>'
__version__ = '0.5.0'


import argparse
import sys
import traceback
import warnings
try:
    from PIL import Image
except:
    warnings.warn('Could not find PIL. Only encode_imdata and decode_imdata will work.',
                  ImportWarning, stacklevel=2)


__all__ = ('encode_imdata','encode_inplace', 'encode',
           'decode_imdata', 'decode',
           'Steganographer',
           'encode_files', 'decode_files')


def _validate_image(image):
    if image.mode not in ('RGB', 'RGBA', 'CMYK'):
        raise ValueError('Unsupported pixel format: '
                         'image must be RGB, RGBA, or CMYK')
    if image.format == 'JPEG':
        raise ValueError('JPEG format incompatible with steganography')


def encode_imdata(imdata, data):
    '''given a sequence of pixels, returns an iterator of pixels with
    encoded data'''

    datalen = len(data)
    if datalen == 0:
        raise ValueError('data is empty')
    if datalen * 3 > len(imdata):
        raise ValueError('data is too large for image')

    imdata = iter(imdata)

    for i in range(datalen):
        pixels = [value & ~1 for value in
                  imdata.__next__()[:3] + imdata.__next__()[:3] + imdata.__next__()[:3]]
        byte = data[i]
        for j in range(7, -1, -1):
            pixels[j] |= byte & 1
            byte >>= 1
        if i == datalen - 1:
            pixels[-1] |= 1
        pixels = tuple(pixels)
        yield pixels[0:3]
        yield pixels[3:6]
        yield pixels[6:9]


def encode_inplace(image, data):
    '''hides data in an image'''

    _validate_image(image)

    w = image.size[0]
    (x, y) = (0, 0)
    for pixel in encode_imdata(image.getdata(), data):
        image.putpixel((x, y), pixel)
        if x == w - 1:
            x = 0
            y += 1
        else:
            x += 1


def encode(image, data):
    '''generates an image with hidden data, starting with an existing
    image and arbitrary data'''

    image = image.copy()
    encode_inplace(image, data)
    return image


def decode_imdata(imdata):
    '''Given a sequence of pixels, returns an iterator of characters
    encoded in the image'''

    imdata = iter(imdata)
    while True:
        pixels = list(imdata.__next__()[:3] + imdata.__next__()[:3] + imdata.__next__()[:3])
        byte = 0
        for c in range(7):
            byte |= pixels[c] & 1
            byte <<= 1
        byte |= pixels[7] & 1
        yield chr(byte)
        if pixels[-1] & 1:
            break


def decode(image):
    '''extracts data from an image'''

    _validate_image(image)

    return ''.join(decode_imdata(image.getdata()))


class Steganographer:
    'deprecated'
    def __init__(self, image):
        self.image = image
        warnings.warn('Steganographer class is deprecated, and will be removed before 1.0',
                      DeprecationWarning, stacklevel=2)
    def encode(self, data):
        return encode(self.image, data)
    def decode(self):
        return decode(self.image)


def encode_files(image_in, data_in, image_out, format):
    """Main function for encoding data in image files.
    @param image_in: Filename of image in which to embed data
    @param data_in: Filename of text file containing data
    @param image_out: Output file name
    @param format: Format of ouput image, use 'None' to keep the input format
    @return: None (image_out written to disk)
    """
    image = Image.open(image_in)
    if not hasattr(data_in, 'read'):
        data_in = open(data_in, 'rb')
    data = data_in.read()
    if format is None and hasattr(image_out, 'write'):
        format = image.format
    encode_inplace(image, data)
    image.save(image_out, format)
    data_in.close()
    image.close()


def decode_files(image_in, data_out):
    """Main function for decoding data from image files.
    @param image_in: Filename of image containing embeded data
    @param data_out: Filename of text file to write the data
    @return: None (data written to disk in data_out)
    """
    image = Image.open(image_in)
    if not hasattr(data_out, 'write'):
        data_out = open(data_out, 'w')
    data_out.write(decode(image))
    data_out.close()


def main():
    parser = argparse.ArgumentParser(description='Hide data in an image, or read hidden data from an image.')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('-d', '--decode', action='store_true', default=False,
                      help='given an image, write out the hidden data file')
    parser.add_argument('-e', '--encode', action='store_true', default=False,
                      help='given an image and data file, write out a new image file with the data hidden in it')
    parser.add_argument('-f', '--format', metavar='FORMAT',
                      help='output image format (optional)')
    parser.add_argument('-i', '--image-in=', dest='image_in', metavar='FILE', default=sys.stdin,
                      help='read in image FILE for decoding or encoding')
    parser.add_argument('-t', '--data-in=', dest='data_in', metavar='FILE', default=sys.stdin,
                      help='read in data FILE for encoding')
    parser.add_argument('-o', '--out=', dest='out', metavar='FILE', default=sys.stdout,
                      help='write out to FILE, data when decoding, image when encoding')
    args = parser.parse_args()

    if args.decode == args.encode:
        parser.print_usage()
        sys.stderr.write('Choose either encode (-e) or decode (-d).\n')

    try:
        if args.decode:
            decode_files(args.image_in, args.out)
        elif args.encode:
            encode_files(args.image_in, args.data_in, args.out, args.format)
    except (TypeError, ValueError) as e:
        message = 'error: ' + str(e) + '\n'
        sys.stderr.write(message)
        sys.exit(1)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        message = e.__class__.__name__ + ': ' + str(e) + '\n'
        sys.stderr.write(message)
        if args.debug:
            traceback.print_tb(sys.exc_traceback)
        sys.exit(3)

if __name__ == '__main__':
    main()
