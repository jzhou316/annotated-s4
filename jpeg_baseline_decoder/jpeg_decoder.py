"""JPEG decoder from bytes.

Able to output intermediate results (Huffman decoding, de-quantization, recovered DCT coefficients), etc.

With self-contained and thorough error handling and information printing.

NOTE
- only works for the baseline JPEG compression type
- only works with 4:4:4 color subsampling (no chroma subsampling)
- not taking care of the Researt Interval for DC delta encoding
- not supporting 4 channels CMYK
"""
from multiprocessing.sharedctypes import Value
from struct import unpack
import math
from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# add handler to output to console if not existing
if not logger.handlers:
    # Initialize the console logging
    c_handler = logging.StreamHandler()
    # c_handler.setLevel(logging.DEBUG)
    # c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_format = logging.Formatter('%(message)s')
    c_handler.setFormatter(c_format)
    logger.addHandler(c_handler)


"""
http://imrannazar.com/Let%27s-Build-a-JPEG-Decoder%3A-File-Structure

Most of the different types of segment have a length value (taking 2 bytes) right after the marker,
which denotes how long the segment is in bytes (including the length value);
this can be used to skip over segments that a decoder does not know about.

https://en.wikipedia.org/wiki/JPEG_File_Interchange_Format
Mostly with the following pattern:
FF xx s1 s2 [data bytes]
The bytes s1 and s2 are taken together to represent a big-endian 16-bit integer specifying the length
of the following "data bytes" plus the 2 bytes used to represent the length. In other words, s1 and s2
specify the number of the following data bytes as {\displaystyle 256\cdot s1+s2-2}{\displaystyle 256\cdot s1+s2-2}.

There are three exceptions to this general rule: SOI, EOI, SOS.
"""
marker_mapping = {
    0xFFD8: "Start of Image (SOI)",
    0xFFE0: "Application Default Header (APP0)",    # JFIF (there are also other APP markers)
    0xFFDB: "Define Quantization Table (DQT)",
    0xFFC0: "Start of Frame (SOF, Baseline DCT)",
    0xFFDD: "Define Restart Interval (DRI)",    # NOTE not taken care of
    0xFFC4: "Define Huffman Table (DHT)",
    0xFFDA: "Start of Scan (SOS)",
    0xFFD9: "End of Image (EOI)",
}
# marker synonyms
SOF0 = 0xFFC0
SOF1 = 0xFFC1
SOF2 = 0xFFC2
SOF3 = 0xFFC3

SOF5 = 0xFFC5
SOF6 = 0xFFC6
SOF7 = 0xFFC7

SOF9 = 0xFFC9
SOF10 = 0xFFCA
SOF11 = 0xFFCB

SOF13 = 0xFFCD
SOF14 = 0xFFCE
SOF15 = 0xFFCF

# Define Huffman Table(s)
DHT = 0xFFC4

# JPEG extensions
JPE = 0xFFC8

# Define Arithmetic Coding Conditioning(s)
DAC = 0xFFCC

# Restart Interval Markers
RST0 = 0xFFD0
RST1 = 0xFFD1
RST2 = 0xFFD2
RST3 = 0xFFD3
RST4 = 0xFFD4
RST5 = 0xFFD5
RST6 = 0xFFD6
RST7 = 0xFFD7

# Other Markers
SOI = 0xFFD8
EOI = 0xFFD9
SOS = 0xFFDA
DQT = 0xFFDB
DNL = 0xFFDC
DRI = 0xFFDD
DHP = 0xFFDE
EXP = 0xFFDF

# TODO finish this and add comments, and change the comparison in decode to have if marker == SOI instead of 0xFFD8



def PrintMatrix(m):
    """
    A convenience function for printing matrices
    """
    for j in range(8):
        print("|", end="")
        for i in range(8):
            print("%d  |" % m[i + j * 8], end="\t")
        print()
    print()


def Clamp(col):
    """
    Makes sure col is between 0 and 255.
    """
    col = 255 if col > 255 else col
    col = 0 if col < 0 else col
    return int(col)


def ColorConversion(Y, Cr, Cb):
    """
    Converts Y, Cr and Cb to RGB color space
    """
    R = Cr * (2 - 2 * 0.299) + Y
    B = Cb * (2 - 2 * 0.114) + Y
    G = (Y - 0.114 * B - 0.299 * R) / 0.587
    return (Clamp(R + 128), Clamp(G + 128), Clamp(B + 128))


def DrawMatrix(x, y, matL, matCb, matCr):
    """
    Loops over a single 8x8 MCU and draws it on Tkinter canvas
    """
    for yy in range(8):
        for xx in range(8):
            c = "#%02x%02x%02x" % ColorConversion(
                matL[yy][xx], matCb[yy][xx], matCr[yy][xx]
            )
            x1, y1 = (x * 8 + xx) * 2, (y * 8 + yy) * 2
            x2, y2 = (x * 8 + (xx + 1)) * 2, (y * 8 + (yy + 1)) * 2
            w.create_rectangle(x1, y1, x2, y2, fill=c, outline=c)


def map_mcu2img(image, width, x, y, matL, matCb, matCr):
    """Map the 8x8 MCU to the correct positions in the image to be recovered.
    """
    for yy in range(8):
        for xx in range(8):
            image[(x*8+xx) + ((y*8+yy) * width)] = ColorConversion(matL[yy][xx], matCb[yy][xx], matCr[yy][xx])

    return


def removeFF00(data):
    """
    Removes 0x00 after 0xff in the image scan section of JPEG
    """
    datapro = []
    i = 0
    while True:
        b, bnext = unpack("BB", data[i : i + 2])
        if b == 0xFF:
            if bnext != 0:
                break
            datapro.append(data[i])
            i += 2
        else:
            datapro.append(data[i])
            i += 1
    return datapro, i


def clean_scan_bitstream(data):
    """Do a pass over the bitstream data inside the scan to clean up the bitstream, to prepare for Huffman decoding.
    The `data` should start with the bitstream inside Scan section but after the SOS header
    (only starting the entropy-encoded bits).

    The cleaning includes:
    - Remove 0x00 after 0xFF in the scan bitstream, so that 0xFF is included in the entropy-coded data (byte stuffing)
      https://docs.fileformat.com/image/jpeg/
    - Remove 0xFF after 0xFF, as consecutive 0xFF can exist
    - Remove RST0 - RST7 markers (restart interval markers)
    - Stop at the other 0xFF markers; should be EOI (0xFFD9) for baseline JPEG.
    """
    data_huffman = []
    i = 0
    while True:
        b, bnext = unpack('BB', data[i:i+2])
        if b == 0xFF:
            if bnext == 0xFF:
                # consecutive 0xFF -> skip to only keep one
                i += 1
            elif bnext == 0x00:
                # 0xFF00 -> byte stuffing, remove 0x00 and keep 0xFF in the entropy-coded bitstream
                data_huffman.append(data[i])
                i += 2
            elif RST0 <= bnext <= RST7:
                # restart interval markers -> skip (restart interval can be decided with the DRI segment from the header)
                i += 2
            elif bnext == 0xD9:
                # EOS marker -> end the bitstream
                break
            else:
                raise ValueError(f'Error - invalid marker {data[i:i+2]} encountered in the Scan section for baseline JPEG '
                                 '(other types, e.g. progressive JPEG is unsupported now)/n')
        else:
            data_huffman.append(data[i])
            i += 1

    return data_huffman, i





def get_array(type, bitstream, length):
    """
    A convenience function for unpacking an array from bitstream
    """
    assert type in ['B', 'H'], 'only supporting reading 8-bit value (1 byte) or 16-bit value (2 bytes) to unsigned int'
    s = ""
    for i in range(length):
        s = s + type
    if type == 'B':
        array = list(unpack(s, bitstream[:length]))
    elif type == 'H':
        s = '>' + s    # big-endian
        array = list(unpack(s, bitstream[:2 * length]))
    else:
        ...
    return array


def DecodeNumber(code, bits):
    l = 2 ** (code - 1)
    if bits >= l:
        return bits
    else:
        return bits - (2 * l - 1)


class IDCT:
    """
    An inverse Discrete Cosine Transformation Class
    """

    def __init__(self):
        self.base = [0] * 64
        self.zigzag = [
            [0, 1, 5, 6, 14, 15, 27, 28],
            [2, 4, 7, 13, 16, 26, 29, 42],
            [3, 8, 12, 17, 25, 30, 41, 43],
            [9, 11, 18, 24, 31, 40, 44, 53],
            [10, 19, 23, 32, 39, 45, 52, 54],
            [20, 22, 33, 38, 46, 51, 55, 60],
            [21, 34, 37, 47, 50, 56, 59, 61],
            [35, 36, 48, 49, 57, 58, 62, 63],
        ]
        self.idct_precision = 8
        self.idct_table = [
            [
                (self.NormCoeff(u) * math.cos(((2.0 * x + 1.0) * u * math.pi) / 16.0))
                for x in range(self.idct_precision)
            ]
            for u in range(self.idct_precision)
        ]

    def NormCoeff(self, n):
        if n == 0:
            return 1.0 / math.sqrt(2.0)
        else:
            return 1.0

    def rearrange_using_zigzag(self):
        for x in range(8):
            for y in range(8):
                self.zigzag[x][y] = self.base[self.zigzag[x][y]]
        return self.zigzag

    def perform_IDCT(self):
        out = [list(range(8)) for i in range(8)]

        for x in range(8):
            for y in range(8):
                local_sum = 0
                for u in range(self.idct_precision):
                    for v in range(self.idct_precision):
                        local_sum += (
                            self.zigzag[v][u]
                            * self.idct_table[u][x]
                            * self.idct_table[v][y]
                        )
                out[y][x] = local_sum // 4
        self.base = out


class HuffmanTable:
    """
    A Huffman Table class
    """

    def __init__(self):
        self.root = []
        self.elements = []

    def BitsFromLengths(self, root, element, pos):
        if isinstance(root, list):
            if pos == 0:
                if len(root) < 2:
                    root.append(element)
                    return True
                return False
            for i in [0, 1]:
                if len(root) == i:
                    root.append([])
                if self.BitsFromLengths(root[i], element, pos - 1) == True:
                    return True
        return False

    def GetHuffmanBits(self, lengths, elements):
        self.elements = elements
        ii = 0
        for i in range(len(lengths)):
            for j in range(lengths[i]):
                self.BitsFromLengths(self.root, elements[ii], i)
                ii += 1

    def Find(self, st):
        r = self.root
        while isinstance(r, list):
            r = r[st.get_bit()]
        return r

    def get_symbol(self, st):
        while True:
            res = self.Find(st)
            if res == 0:
                return 0
            elif res != -1:
                return res


class BitStream:
    """
    A bit stream class with convenience methods
    """

    def __init__(self, data):
        self.data = data
        self.pos = 0

    def get_bit(self):
        b = self.data[self.pos >> 3]
        s = 7 - (self.pos & 0x7)
        self.pos += 1
        return (b >> s) & 1

    def get_bit_n(self, n):
        val = 0
        for _ in range(n):
            val = val * 2 + self.get_bit()
        return val

    def len(self):
        return len(self.data)


@dataclass(frozen=False)
class ColorComponent:
    """Store the information of each color component read from Start of Frame (SOF) segment."""
    component_id: int
    sampling_factor_vertical: int
    sampling_factor_horizontal: int
    quantization_table_id: int
    huffman_DC_table_id: int = None
    huffman_AC_table_id: int = None
    used_in_scan: bool = False


class JPEG:
    """
    JPEG class for decoding a baseline encoded JPEG image
    """

    def __init__(self, jpeg_file=None, jpeg_bytes=None, verbose=True, verbose_details=True):
        # set logging level
        if verbose:
            logger.setLevel(logging.INFO)
            if verbose_details:
                # this will print our more details, such as quantization tables and Huffman table information
                logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.NOTSET)

        # header information
        self.huffman_tables = {}
        self.quantization_tables = {}    # maximum 4, with ids 0 (DC), 1 (AC), 2, 3
        self.quantMapping = []
        self.color_components = {}    # maximum 4, with ids starting from 1 (or 0, but not standard)
        self.num_components = 0
        self.huffman_DC_tables = {}    # maximum 4, with ids 0, 1, 2, 3
        self.huffman_AC_tables = {}    # maximum 4, with ids 0, 1, 2, 3

        self.height = 0
        self.width = 0
        self.block_height = 0
        self.block_width = 0

        self.restart_interval = 0

        # NOTE these are only used for progressive JPEG -> for baseline JPEG, these values are fixed
        #      (there is only one scan)
        self.start_of_selection: int = 0
        self.end_of_selection: int = 63
        self.successive_approximation_high: int = 0
        self.successive_approximation_low: int = 0

        # intermediate decoding results
        self.huffman_decoded_seq_stream = None
        self.huffman_decoded_seq_zero_recovered_stream = None

        # image to be recovered
        self.image = []

        # decode flag
        self.decoded = False

        if jpeg_file is not None:
            with open(jpeg_file, "rb") as f:
                self.img_data = f.read()
        else:
            assert jpeg_bytes is not None, 'jpg image path not provided -> must provide jpg bytes directly\n'
            self.img_data = jpeg_bytes

    def readQuantizationTable(self, data):
        # read the length
        (len_segment,) = unpack('>H', data[:2])
        data = data[2:]

        # read in the quantization table
        # NOTE there could be more than 1 tables in the same DQT segment (there could also be multiple DQT segments)
        length = len_segment - 2    # NOTE `len_segment` includes the length byte at the beginning
        while length > 0:

            # read one byte of table info
            (table_info,) = unpack('B', data[0:1])
            data = data[1:]
            length -= 1

            high_nibble, low_nibble = table_info >> 4, table_info & 0x0F
            value_bits, table_id = high_nibble, low_nibble
            assert value_bits in [0, 1], 'Error - quantization table value bit indicator must be a bool ' \
                '0: 8-bit values (most common), 1: 16-bit values\n'
            assert table_id <= 3, 'Error - maximum quantization table ID is 3 ' \
                '(JPEG allows 4 quantization tables with IDs 0, 1, 2, 3)\n'

            logger.info(f'quantization table: ID {table_id}')

            # read in the quantization table
            if value_bits == 0:
                # 8-bit values
                self.quantization_tables[table_id] = get_array('B', data[:64], 64)
                data = data[64:]
                length -= 64
            else:
                # 16-bit values
                self.quantization_tables[table_id] = get_array('H', data[:128], 64)
                data = data[128:]
                length -= 128

            logger.debug(' '.join(list(map(str, self.quantization_tables[table_id]))))

        if length != 0:
            raise ValueError('Error - DQT invalid\n')

        return len_segment

    def BuildMatrix(self, st, idx, quant, olddccoeff):
        i = IDCT()

        code = self.huffman_tables[0 + idx].get_symbol(st)
        bits = st.get_bit_n(code)
        dccoeff = DecodeNumber(code, bits) + olddccoeff

        i.base[0] = (dccoeff) * quant[0]
        l = 1
        while l < 64:
            code = self.huffman_tables[16 + idx].get_symbol(st)
            if code == 0:
                break

            # The first part of the AC key_len
            # is the number of leading zeros
            if code > 15:
                l += code >> 4
                code = code & 0x0F

            bits = st.get_bit_n(code)

            if l < 64:
                coeff = DecodeNumber(code, bits)
                i.base[l] = coeff * quant[l]
                l += 1

        i.rearrange_using_zigzag()
        i.perform_IDCT()

        return i, dccoeff

    def StartOfScan(self, data, hdrlen):
        data, lenchunk = removeFF00(data[hdrlen:])

        st = BitStream(data)
        oldlumdccoeff, oldCbdccoeff, oldCrdccoeff = 0, 0, 0
        for y in range(self.height // 8):
            for x in range(self.width // 8):
                matL, oldlumdccoeff = self.BuildMatrix(
                    st, 0, self.quantization_tables[self.quantMapping[0]], oldlumdccoeff
                )
                matCr, oldCrdccoeff = self.BuildMatrix(
                    st, 1, self.quantization_tables[self.quantMapping[1]], oldCrdccoeff
                )
                matCb, oldCbdccoeff = self.BuildMatrix(
                    st, 1, self.quantization_tables[self.quantMapping[2]], oldCbdccoeff
                )

                # DrawMatrix(x, y, matL.base, matCb.base, matCr.base)

                map_mcu2img(self.image, self.width, x, y, matL.base, matCb.base, matCr.base)

        return lenchunk + hdrlen

    def readRestartInterval(self, data):
        # read the length
        (len_segment,) = unpack('>H', data[:2])
        data = data[2:]

        assert len_segment == 4, 'Error - DRI segment length must be 4 bytes\n'
        self.restart_interval = unpack('>H', data[:2])

        assert self.restart_interval == 0, 'Unsupported - Restart Interval other than 0\n'

        return len_segment

    def readBaselineStartOfFrame(self, data):
        """
        NOTE only work for Baseline JPEG compression type.
             There are four types of compressed supported by JPEG standard:
             Baseline, Extended Sequential, Progressive, Lossless
        """
        if len(self.color_components) != 0:
            # there cannot be more than one Frame segment
            raise ValueError('Error - Multiple SOFs detected\n')

        # read the length
        (len_segment,) = unpack('>H', data[:2])
        data = data[2:]

        # read the header about image information
        precision, self.height, self.width, num_components = unpack(">BHHB", data[0:6])
        data = data[6:]
        self.block_height = (self.height + 7) // 8
        self.block_width = (self.width + 7) // 8

        assert precision == 8, f'Invalid precision {precision} -> must be 8'
        # bits/sample, always 8 (12 and 16 not supported by most software)

        logger.info("size of image: %ix%i" % (self.height,  self.width))
        logger.info("size of MCU blocks: %ix%i" % (self.block_height, self.block_width))
        logger.info(f'number of channels [1 for grayscale, 3 for YCbCr (or YIQ), 4 for CMYK]: {num_components}')
        assert num_components in [1, 3], f'Unsupported - {num_components} color components'

        # initialize a blank image of given size
        # NOTE there could be at most 1 SOF segment for baseline JPEG to specify the image information
        # TODO check this and print error if it violates
        self.image = [0] * (self.width * self.height)

        # loop over each channel
        for i in range(num_components):
            component_id, sampling_factor, QtbId = unpack("BBB", data[0:3])
            data = data[3:]
            # component_id (1 byte): component ID, 1 = Y, 2 = Cb, 3 = Cr, 4 = I, 5 = Q
            # sampling_factor (1 byte): bit 0-3 vertical, bit 4-7 horizontal
            # QtbID (1 byte): quantization table ID
            # NOTE component_id is usually 1, 2, 3 but rarely can be seens as 0, 1, 2

            # https://stackoverflow.com/questions/42896154/python-split-byte-into-high-low-nibbles
            high_nibble, low_nibble = sampling_factor >> 4, sampling_factor & 0x0F

            self.quantMapping.append(QtbId)

            # sampling factor could be 1, 2
            self.color_components[component_id] = ColorComponent(
                component_id,
                sampling_factor_vertical=low_nibble,
                sampling_factor_horizontal=high_nibble,
                quantization_table_id=QtbId
                )

            if sampling_factor != 17:
                # Or: if hex(sampling_factor) != '0x11'
                # both high_nibble and low_nibble would be 1, and the 8 bit number would be 17
                raise NotImplementedError('Unsupported - hex(sampling_factor) - '
                                          'Currently only support sampling factor 4:4:4 (no chroma subsampling)')

        logger.info('color components:')
        for k, v in self.color_components.items():
            logger.info(v)

        assert len_segment == 2 + 6 + 3 * num_components, 'Error - SOF invalid\n'

        return len_segment

    def readHuffmanTable(self, data):
        # read the length
        (len_segment,) = unpack('>H', data[:2])
        data = data[2:]

        length = len_segment - 2    # NOTE `len_segment` includes the length byte at the beginning
        while length > 0:

            # read one byte of table info
            (table_info,) = unpack('B', data[0:1])
            data = data[1:]
            length -= 1

            upper_nibble, lower_nibble = table_info >> 4, table_info & 0x0F
            is_ac_table = bool(upper_nibble)    # 0 or 1
            table_id = lower_nibble    # 0, 1, 2, or 3
            assert table_id <= 3, f'Error - Invalid Huffman table ID {table_id}\n'

            logger.info(f'Huffman table: {"DC" if not is_ac_table else "AC"} - ID {table_id}')

            # read the number of symbols/codes for length-1 to length-16 bit codes: 16 bytes
            code_counts_per_length = get_array('B', data[0:16], 16)
            data = data[16:]
            length -= 16

            # read the symbols
            symbols = []
            for i, code_counts in enumerate(code_counts_per_length, start=1):
                code_symbols = get_array('B', data[0:code_counts], code_counts)
                data = data[code_counts:]
                length -= code_counts

                symbols += code_symbols

                logger.debug(' ' * 4 + f'code_length {i}: ' + ' '.join([f'{hex(x)}' for x in code_symbols]))

            if len(symbols) > 176:
                raise ValueError(f'Error - Too many symbols in Huffman table {len(symbols)}\n')

            htable = HuffmanTable()
            htable.GetHuffmanBits(code_counts_per_length, symbols)

            if is_ac_table:
                self.huffman_AC_tables[table_id] = htable
            else:
                self.huffman_DC_tables[table_id] = htable

        if length != 0:
            raise ValueError('Error - DHT invalid\n')

        return len_segment

    def readUnknownSegment(self, data):
        # read the length
        (len_segment,) = unpack('>H', data[:2])
        data = data[2:]

        return len_segment

    def readStartOfScan(self, data):
        if len(self.color_components) == 0:
            # there cannot be more than one Frame segment
            raise ValueError('Error - SOS detected before SOF\n')

        # read the length
        (len_segment,) = unpack('>H', data[:2])
        data = data[2:]

        # read the number of components
        (num_components,) = unpack('B', data[:1])
        data = data[1:]

        if num_components != len(self.color_components):
            raise ValueError('Unsupported - Scan must have the same number of color components defined in SOF section. '
                             'Variations such as progressive JPEG is not supported.')
        self.num_components = num_components

        # iterate over the `num_components` components to define information for each color component
        for n in range(num_components):
            # component ID: 1 byte, should be 1, 2, 3 (4, 5 not supported) (could also start from 0 in some cases)
            (component_id,) = unpack('B', data[:1])
            data = data[1:]
            assert component_id in self.color_components

            # change color component state
            if self.color_components[component_id].used_in_scan:
                raise ValueError(f'Error - Duplicate color component ID in Scan {component_id}\n')
            self.color_components[component_id].used_in_scan = True

            # Huffman table ID: 1 byte, upper nibble for DC table ID, lower nibble for AC table ID
            (huffman_table_ids,) = unpack('B', data[:1])
            data = data[1:]
            upper_nibble, lower_nibble = huffman_table_ids >> 4, huffman_table_ids & 0x0F
            self.color_components[component_id].huffman_DC_table_id = upper_nibble
            self.color_components[component_id].huffman_AC_table_id = lower_nibble

        # read 3 more bytes: options for progressive JPEG -> these values are fixed for baseline JPEG
        assert self.start_of_selection == unpack('B', data[:1])[0]    # return type is always tuple
        data = data[1:]
        assert self.end_of_selection == unpack('B', data[:1])[0]
        data = data[1:]
        (successive_approximation,) = unpack('B', data[:1])
        data = data[1:]
        assert self.successive_approximation_high == successive_approximation >> 4
        assert self.successive_approximation_low == successive_approximation & 0x0F

        assert len_segment == 2 + 1 + num_components * 2 + 3, 'Error - SOS invalid\n'
        # NOTE the length only covers up to here -> the length of the Huffman bitstream is not recorded

        return len_segment

    def readScan(self, data):
        """There is only one scan for baseline JPEG.
        There could be additional scans for progressive JPEG, which we do not support now.
        """
        # decode the first scan
        len_sos = self.readStartOfScan(data)
        # length is not explicitly recorded for SOS segment -> we have to count the bytes as we decode Huffman
        len_scan, huffman_decoded_seq_stream, huffman_decoded_seq_zero_recovered_stream = \
            self.decode_huffman_bitstream(data[len_sos:])

        len_segment = len_sos + len_scan

        self.huffman_decoded_seq_stream = huffman_decoded_seq_stream
        self.huffman_decoded_seq_zero_recovered_stream = huffman_decoded_seq_zero_recovered_stream

        # decode additional scans, if any (TODO for progressive JPEG)
        pass

        return len_segment

    @staticmethod
    def _clean_scan_bitstream(data):
        """Do a pass over the bitstream data inside the scan to clean up the bitstream, to prepare for Huffman decoding.
        The `data` should start with the bitstream inside Scan section but after the SOS header
        (only starting the entropy-encoded bits).

        The cleaning includes:
        - Remove 0x00 after 0xFF in the scan bitstream, so that 0xFF is included in the entropy-coded data
          (byte stuffing, https://docs.fileformat.com/image/jpeg/)
        - Remove 0xFF after 0xFF, as consecutive 0xFF can exist
        - Remove RST0 - RST7 markers (restart interval markers)
        - Stop at the other 0xFF markers; should be EOI (0xFFD9) for baseline JPEG.
        """
        data_huffman = []
        i = 0
        while True:
            b, bnext = unpack('BB', data[i:i + 2])
            if b == 0xFF:
                if bnext == 0xFF:
                    # consecutive 0xFF -> skip to only keep one
                    i += 1
                elif bnext == 0x00:
                    # 0xFF00 -> byte stuffing, remove 0x00 and keep 0xFF in the entropy-coded bitstream
                    data_huffman.append(data[i])
                    i += 2
                elif RST0 <= bnext <= RST7:
                    # restart interval markers -> skip (restart interval can be decided with the DRI segment
                    # from the header)
                    i += 2
                elif bnext == 0xD9:
                    # EOS marker -> end the bitstream
                    break
                else:
                    raise ValueError(f'Error - invalid marker {data[i:i+2]} encountered in the Scan section '
                                     'for baseline JPEG (other types, e.g. progressive JPEG is unsupported now)/n')
            else:
                data_huffman.append(data[i])
                i += 1

        return data_huffman, i

    @staticmethod
    def decode_huffman_block_component(bitstream, huffman_DC_table, huffman_AC_table):
        """Huffman decoding of a block component - one MCU inside a color component.
        Reminder: the bitstream is ordered as
        MCU0_channel1 MCU0_channel2 MCU0_channel3 | MCU0_channel1 MCU1_channel2 MCU1_channel3 | ...

        Bytes are not aligned between block components.
        They are aligned before RST0-7 or before EOI, by appending 0s.

        Return a sequence of values (representing quantization coefficients):
        DC(delta) (zero-run length, AC) (zero-run length, AC) ... (0, 0)(end of block, this is a unique marker)
        """
        huffman_decoded_seq = []
        huffman_decoded_seq_zero_recovered = []

        # ===== get the DC value for this block component
        symbol = huffman_DC_table.get_symbol(bitstream)    # NOTE this already moved past the code bits
        # each symbol is 1 byte
        value_bits_len = symbol
        # for DC value, the upper nibble is always 0: b0000xxxx
        # and the value is no larger than 11
        if value_bits_len > 11:
            raise ValueError('Error - DC coefficient bits length greater than 11\n')

        # get the value
        value_bits = bitstream.get_bit_n(value_bits_len)
        dc_coeff = DecodeNumber(value_bits_len, value_bits)

        huffman_decoded_seq.append(dc_coeff)
        huffman_decoded_seq_zero_recovered.append(dc_coeff)

        # ===== get the AC values for this block component
        num_coeff = 1
        while num_coeff < 64:
            symbol = huffman_AC_table.get_symbol(bitstream)    # NOTE this already moved past the code bits
            # each symbol is 1 byte

            # symbol 0x00 means fill remainder of component with 0
            if symbol == 0x00:
                huffman_decoded_seq += [0, 0]
                huffman_decoded_seq_zero_recovered += [0] * (64 - num_coeff)
                num_coeff = 64
                continue    # equivlent to breaking the loop

            # otherwise, read next component coefficient
            upper_nibble, lower_nibble = symbol >> 4, symbol & 0x0F
            num_zeros = upper_nibble
            value_bits_len = lower_nibble    # maximum is 10. Could be 0: 0xF0 means 16 zeros

            if num_coeff + num_zeros >= 64:
                raise ValueError('Error - Zero run-length exceeded block component\n')
            num_coeff += num_zeros

            if value_bits_len > 10:
                raise ValueError('Error - AC coefficient length greater than 10\n')

            # get the value
            value_bits = bitstream.get_bit_n(value_bits_len)
            ac_coeff = DecodeNumber(value_bits_len, value_bits)

            # if num_zeros == 0xF:
            #     print(value_bits_len)    # this would be 0: no next bits need to be read for the value 0
            #     print(ac_coeff)    # this would be 0.0
            #     breakpoint()

            num_coeff += 1

            huffman_decoded_seq += [num_zeros, ac_coeff]    # NOTE `ac_coeff` is of type float when it is 0
            huffman_decoded_seq_zero_recovered = huffman_decoded_seq_zero_recovered + [0] * num_zeros + [ac_coeff]

        return huffman_decoded_seq, huffman_decoded_seq_zero_recovered

    def decode_huffman_bitstream(self, data):
        """Huffman decoding of the entropy-encoded bitstream.
        Reminder: the bitstream is ordered as
        MCU0_channel1 MCU0_channel2 MCU0_channel3 | MCU0_channel1 MCU1_channel2 MCU1_channel3 | ...
        """
        # run a pass over the bitstream, do some cleaning (e.g. remove 0x00 after 0xFF),
        # get the length of the Scan section corresponding to Huffman-encoded data, right before EOI (0xFFD9)
        # (NOTE for baseline JPEG only)
        data_huffman, len_segment = self._clean_scan_bitstream(data)

        bitstream = BitStream(data_huffman)
        huffman_decoded_seq_stream = []
        huffman_decoded_seq_zero_recovered_stream = []
        # decode the bitstream, one block (all color components) at a time
        for y in range(self.block_height):
            for x in range(self.block_width):
                for component_id, component in self.color_components.items():
                    # NOTE we have already guaranteed that the color components in SOS are exactly matching those in SOF
                    huffman_decoded_seq, huffman_decoded_seq_zero_recovered = self.decode_huffman_block_component(
                        bitstream,
                        huffman_DC_table=self.huffman_DC_tables[component.huffman_DC_table_id],
                        huffman_AC_table=self.huffman_AC_tables[component.huffman_AC_table_id],
                        )

                    huffman_decoded_seq_stream += huffman_decoded_seq
                    huffman_decoded_seq_zero_recovered_stream += huffman_decoded_seq_zero_recovered

        return len_segment, huffman_decoded_seq_stream, huffman_decoded_seq_zero_recovered_stream

    def decode(self):
        data = self.img_data
        while True:
            # get the segment marker
            assert data[0:1] == b'\xFF', 'Error - marker must start with the byte 0xFF\n'
            (marker,) = unpack(">H", data[0:2])
            # convert marker back to bytes:
            # from struct import pack; marker_bytes = pack('>H', marker)  # this would equal to data[:2]
            logger.info(f'{marker:#X}' + ' - ' + marker_mapping.get(marker, 'unknown'))
            # hex(marker) would equal to f'{marker:#x}'

            # move past the marker
            data = data[2:]

            if marker == 0xFFD8:
                # first two bytes of the bitstream must be 0xFFD8, SOI
                # no length bytes
                pass

            elif marker == 0xFFD9:
                # last two bytes of the bitstream must be 0xFFD9, EOI
                # no length bytes
                break

            else:
                if marker == 0xFFDB:
                    # quantization table
                    len_segment = self.readQuantizationTable(data)

                elif marker == 0xFFDD:
                    # restart interval
                    len_segment = self.readRestartInterval(data)

                elif marker == 0xFFC0:
                    # Start of Frame. NOTE we only support the baseline JPEG compression now
                    len_segment = self.readBaselineStartOfFrame(data)

                elif marker == 0xFFC4:
                    # Huffman table
                    len_segment = self.readHuffmanTable(data)

                elif marker == 0xFFDA:
                    # Start of Scan
                    # NOTE for baseline JPEG, there is only one scan section containing all MCUs
                    # no length bytes for the Huffman bitstream
                    len_segment = self.readScan(data)
                    # two parts: read Start of Scan + the Scan itself (doing Huffman decoding here)

                else:
                    # other marker segments that we simply skip now
                    len_segment = self.readUnknownSegment(data)

                # move to the next segment
                data = data[len_segment:]

            if len(data) == 0:
                assert marker == 0xFFD9, 'Error - JPEG bytes not ending with EOI marker\n'
                break

        self.decoded = True

        return self.width, self.height, self.image

    def get_decoded_huffman_stream(self):
        if not self.decoded:
            _ = self.decode()
        return self.huffman_decoded_seq_stream

    def get_decoded_huffman_zero_recovered_stream(self):
        if not self.decoded:
            _ = self.decode()
        return self.huffman_decoded_seq_zero_recovered_stream


if __name__ == "__main__":
    # # ===== start from a jpeg file saved in storage
    # from pathlib import Path

    # jpeg_path = Path(__file__).resolve().parent.parent / 'jpegs' / 'kaori.jpg'
    # img = JPEG(jpeg_path, verbose=True, verbose_details=True)
    # width, height, image = img.decode()

    # breakpoint()

    # # show image
    # from PIL import Image
    # img = Image.new("RGB", (width, height))
    # img.putdata(image)
    # # img.show()

    # breakpoint()

    # ===== start from an RGB image stored in an numpy array
    from pathlib import Path

    from jpeg_baseline_decoder.check_diff_jpeg import (jpg_bytes_from_file,
                                                       load_jpg2img,
                                                       compress_jpg, compress_jpg_torch, compress_jpg_simplejpeg)

    jpeg_path = Path(__file__).resolve().parent.parent / 'jpegs' / 'kaori.jpg'
    flag = 1

    # jpeg_path = Path(__file__).resolve().parent.parent / 'jpegs' / 'pyramid.jpg'
    # flag = 1

    jpeg_path = Path(__file__).resolve().parent.parent / 'jpegs' / 'ILSVRC2012_val_00006216.JPEG'
    flag = 1

    # jpeg_path = Path(__file__).resolve().parent.parent / 'jpegs' / '8-bit-256-x-256-Grayscale-Lena-Image_W640.jpg'
    # flag = 0

    bytes_file = jpg_bytes_from_file(jpeg_path)

    img = load_jpg2img(jpeg_path, flag)
    # img is np.ndarray, of size (nrows, ncols, 3) if flag else (nrows, ncols)

    # encode to jpeg
    quality = 50

    bytes_cv2 = compress_jpg(img, quality)
    bytes_tv = compress_jpg_torch(img, quality)
    bytes_sim = compress_jpg_simplejpeg(img, quality, colorsubsampling='444')

    jpg_bytes = bytes_file    # use original jpg code directly read from file
    # jpg_bytes = bytes_cv2    # use cv2 compressed jpg code
    # jpg_bytes = bytes_tv    # use torchvision compressed jpg code
    # jpg_bytes = bytes_sim    # use simplejpeg compressed jpg code

    jpg = JPEG(jpeg_bytes=jpg_bytes, verbose=True, verbose_details=True)
    width, height, image = jpg.decode()

    decoded_huffman_stream = jpg.get_decoded_huffman_stream()
    decoded_huffman_zero_recovered_stream = jpg.get_decoded_huffman_zero_recovered_stream()
    print(f'length of decoded huffman stream: {len(decoded_huffman_stream)}')
    print(f'length of decoded huffman stream with zeros recovered: {len(decoded_huffman_zero_recovered_stream)}')
    breakpoint()

    # show image
    from PIL import Image
    img = Image.new("RGB", (width, height))
    img.putdata(image)
    # img.show()

    # breakpoint()
