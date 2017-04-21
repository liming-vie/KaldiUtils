#include "nnet_reader.h"

#include <stdlib.h>
#include <iostream>
#include <locale>
#include <algorithm>
#include <limits>

DNNPara::DNNPara() {
    for (int i=0; i<MAXLAYER; ++i) {
        weights[i] = NULL;
        bias[i] = NULL;
        layer_sizes[i] = 0;
    }
    num_layers = 0;
}

DNNPara::~DNNPara() {
    Uninit();
}

#define UNINIT_POINTER(variable, i) \
    if (NULL != variable[i]) {      \
        delete []variable[i];       \
        variable[i] = NULL;         \
    }
void DNNPara::Uninit() {
    for (int i=0; i<num_layers; ++i) {
        UNINIT_POINTER(weights, i);
        UNINIT_POINTER(bias, i);
        layer_sizes[i] = 0;
    }
    num_layers = 0;
}

FileInput::FileInput(const std::string &filepath) {
    filepath_ = filepath;

    is_.open(filepath.c_str(), std::ios_base::in
            | std::ios_base::binary); // always open in binary
    if (!is_.is_open()) {
        fprintf(stderr, "FileInput, open file %s fail\n", filepath.c_str());
        exit(1);
    }
    // check is content binary or text
    if (is_.peek() == '\0') {
        is_.get();
        is_.get();   // no error check for 'B'
        binary_ = true;
    } else {
        binary_ = false;
    }
}

FileInput::~FileInput() {
    if (is_.is_open()) {
        is_.close();
    }
}

int FileInput::Peek() {
    if (!binary_) is_ >> std::ws; // eat up whitespace
    return is_.peek();
}

float FileInput::ReadFloat() {
    float f;
    if (binary_) {
        double d;
        int c = is_.peek();
        if (c == sizeof(f)) {
            is_.get();
            is_.read(reinterpret_cast<char*>(&f), sizeof(f));
        } else if (c == sizeof(d)) {
            is_.get();
            is_.read(reinterpret_cast<char*>(&d), sizeof(d));
            f = d;
        } else {
            fprintf(stderr, "FileInput::ReadFloat, unkown type\n");
            exit(1);
        }
    } else {
        is_ >> f;
    }
    return f;
}

int FileInput::ReadInt32() {
    int i;
    if (binary_) {
        int len_c_in = is_.get();
        char len_c = static_cast<char>(len_c_in);
        char len_c_expected = static_cast<char>(sizeof(i));
        if (len_c != len_c_expected) {
            fprintf(stderr, "FileInput::ReadInt32, unkown type\n");
            exit(1);
        }
        is_.read(reinterpret_cast<char *>(&i), sizeof(i));
    } else {
        is_ >> i;
    }
    return i;
}

#define CHECK_VECTOR_SIZE(s1, s2)   \
    if (s1 != s2) {                 \
        fprintf(stderr, "FileInput::ReadVector, size unmatch\n");\
        exit(1);                    \
    }

void FileInput::ReadVector(float* data, int size) {
    if (binary_) {
        int peekval = Peek();
        if (peekval == 'D') {
            fprintf(stderr, "FileInput::ReadVector, unexpected token start D\n");
            exit(1);
        }
        ExpectToken("FV");  // float vector
        int tsize = ReadInt32();
        CHECK_VECTOR_SIZE(tsize, size);
        if (size > 0) {
            is_.read(reinterpret_cast<char*>(data), sizeof(float)*size);
        }
    } else { // text mode
        std::string str;
        is_ >> str;
        if (str == "[]") {
            CHECK_VECTOR_SIZE(0, size);
            return;
        }
        int idx = 0;
        while(1) {
            int i = is_.peek();
            if (i == '-' || (i>='0' && i<='9')) { // number
                is_ >> data[idx++];
            } else if (i == ' ' || i == '\t') {
                is_.get();
            } else if (i == ']') {  // finish
                is_.get();
                i = is_.peek();
                if (static_cast<char>(i) == '\r') { // get \r\n
                    is_.get();
                    is_.get();
                } else if(static_cast<char>(i) == '\n') {
                    is_.get();
                }
                break;
            } else { // inf or Nan
                is_ >> str;
                std::transform(str.begin(), str.end(), str.begin(), ::tolower);
                if (str == "inf" || str == "infinity") {
                    data[idx++] = std::numeric_limits<float>::infinity();
                } else if (str == "nan") {
                    data[idx++] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        CHECK_VECTOR_SIZE(size, idx);
    }
}

#define CHECK_ROWS_AND_COLS(r1, c1, r2, c2) \
    if (r1 != r2 || c1 != c2) {             \
        fprintf(stderr, "FileInput::ReadMatrix, row and cols unmatch\n"); \
        exit(1);                            \
    }
void FileInput::ReadMatrix(float* data, int num_rows, int num_cols) {
    int rows, cols;
    if (binary_) {
        int peekval = Peek();
        if (peekval == 'C' || peekval == 'D') {
            fprintf(stderr, "FileInput::ReadMatrix, unexpected token start %c\n",
                    peekval);
            exit(1);
        }
        ExpectToken("FM"); // float matrix
        rows = ReadInt32();
        cols = ReadInt32();
        CHECK_ROWS_AND_COLS(rows, cols, num_rows, num_cols)
        is_.read(reinterpret_cast<char*>(data), sizeof(float)*rows*cols);
    } else { // text mode
        std::string str;
        is_ >> str; // get a token
        if (str == "[]") {
            CHECK_ROWS_AND_COLS(0, 0, num_rows, num_cols)
            return;
        }
        int idx = 0;
        while (1) {
            int i = is_.peek();
            if (static_cast<char>(i) == ']') { // finish
                is_.get(); // eat ']'
                i = is_.peek();
                if (static_cast<char>(i) == '\r') { // get \r\n
                    is_.get();
                    is_.get();
                } else if (static_cast<char>(i) == '\n') { // get \n
                    is_.get();
                }
                break;
            } else if (static_cast<char>(i) == '\n'
                    || static_cast<char>(i) == ';') { // end of matrix row
                is_.get();
                ++cols;
            } else if (isspace(i) ) {
                is_.get();
            } else if ((i>='0' && i<='9') || i=='-') { // number
                is_ >> data[idx++];
            } else { // NaN or inf or error
                is_ >> str;
                std::transform(str.begin(), str.end(), str.begin(), ::tolower);
                if (str == "inf" || str == "infinity") {
                    data[idx++] = std::numeric_limits<float>::infinity();
                } else if (str == "nan") {
                    data[idx++] = std::numeric_limits<float>::quiet_NaN();
                }
            }
        }
        rows = idx / cols;
        CHECK_ROWS_AND_COLS(rows, cols, num_rows, num_cols)
    }
}

void FileInput::ReadToken(std::string &token) {
    if (!binary_) {
        is_ >> std::ws;  // consume whitespace.
    }
    is_ >> token;
    fprintf(stdout, "%s\n", token.c_str());
    if (!isspace(is_.peek())) {
        fprintf(stderr, "FileInput::ReadToken, expect space after token\n");
        exit(1);
    }
    is_.get(); // consume the space
}

int FileInput::PeekToken() {
    if (!binary_) {
        is_ >> std::ws;  // consume whitespace.
    }
    bool read_bracket;
    if (static_cast<char>(is_.peek()) == '<') {
        read_bracket = true;
        is_.get();
    } else {
        read_bracket = false;
    }
    int ans = is_.peek();
    if (read_bracket) {
        if (!is_.unget()) {
            fprintf(stderr, "FileInput::PeekToken, error ungetting '<'\n");
            exit(1);
        }
    }
    return ans;
}

void FileInput::ExpectToken(std::string token) {
    std::string str;
    ReadToken(str);
    if (token != str) {
        fprintf(stderr, "FileInput::ExpectToken, expect %s, found %s\n",
            token.c_str(), str.c_str());
        exit(1);
    }
}

NNetReader::NNetReader(DNNPara &dnn_para) : dnn_para_(dnn_para) {
}

void NNetReader::Read(const std::string& model_file) {
    FileInput fin(model_file);
    dnn_para_.Uninit();

    int first_char;
    int dim_out, dim_in;
    std::string token;
    // read components one by one
    while (true) {
        first_char = fin.Peek();
        // end of file
        if (first_char == EOF)  {
            break;
        }
        // get component token
        fin.ReadToken(token);
        // skip the optional initial token
        if (token == "<Nnet>") {
            fin.ReadToken(token);
        }
        // network ends after terminal token appears
        if (token == "</Nnet>") {
            break;
        }
        // read the dims
        dim_out = fin.ReadInt32();
        dim_in = fin.ReadInt32();
        // read the component
        ReadOneLayer(token, fin, dim_in, dim_out);
        // 'eat' the component separtor (can be already consumed')
        if ('<' == fin.Peek() && '!' == fin.PeekToken()) {
            fin.ExpectToken("<!EndOfComponent>");
        }
    }
}

void NNetReader::ReadOneLayer(std::string &token, FileInput &fin,
        int dim_in, int dim_out) {
    std::transform(token.begin(), token.end(), token.begin(), ::tolower);
    if (token == "<softmax>") { // output layer
        ++dnn_para_.num_layers_;
        return;
    } else if(token == "<sigmoid>") { // do nothing
        return;
    } else if (token != "<affinetransform>") {
        fprintf(stderr, "NNetReader::ReadOneLayer, Unexpected token %s\n",
                token.c_str());
        exit(1);
    }
    // read affine transform
    while ('<' == fin.Peek()) {
        int first_char = fin.PeekToken();
        fin.ReadToken(token);
        switch (first_char) {
            case 'L': // learn rate coef
            case 'B': // bias learn rate coef
            case 'M': // max norm
                fin.ReadFloat();
                break;
            default:
                fprintf(stderr, "NNetReader::ReadOneLayer, unkown token %s\n",
                    token.c_str());
        }
    }
    int vi=dnn_para_.num_layers++;
    // save layer size info
    if (vi > 0 && dnn_para_.layer_sizes[vi] != dim_in) {
        fprintf(stderr, "NNetReader::ReadOneLayer, layer dims unmatch\n");
        exit(1);
    }
    dnn_para_.layer_sizes[vi] = dim_in; // no error check
    dnn_para_.layer_sizes[vi+1] = dim_out;
    // read weight matrix
    dnn_para_.weights[vi] = new float[dim_out * dim_in];
    fin.ReadMatrix(dnn_para_.weights[vi], dim_out, dim_in);
    // read bias vector
    dnn_para_.bias[vi] = new float[dim_out];
    fin.ReadVector(dnn_para_.bias[vi], dim_out);
}
