#ifndef WAKEUP_NNET_READER_H_
#define WAKEUP_NNET_READER_H_

#include <fstream>
#include <string>

#define MAXLAYER 10

struct DNNPara {
    float *weights[MAXLAYER];       // weights for layers
    float *bias[MAXLAYER];          // biases for layers

    int num_layers;
    // [i] is the input dim of i-th layer
    // and output dim of (i-1)-th layer
    int layer_sizes[MAXLAYER+1];

    DNNPara();
    ~DNNPara();
    void Uninit();
};  // struct DNNPara

class FileInput {
  public:
    FileInput(const std::string &filepath);
    ~FileInput();

    int Peek();

    int ReadInt32();
    float ReadFloat();

    void ReadVector(float* data, int size);
    // data should be of size (num_rows * num_cols)
    void ReadMatrix(float* data, int num_rows, int num_cols);

    void ReadToken(std::string &token);
    int PeekToken();
    void ExpectToken(std::string token);

    std::ifstream &Stream() { return is_; }
    const std::string &filepath() { return filepath_; }

  private:
    bool binary_; // specify the content binary
    std::string filepath_;
    std::ifstream is_;
};  // class FileInput

// currently only support Sigmoid, Softmax and AffineTransform Components
class NNetReader {
  public:
    NNetReader(DNNPara &dnn_para);

    void Read(const std::string& model_file);

  private:
    void ReadOneLayer(std::string &token, FileInput &fin,
            int dim_in, int dim_out);

  private:
    DNNPara &dnn_para_;
};  // class NNetReader

#endif  // WAKEUP_NNET_READER_H_
