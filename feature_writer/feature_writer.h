#ifndef KALDI_FEATURE_WRITER_H__
#define KALDI_FEATURE_WRITER_H__

#include <fstream>
#include <string>

class FileMatrixOutput {
 public:
  FileMatrixOutput(const std::string &filepath, bool binary);
  ~FileMatrixOutput();

  bool Open(const std::string &filepath, bool binary);
  bool Close();
  bool Write(const float *data, int num_row, int num_col);

  std::ofstream &ofstream() { return os_; }
  const std::string &filepath() {return filepath_;}

 private:
  bool binary_;
  std::string filepath_;
  std::ofstream os_;
}; // class FileMatrixOutput

class FeatureWriter {
 public:
  FeatureWriter(const std::string &arkpath, const std::string &scppath,
    bool binary); // is write archive file binary
  ~FeatureWriter() {}

  void WriteFeature(float *feature, int frame_num, int num_cols,
         const std::string &uttrid);

 private:
  void WriteSCP(const std::string &uttrid);

  FileMatrixOutput ark_output_;
  FileMatrixOutput scp_output_;
  const std::string ark_filepath_;
  const std::string scp_filepath_;
}; // class FeatureWriter

#endif  // KALDI_FEATURE_WRITER_H__
