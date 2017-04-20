#include "feature_writer.h"

#include <sstream>
#include <stdio.h>
#include <memory.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

FileMatrixOutput::FileMatrixOutput(const std::string &filepath, bool binary) {
  Open(filepath, binary);
  if (os_.precision() < 7) {
	  os_.precision(7);
  }
}

FileMatrixOutput::~FileMatrixOutput() {
  if (os_.is_open()) {
    os_.close();
    if (os_.fail()) {
      fprintf(stderr, "~FileMatrixOutput, close file %s fail",
        filepath_.c_str());
    }
  }
}

bool FileMatrixOutput::Open(const std::string &filepath, bool binary) {
  if (os_.is_open()) {
    fprintf(stderr, "FileMatrixOutput::Open, file %s already open",
      filepath.c_str());
    return false;
  }
  binary_ = binary;
  filepath_ = filepath;
  os_.open(filepath.c_str(),
    binary ? std::ios_base::out | std::ios_base::binary
           : std::ios_base::out);
  return os_.is_open();
}

bool FileMatrixOutput::Close() {
  if (!os_.is_open()) {
    fprintf(stderr, "FileMatrixOutput::Close, file %s is not open",
      filepath_.c_str());
    return false;
  }
  os_.close();
  return !(os_.fail());
}

bool FileMatrixOutput::Write(const float *data, int num_row, int num_col) {
  if (!os_.is_open()) {
    fprintf(stderr, "FileMatrixOutput::Write, file %s is not open",
      filepath_.c_str());
    return false;
  }
  if (binary_) {
	// binary header
    os_.put('\0');
    os_.put('B');
	// write token, specifying float matrix
	os_ << "FM" << ' ';
	// size info
    char lenc = static_cast<char>(sizeof(int));
    os_.put(lenc);
    os_.write(reinterpret_cast<const char *>(&num_row), sizeof(int));
    os_.put(lenc);
    os_.write(reinterpret_cast<const char *>(&num_col), sizeof(int));
	// data
    os_.write(reinterpret_cast<const char *>(data), sizeof(float)
      * num_row * num_col);
  } else {
    if (num_col == 0) {
      os_ << " [ ]\n";
    } else {
      os_ << " [";
      for (int i = 0; i < num_row; i++) {
        os_ << "\n ";
        int cur_base = num_col * i;
        for (int j = 0; j < num_col; j++)
          os_ << data[j+cur_base] << " ";
      }
      os_ << "]\n";
    }
  }
  return os_.good();
}

FeatureWriter::FeatureWriter(const std::string &arkpath,
 const std::string &scppath, bool binary) :
 ark_output_(arkpath, binary),
 scp_output_(scppath, false),
 ark_filepath_(arkpath),
 scp_filepath_(scppath) { }

void FeatureWriter::WriteSCP(const std::string &uttrid) {
  // offset_rxfilename
  std::ostream::pos_type ark_os_pos = ark_output_.ofstream().tellp();
  std::ostringstream ss;
  ss << ':' << ark_os_pos;
  std::string offset_rxfilename = ark_filepath_ + ss.str();
  // write
  scp_output_.ofstream() << uttrid << ' ' << offset_rxfilename << '\n';
}

void FeatureWriter::WriteFeature(float *feature, int frame_num, int num_cols,
 const std::string &uttrid) {
  ark_output_.ofstream() << uttrid << ' ';
  WriteSCP(uttrid);
  ark_output_.Write(feature, frame_num, num_cols);
}
