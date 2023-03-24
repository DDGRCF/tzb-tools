#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <stdint.h>
#include <bitset>

using std::string;
using std::vector;
using std::cout;
using std::endl;

int main(int argc, char** argv) {
  string img_file_path = "/disk0/dataset/TianzhiBk/LargeDataset/base_images/入间机场.tif";
  std::ifstream img_file(img_file_path, std::ios::in | std::ios::binary);

  u_int64_t val;
  img_file.read(reinterpret_cast<char*>(&val), 8);
  // cout << std::bitset<sizeof(val) * 8>(val) << endl; 
  // cout << std::bitset<sizeof(val) * 8>(val) << endl; 
  cout << std::hex << val << endl;
  return 0;
}

