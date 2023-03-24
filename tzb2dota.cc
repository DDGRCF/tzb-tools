#include <dirent.h>
#include <errno.h>
#include <glob.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <list>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

inline vector<string> split(const string& line, const char& delim = ' ') {
  stringstream ss(line);
  vector<string> res;
  string word;
  while (getline(ss, word, delim)) {
    res.push_back(word);
  }
  return res;
}

inline string path_basename(const string& path) {
  auto left = path.rfind('/');
  if (left == string::npos) {
    return path;
  }
  return path.substr(left + 1);
}

inline bool is_exist(const string& path) {
  struct stat statbuf;
  int ret = stat(path.c_str(), &statbuf);
  if (ret == -1) {
    return false;
  }
  return true;
}

inline bool is_dir(const string& path) {
  if (!is_exist(path)) {
    return false;
  }
  struct stat statbuf;
  stat(path.c_str(), &statbuf);
  if (!S_ISDIR(statbuf.st_mode)) {
    return false;
  }
  return true;
}

inline bool is_file(const string& path) {
  if (!is_exist(path)) {
    return false;
  }
  struct stat statbuf;
  stat(path.c_str(), &statbuf);
  if (!S_ISREG(statbuf.st_mode)) {
    return false;
  }
  return true;
}

inline string path_dirname(const string& path) {
  auto left = path.rfind('/');
  if (left == std::string::npos) {
    return path;
  }
  return path.substr(0, left);
}

inline vector<string> path_glob(const string& path) {
  std::vector<std::string> res{};
  const std::string& dirname = path_dirname(path);
  if (!is_exist(dirname)) {
    return res;
  }
  int ret;
  glob_t globbuf{0};
  do {
    ret = ::glob(path.c_str(), GLOB_NOSORT, nullptr, &globbuf);
    if (ret != 0) {
      break;
    }
    res.reserve(globbuf.gl_pathc);
    for (size_t i = 0; i < globbuf.gl_pathc; i++) {
      res.emplace_back(globbuf.gl_pathv[i]);
    }
  } while (0);
  globfree(&globbuf);
  return res;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    cerr << "expect 3 but get " << argc << endl;
    exit(1);
  }
  const string input_dir = argv[1];
  const string output_dir = argv[2];
  int ret = mkdir(output_dir.c_str(), 0744);
  if (ret == -1) {
    cerr << "mkdir " << output_dir << " " << strerror(errno) << endl;
    exit(1);
  }

  auto&& ori_ann_set = path_glob(input_dir + "/*.txt");
  cout << "get ann " << ori_ann_set.size() << endl;
  for (auto& ori_ann_path : ori_ann_set) {
    ifstream ori_ann_file(ori_ann_path);
    const string& basename = path_basename(ori_ann_path);
    const string& dst_ann_path = output_dir + "/" + basename;
    ofstream dst_ann_file(dst_ann_path);

    string line;
    list<string> lines;
    while (getline(ori_ann_file, line)) {
      auto left = line.rfind('\r');
      if (left != string::npos) {
        line = line.substr(0, left);
      }
      if (line.empty()) {
        continue;
      }
      auto line_split = split(line);
      vector<string> line_res;
      line_res.reserve(line_split.size() + 1);
      for (int i = 1; i < 9; i++) {
        line_res.push_back(line_split[i]);
      }
      line_res.push_back(line_split[0]);
      line_res.push_back(to_string(0));
      const string line_to =
          accumulate(line_res.begin(), line_res.end(), string(""),
                     [](string& lhs, const string& rhs) {
                       return lhs.empty() ? rhs : lhs + " " + rhs;
                     });
      lines.push_back(line_to);
    }
    int i = 0;
    for (auto& line : lines) {
      dst_ann_file << line;
      if (i < lines.size() - 1) {
        dst_ann_file << endl;
      }
      i++;
    }
  }
  return 0;
}