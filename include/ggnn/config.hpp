/* Copyright 2019 ComputerGraphics Tuebingen. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
// Authors: Fabian Groh, Patrick Wieschollek, Hendrik P.A. Lensch

#ifndef CONFIG_HPP_
#define CONFIG_HPP_

#include <stdarg.h>
#include <stdio.h>
#include <sys/stat.h>

#include <fstream>
#include <string>

#include "glog/logging.h"

inline bool exists(const std::string& name) {
  struct stat buffer;
  return (stat(name.c_str(), &buffer) == 0);
}

inline void lprintf(const int log_level, const char* format, ...) {
  if (VLOG_IS_ON(log_level)) {
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
  }
}

#endif  // CONFIG_HPP_
