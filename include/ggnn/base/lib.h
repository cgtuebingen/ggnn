/* Copyright 2025 ComputerGraphics Tuebingen. All Rights Reserved.

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
// Authors: Fabian Groh, Lukas Ruppert, Patrick Wieschollek, Hendrik P.A. Lensch
// converted to GGNN library by: Lukas Ruppert, Deborah Kornwolf

#ifndef INCLUDE_GGNN_LIB_H
#define INCLUDE_GGNN_LIB_H

// list of types used to instantiate templates - extend as needed

#define GGNN_VALUES(F, ...) \
  F(__VA_OPT__(__VA_ARGS__, ) float);

#define GGNN_BASES(F, ...)              \
  F(__VA_OPT__(__VA_ARGS__, ) uint8_t); \
  F(__VA_OPT__(__VA_ARGS__, ) float);

#define GGNN_KEYS(F, ...) \
  F(__VA_OPT__(__VA_ARGS__, ) int32_t);

#define GGNN_QUERYS(F, ...)         \
  F(__VA_OPT__(__VA_ARGS__, ) 32);  \
  F(__VA_OPT__(__VA_ARGS__, ) 64);  \
  F(__VA_OPT__(__VA_ARGS__, ) 128); \
  F(__VA_OPT__(__VA_ARGS__, ) 256); \
  F(__VA_OPT__(__VA_ARGS__, ) 512); \
  F(__VA_OPT__(__VA_ARGS__, ) 1024);

#define GGNN_TOPS(F, ...)              \
  F(__VA_OPT__(__VA_ARGS__, ) 128, 4); \
  F(__VA_OPT__(__VA_ARGS__, ) 256, 4); \
  F(__VA_OPT__(__VA_ARGS__, ) 256, 8); \
  F(__VA_OPT__(__VA_ARGS__, ) 512, 8);

#define GGNN_MERGES(F, ...)            \
  F(__VA_OPT__(__VA_ARGS__, ) 32, 4);  \
  F(__VA_OPT__(__VA_ARGS__, ) 64, 4);  \
  F(__VA_OPT__(__VA_ARGS__, ) 128, 4); \
  F(__VA_OPT__(__VA_ARGS__, ) 256, 4); \
  F(__VA_OPT__(__VA_ARGS__, ) 256, 8); \
  F(__VA_OPT__(__VA_ARGS__, ) 512, 8);

#define GGNN_SYMS(F, ...)              \
  F(__VA_OPT__(__VA_ARGS__, ) 64, 4);  \
  F(__VA_OPT__(__VA_ARGS__, ) 128, 4); \
  F(__VA_OPT__(__VA_ARGS__, ) 256, 4); \
  F(__VA_OPT__(__VA_ARGS__, ) 256, 8); \
  F(__VA_OPT__(__VA_ARGS__, ) 512, 8);

#define GGNN_DIST_STATS(F, ...) \
  F(__VA_OPT__(__VA_ARGS__, ) false);

#define GGNN_WRITE_DISTS(F, ...) \
  F(__VA_OPT__(__VA_ARGS__, ) true);

#define GGNN_EVAL(F, ...) \
  F(__VA_ARGS__);

#define GGNN_INSTANTIATE_STRUCT(T, ...) \
  template struct T<__VA_ARGS__>;

#define GGNN_INSTANTIATE_CLASS(T, ...) \
  template class T<__VA_ARGS__>;

#endif  // INCLUDE_GGNN_LIB_H
