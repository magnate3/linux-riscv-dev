# Copyright (c) 2018 Cisco and/or its affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include_directories(${CMAKE_SOURCE_DIR})

# for generated API headers:
include_directories(${CMAKE_BINARY_DIR})

add_vpp_plugin(my
  SOURCES
  node.c
  my.c

  MULTIARCH_SOURCES
  node.c

  API_FILES
  my.api


  COMPONENT vpp-plugin-my
)
#add_vpp_plugin(my
#  SOURCES
#  node.c
#  my.c
#
#  MULTIARCH_SOURCES
#  node.c
#
#  API_FILES
#  my.api
#
#  API_TEST_SOURCES
#  my_test.c
#
#  COMPONENT vpp-plugin-my
#)
