find_package(PkgConfig REQUIRED)
pkg_check_modules(ZMQ libzmq)

if(ZMQ_FOUND)
  add_definitions(-DWITH_ZMQ)
  include_directories(${ZMQ_INCLUDE_DIRS})
endif()

build_exec(
  EXECNAME gnb_scheduler
  SOURCE_FILES gnb_scheduler.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libopengym}
    ${ZMQ_LIBRARIES}
  EXECUTABLE_DIRECTORY_PATH ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/scratch/gnb_scheduler
)
