add_subdirectory(src)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
#target_link_libraries( ${OpenCV_LIBS} )