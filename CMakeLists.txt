project(main)

find_package(OpenCV REQUIRED)
add_executable(main
    main.cpp
    mosaic.h mosaic.cpp
)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
