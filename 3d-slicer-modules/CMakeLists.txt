cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(AnnotationSupervisor)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/AnnotationSupervisor")
set(EXTENSION_CATEGORY "Segmentation")
set(EXTENSION_CONTRIBUTORS "Antoine P. Sanner (University Hospital Mayence, Germany)")
set(EXTENSION_DESCRIPTION "An extension to have an overview of study cohorts and images to annotate.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/AnnotationSupervisor.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/AnnotationSupervisor/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(AnnotationOverview)
add_subdirectory(test)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
