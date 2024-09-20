all:
    # Install pycocotools locally
	python setup.py build_ext --inplace
	rm -rf build

interactive:
    # Install pycocotools as an interactive framework
	pip install -e .
	rm -rf build

install:
    # Install pycocotools to the Python site-packages
	python -m pip install .
	rm -rf build
