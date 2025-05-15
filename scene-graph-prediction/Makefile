all:
    # Install scene_graph_prediction locally
	python setup.py build_ext --inplace
	rm -rf build

interactive:
    # Install scene_graph_prediction as an interactive framework
	pip install -e .
	rm -rf build

install:
    # Install scene_graph_prediction to the Python site-packages
	python -m pip install .
	rm -rf build
