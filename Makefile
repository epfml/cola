.PHONY: clean
clean:
	rm -f -r build/
	rm -f fast_cd/*.so
	rm -f fast_cd/*.c

.PHONY: build
build: 
	python setup.py build_ext --inplace

.PHONY: install
install:
	python setup.py install
