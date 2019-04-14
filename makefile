C=`pkg-config --libs opencv`
D=-I /usr/local/include/raspicam -lraspicam -lraspicam_cv -lopencv_core -lopencv_highgui 
E=-I /usr/include/opencv2/ -lhighgui -lcore
F=-I /usr/local/include/ -lwiringPiDev -lwiringPi
H=-lmvnc

.PHONY: all
all: ssd


.PHONY:	run
ssd: 
	@echo "\nmaking myssd"
	g++ myssd.cpp pca9685.c -o myssd $(C) $(D) $(F) $(G) $(H)
	@echo "Created run executable"

.PHONY: run
run: ssd
	@echo "\nmaking myssd"
	./myssd

.PHONY: help
help:
	@echo "possible make targets: ";
	@echo "  make help - shows this message";
	@echo "  make all - makes the following: cpp, myssd_cpp";
	@echo "  make run - builds the run executable example";
	@echo "  make run - runs the run executable example program";
	@echo "  make clean - removes all created content and temporary files";

clean: clean
	@echo "\nmaking clean";
	rm -f myssd
