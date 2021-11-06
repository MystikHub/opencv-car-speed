debug:
	gcc main.cpp\
		-lm\
		-lstdc++\
		`pkg-config --libs --cflags opencv4`\
		-g -o vehicle-speed -Wall
	./vehicle-speed "filename"
	rm ./vehicle-speed

build:
	gcc main.cpp\
		-lm\
		-lstdc++\
		`pkg-config --libs --cflags opencv4`\
		-g -o vehicle-speed -Wall