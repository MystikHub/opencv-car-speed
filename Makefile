debug:
	gcc main.cpp\
		-lm\
		-lstdc++\
		`pkg-config --libs --cflags opencv4`\
		-g -o vehicle-speed
	./vehicle-speed "filename"
	rm ./vehicle-speed