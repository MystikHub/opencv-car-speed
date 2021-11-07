debug:
	gcc main.cpp\
		-lm\
		-lstdc++\
		`pkg-config --libs --cflags opencv4`\
		-g -o vehicle-speed -Wall
	./vehicle-speed "filename"
	rm ./vehicle-speed

# Same as above without running the program
# Useful for debugging in VS Code
build:
	gcc main.cpp\
		-lm\
		-lstdc++\
		`pkg-config --libs --cflags opencv4`\
		-g -o vehicle-speed -Wall