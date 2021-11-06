debug:
	gcc main.cpp\
		-lm\
		-lstdc++\
		`pkg-config --libs --cflags opencv4`\
		-g -o application
	./application "filename"
	rm ./application