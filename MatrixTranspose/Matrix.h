#pragma once

template <typename T, const unsigned int WIDTH, const unsigned int HEIGHT>
class Matrix {
private:
	std::unique_ptr<T[]> data;
	static const size_t NUM_ELEMENTS = WIDTH * HEIGHT;
public:
	Matrix(const T(&data)[NUM_ELEMENTS]) : data(new T[NUM_ELEMENTS]()) {
		for (unsigned int i = 0; i < NUM_ELEMENTS; i++) {
			this->data.get()[i] = data[i];
		}
	}

	Matrix() : data(new T[NUM_ELEMENTS]) {};

	T operator[](int i) const {
		return data.get()[i];
	}

	T& operator[](int i) {
		return data.get()[i];
	}

	bool operator==(const Matrix& other) const {
		for (unsigned int i = 0; i < NUM_ELEMENTS; i++) {
			if (data[i] != other.data[i]) {
				return false;
			}
		}
		return true;
	}

	T * getData() {
		return data.get();
	}

	const T * getData() const {
		return data.get();
	}
};

template <typename T, const unsigned int WIDTH, const unsigned int HEIGHT>
std::ostream& operator<<(std::ostream& os, const Matrix<T, WIDTH, HEIGHT>& matrix) {
	os.precision(6);
	for (unsigned int j = 0; j < HEIGHT; j++) {
		std::string delimiter("");
		for (unsigned int i = 0; i < WIDTH; i++) {
			os << delimiter << matrix[j*WIDTH + i];
			delimiter = ", ";
		}
		os << std::endl;
	}
	return os;
}