#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math_functions.h>
#include <curand_kernel.h>

#define IS_ZERO(val) val > -EPSILON && val < EPSILON

#define WIDTH 2000
#define HEIGHT 2000

#define EPSILON 0.0001
#define M_PI 3.14159265
#define MAX_DISTANCE 10.0

#define BAIL_OUT 100.0
#define PALLETE_SCALE 10.0
#define PALLETE_OFFSET 0.0

#define CROSS(v1, v2) vec3((v1).y * (v2).z - (v1).z * (v2).y, (v1).z * (v2).x - (v1).x * (v2).z, (v1).x * (v2).y - (v1).y * (v2).x)
#define DOT(v1, v2) ((v1).x * (v2).x + (v1).y * (v2).y + (v1).z * (v2).z)
#define DOT_2(v1, v2) ((v1).x * (v2).x + (v1).y * (v2).y)
#define LENGTH(v) (sqrt((v).x * (v).x + (v).y * (v).y + (v).z * (v).z))
#define NORMALIZE(v) ((v) / LENGTH(v))

using namespace std;

class ivec2 {
public:
    int x;
    int y;

    ivec2() : x(0), y(0) {}
    ivec2(int x, int y) : x(x), y(y) {}
};

struct vec3 {
    double x;
    double y;
    double z;

    __host__ __device__ vec3() : x(0.0), y(0.0), z(0.0) {}
    __host__ __device__ vec3(double x, double y, double z) : x(x), y(y), z(z) {}
    __host__ __device__ vec3(const vec3& other) : x(other.x), y(other.y), z(other.z) {}

    __host__ __device__ vec3 operator+(const vec3& other) const {
        return vec3(x + other.x, y + other.y, z + other.z);
    }
    __host__ __device__ vec3 operator-(const vec3& other) const {
        return vec3(x - other.x, y - other.y, z - other.z);
    }
    __host__ __device__ vec3 operator*(const vec3& other) const {
        return vec3(x * other.x, y * other.y, z * other.z);
    }
    __host__ __device__ vec3 operator/(const vec3& other) const {
        return vec3(x / other.x, y / other.y, z / other.z);
    }
    __host__ __device__ vec3 operator*(double scalar) const {
        return vec3(x * scalar, y * scalar, z * scalar);
    }
    __host__ __device__ vec3 operator/(double scalar) const {
        return vec3(x / scalar, y / scalar, z / scalar);
    }
};
struct vec2 {
    double x;
    double y;

    __device__ vec2() : x(0.0), y(0.0) {}
    __device__ vec2(double x, double y) : x(x), y(y) {}
    __device__ vec2(const vec2& other) : x(other.x), y(other.y) {}

    __device__ vec2 operator+(const vec2& other) const {
        return vec2(x + other.x, y + other.y);
    }
    __device__ vec2 operator-(const vec2& other) const {
        return vec2(x - other.x, y - other.y);
    }
    __device__ vec2 operator*(const vec2& other) const {
        return vec2(x * other.x, y * other.y);
    }
    __device__ vec2 operator/(const vec2& other) const {
        return vec2(x / other.x, y / other.y);
    }
    __device__ vec2 operator*(double scalar) const {
        return vec2(x * scalar, y * scalar);
    }
    __device__ vec2 operator/(double scalar) const {
        return vec2(x / scalar, y / scalar);
    }
};

class fcolor {
public:
    float red;
    float green;
    float blue;

    __host__ __device__ fcolor() : red(0.0), green(0.0), blue(0.0) {}
    __host__ __device__ fcolor(float red, float green, float blue) : red(red), green(green), blue(blue) {}
    __host__ __device__ fcolor(vec3 vector_color) : red(vector_color.x), green(vector_color.y), blue(vector_color.z) {}

    __host__ __device__ fcolor operator+(const fcolor& other) const {
        return fcolor(red + other.red, green + other.green, blue + other.blue);
    }
    __host__ __device__ fcolor operator-(const fcolor& other) const {
        return fcolor(red - other.red, green - other.green, blue - other.blue);
    }
    __host__ __device__ fcolor operator*(const fcolor& other) const {
        return fcolor(red * other.red, green * other.green, blue * other.blue);
    }
    __host__ __device__ fcolor operator*(double scalar) const {
        return fcolor(red * scalar, green * scalar, blue * scalar);
    }
    __host__ __device__ fcolor operator/(float scalar) const {
        return fcolor(red / scalar, green / scalar, blue / scalar);
    }
};
class icolor {
public:
    int red;
    int green;
    int blue;

    __host__ __device__ icolor() : red(0), green(0), blue(0) {}
    __host__ __device__ icolor(int red, int green, int blue) : red(red), green(green), blue(blue) {}
    __host__ __device__ icolor(fcolor other) : red((int)floor(other.red * 255.0)), green((int)floor(other.green * 255.0)), blue((int)floor(other.blue * 255.0)) {}

    __host__ __device__ icolor operator+(const icolor& other) const {
        return icolor(red + other.red, green + other.green, blue + other.blue);
    }

    __host__ __device__ icolor operator-(const icolor& other) const {
        return icolor(red - other.red, green - other.green, blue - other.blue);
    }

    __host__ __device__ icolor operator*(float scalar) const {
        return icolor((int)floor(red * scalar), (int)floor(green * scalar), (int)floor(blue * scalar));
    }

    __host__ __device__ icolor operator/(float scalar) const {
        return icolor((int)floor(red / scalar), (int)floor(green / scalar), (int)floor(blue / scalar));
    }

    __host__ __device__ icolor copy() const {
        return icolor(red, green, blue);
    }
};

struct settings {
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    int iters;
};

__device__ curandState_t getRandState(unsigned int seed) {
    curandState_t state;
    curand_init(seed, 0, 0, &state);
    return state;
}
__device__ double getRandomDouble(curandState_t& state) {
    return curand_uniform(&state);
}
__device__ fcolor pallete(int iters, vec2 z, int max_iters) {
    float nu = (float)iters + 1.0f - log2(log2((float)DOT_2(z, z)));
    float t = (nu * PALLETE_SCALE) / (float)max_iters;
    
    fcolor a = fcolor(0.5f, 0.5f, 0.5f);
    fcolor b = fcolor(0.5f, 0.5f, 0.5f);
    fcolor c = fcolor(1.0f, 1.0f, 1.0f);
    fcolor d = fcolor(0.249f, 1.338f, 1.236f);
    // 0.115f, 0.258f, 1.338f def
    // 1.338f,1.195f,0.222f pink green black
    // 0.249f,1.338f,1.236f purple green black

    fcolor n = (c * t + d) * (M_PI * 2.0f);

    return a + b * fcolor(cos(n.red), cos(n.green), cos(n.blue));
}
__device__ fcolor mandelbrot_pixel(vec2 c, int max_iters) {
    int iters = 0;
    vec2 z = vec2(0.0, 0.0);
    for (int i = 0; i < max_iters; i++) {
        iters = i + 1;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        if (DOT_2(z, z) > BAIL_OUT * BAIL_OUT)
            break;
    }
    if (iters == max_iters)
        return fcolor(0.0f, 0.0f, 0.0f);
    return pallete(iters, z, max_iters);
}
__device__ double map(double value, double imin, double imax, double omin, double omax) {
    double irange = imax - imin;
    double orange = omax - omin;
    return (value - imin) / irange * orange + omin;
}

__global__ void calculate_pixel(ivec2* positions, icolor* colors, int size, int width, int height, int index, double xmin, double xmax, double ymin, double ymax, int max_iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        int x = positions[tid].x;
        int y = positions[tid].y;

        vec2 uv = vec2(x, y) / vec2(width, height);

        uv.x = map(uv.x, 0.0, 1.0, xmin, xmax);
        uv.y = map(uv.y, 0.0, 1.0, ymin, ymax);

        fcolor out_color = mandelbrot_pixel(uv, max_iters);

        // OUTPUT
        colors[tid] = icolor(out_color);
    }
}

class GPU_Image {
public:
    string name;
    int width;
    int height;
    int i;
    double xmin;
    double xmax;
    double ymin;
    double ymax;
    int iters;
    unsigned char* imageData;

    GPU_Image(const string& name, int width, int height, int i, double xmin, double xmax, double ymin, double ymax, int iters) : name(name), width(width), height(height), i(i), xmin(xmin), xmax(xmax), ymin(ymin), ymax(ymax), iters(iters) {}

    void save_bitmap(const string& filename)
    {
        // Define the bitmap file header
        unsigned char bitmapFileHeader[14] = {
                'B', 'M',                     // Signature
                0, 0, 0, 0,           // File size (to be filled later)
                0, 0, 0, 0,           // Reserved
                54, 0, 0, 0        // Pixel data offset
        };

        // Define the bitmap info header
        unsigned char bitmapInfoHeader[40] = {
                40, 0, 0, 0,            // Info header size
                0, 0, 0, 0,             // Image width (to be filled later)
                0, 0, 0, 0,           // Image height (to be filled later)
                1, 0,                         // Number of color planes
                24, 0,                        // Bits per pixel (24 bits for RGB)
                0, 0, 0, 0,          // Compression method (none)
                0, 0, 0, 0,          // Image size (can be set to 0 for uncompressed images)
                0, 0, 0, 0,          // Horizontal resolution (can be set to 0 for uncompressed images)
                0, 0, 0, 0,          // Vertical resolution (can be set to 0 for uncompressed images)
                0, 0, 0, 0,          // Number of colors in the palette (not used for 24-bit images)
                0, 0, 0, 0           // Number of important colors (not used for 24-bit images)
        };

        // Calculate the padding bytes
        int paddingSize = (4 - (width * 3) % 4) % 4;

        // Calculate the file size
        int fileSize = 54 + (width * height * 3) + (paddingSize * height);

        // Fill in the file size in the bitmap file header
        bitmapFileHeader[2] = (unsigned char)(fileSize);
        bitmapFileHeader[3] = (unsigned char)(fileSize >> 8);
        bitmapFileHeader[4] = (unsigned char)(fileSize >> 16);
        bitmapFileHeader[5] = (unsigned char)(fileSize >> 24);

        // Fill in the image width in the bitmap info header
        bitmapInfoHeader[4] = (unsigned char)(width);
        bitmapInfoHeader[5] = (unsigned char)(width >> 8);
        bitmapInfoHeader[6] = (unsigned char)(width >> 16);
        bitmapInfoHeader[7] = (unsigned char)(width >> 24);

        // Fill in the image height in the bitmap info header
        bitmapInfoHeader[8] = (unsigned char)(height);
        bitmapInfoHeader[9] = (unsigned char)(height >> 8);
        bitmapInfoHeader[10] = (unsigned char)(height >> 16);
        bitmapInfoHeader[11] = (unsigned char)(height >> 24);

        // Open the output file
        ofstream file(filename, ios::binary);

        // Write the bitmap headers
        file.write(reinterpret_cast<const char*>(bitmapFileHeader), sizeof(bitmapFileHeader));
        file.write(reinterpret_cast<const char*>(bitmapInfoHeader), sizeof(bitmapInfoHeader));

        // Write the pixel data (BGR format) row by row
        for (int y = height - 1; y >= 0; y--)
        {
            for (int x = 0; x < width; x++)
            {
                // Calculate the pixel position
                int position = (x + y * width) * 3;

                // Write the pixel data (BGR order)
                file.write(reinterpret_cast<const char*>(&imageData[position + 2]), 1); // Blue
                file.write(reinterpret_cast<const char*>(&imageData[position + 1]), 1); // Green
                file.write(reinterpret_cast<const char*>(&imageData[position]), 1);     // Red
            }

            // Write the padding bytes
            for (int i = 0; i < paddingSize; i++)
            {
                file.write("\0", 1);
            }
        }

        // Close the file
        file.close();
    }
    void put_pixel(const int x, const int y, const int r, const int g, const int b) {
        int position = (x + y * width) * 3;
        imageData[position] = r;
        imageData[position + 1] = g;
        imageData[position + 2] = b;
    }
    void generate() {
        imageData = new unsigned char[width * height * 3];

        const int size = width * height;

        // DEFINE PARAMETERS

        ivec2* positions = new ivec2[size];
        icolor* colors = new icolor[size];

        // DEFINE SIZES

        const long vector2int_size = size * sizeof(ivec2);
        const long colorint_size = size * sizeof(icolor);

        // UPDATE I/O PARAMERTERS

        for (int i = 0; i < size; i++) {
            int x = i % width;
            int y = i / width;

            positions[i] = ivec2(x, y);
            colors[i] = icolor();
        }

        // DEFINE D_PARAMETERS

        ivec2* d_positions;
        icolor* d_colors;

        cudaMalloc((void**)&d_positions, vector2int_size);
        cudaMalloc((void**)&d_colors, colorint_size);

        // MEMORY COPY PARAMETERS

        cudaMemcpy(d_positions, positions, vector2int_size, cudaMemcpyHostToDevice);
        cudaMemcpy(d_colors, colors, colorint_size, cudaMemcpyHostToDevice);

        // RUN

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

        calculate_pixel << < blocksPerGrid, threadsPerBlock >> > (d_positions, d_colors, size, width, height, i, xmin, xmax, ymin, ymax, iters);

        cudaMemcpy(colors, d_colors, colorint_size, cudaMemcpyDeviceToHost);

        // PROCESS OUTPUT

        for (int i = 0; i < size; i++) {
            int x = i % width;
            int y = i / width;

            icolor color = colors[i];
            put_pixel(x, height - y - 1, color.red, color.green, color.blue);
        }

        save_bitmap("./output/" + name + ".bmp");

        // FREE MEMORY

        cudaFree(d_positions);
        cudaFree(d_colors);

        delete[] positions;
        delete[] colors;

        delete[] imageData;
    }
};

vector<string> splitStringStr(const string& str, const string& delimiter) {
    vector<string> tokens;
    size_t start = 0;
    size_t end = 0;
    while ((end = str.find(delimiter, start)) != string::npos) {
        tokens.push_back(str.substr(start, end - start));
        start = end + delimiter.length();
    }
    tokens.push_back(str.substr(start));
    return tokens;
}
vector<string> splitStringChr(const string& str, char delimiter) {
    vector<string> tokens;
    stringstream ss(str);
    string token;
    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}
string extractTagValue(const string& xmlString, const string& tag, const string& params, bool is_max_iters) {
    if (is_max_iters) {
        string start = "<max_iterations value='";
        string end = "'/>";
        string ret = splitStringStr(splitStringStr(xmlString, start)[1], end)[0];
        return ret;
    }
    else {
        string startTag = "<" + tag + params + ">";
        string endTag = "</" + tag + ">";
        size_t startPos = xmlString.find(startTag);
        size_t endPos = xmlString.find(endTag);
        if (startPos != string::npos && endPos != string::npos) {
            startPos += startTag.length();
            return xmlString.substr(startPos, endPos - startPos);
        }
        return "";
    }
}
settings get_data(string path) {
    // https://math.hws.edu/eck/js/mandelbrot/MB.html

    ifstream file(path);
    string xmlString((istreambuf_iterator<char>(file)),
        istreambuf_iterator<char>());

    double i_xmin;
    double i_xmax;
    double i_ymin;
    double i_ymax;

    int i_iters;

    stringstream s_xmin(extractTagValue(xmlString, "xmin", "", false));
    stringstream s_xmax(extractTagValue(xmlString, "xmax", "", false));
    stringstream s_ymin(extractTagValue(xmlString, "ymin", "", false));
    stringstream s_ymax(extractTagValue(xmlString, "ymax", "", false));

    stringstream s_iters(extractTagValue(xmlString, "", "", true));


    s_xmin >> i_xmin;
    s_xmax >> i_xmax;
    s_ymin >> i_ymin;
    s_ymax >> i_ymax;

    s_iters >> i_iters;

    settings ret_data = {
        i_xmin,
        i_xmax,
        i_ymin,
        i_ymax,
        i_iters
    };

    return ret_data;
}
settings scale_data(settings data, float zoom) {
    double xrange = data.xmax - data.xmin;
    double yrange = data.ymax - data.ymin;

    // Calculate the aspect ratio of the viewport
    double aspect_ratio = (double)(WIDTH) / (double)HEIGHT;

    // Calculate the new range of x-coordinates and y-coordinates
    double new_xrange = xrange / zoom;
    double new_yrange = yrange / zoom;

    // Calculate the center coordinates of the viewport
    double center_x = (data.xmin + data.xmax) / 2.0;
    double center_y = (data.ymin + data.ymax) / 2.0;

    // Calculate the new xmin, xmax, ymin, and ymax values
    double new_xmin = center_x - new_xrange / 2.0;
    double new_xmax = center_x + new_xrange / 2.0;
    double new_ymin = center_y - new_yrange / (2.0 * aspect_ratio);
    double new_ymax = center_y + new_yrange / (2.0 * aspect_ratio);

    // Update the input coordinates with the new values
    return {
        new_xmin,
        new_xmax,
        new_ymin,
        new_ymax,
        data.iters
    };
}
float get_root_power(settings start_range, settings end_range, int frame_count) {
    return pow((start_range.xmax - start_range.xmin) / (2.0 * (-end_range.xmin + (start_range.xmin + start_range.xmax) / 2.0)), 1.0 / ((float)frame_count - 1.0));
}
int main() {
    bool isxml = true;
    bool isanimation = true;
    int frame_count = 500;

    settings data_settings = {
        -1.0,
        1.0,
        -1.0,
        1.0,
        100
    };

    settings data_settings_start;
    settings data_settings_end;

    string path_data = "./data.xml";
    string path_start = "./data_start.xml";
    string path_end = "./data_end.xml";

    if (isxml) {
        if (isanimation) {
            data_settings_start = get_data(path_start);
            data_settings_end = get_data(path_end);
        }
        else {
            data_settings = get_data(path_data);
        }
    }

    chrono::system_clock::time_point start_time = chrono::system_clock::now();

    if (isanimation) {
        for (int i = 0; i < frame_count; i++) {
            chrono::system_clock::time_point start_frame = chrono::system_clock::now();

            string name = "out_" + to_string(i);

            float zoom = (float)i;
            settings new_data = scale_data(data_settings_start, pow(get_root_power(data_settings_start, data_settings_end, frame_count), zoom));

            
            GPU_Image main = GPU_Image(name, WIDTH, HEIGHT, i, new_data.xmin, new_data.xmax, new_data.ymin, new_data.ymax, new_data.iters);
            main.generate();

            chrono::time_point<chrono::system_clock> end_frame = chrono::system_clock::now();
            chrono::duration<double> duration_frame = end_frame - start_frame;

            cout << i + 1 << "/" << frame_count << " (" << floor((double)(i + 1.0) / (double)frame_count * 1000.0) / 10.0 << "%)" << " " << duration_frame.count() << " seconds | ETA: " << ((double)(frame_count - (i + 1)) * duration_frame.count()) << " seconds" << endl;
        }
    }
    else {
        GPU_Image main = GPU_Image("out_image", WIDTH, HEIGHT, 0, data_settings.xmin, data_settings.xmax, data_settings.ymin, data_settings.ymax, data_settings.iters);
        main.generate();
    }

    chrono::time_point<chrono::system_clock> end_time = chrono::system_clock::now();
    chrono::duration<double> duration = end_time - start_time;
    cout << "Total time taken: " << duration.count() << " seconds" << endl;

    return 0;
}
