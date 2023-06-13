#include <fstream>
#include <cmath>
#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <sstream>

using namespace std;

class Vector2 {
public:
    long double x;
    long double y;

    Vector2() : x(0.0f), y(0.0f) {}
    Vector2(long double x, long double y) : x(x), y(y) {}

    Vector2(const Vector2& other) : x(other.x), y(other.y) {}

    Vector2 operator+(const Vector2& other) const {
        return Vector2(x + other.x, y + other.y);
    }

    Vector2 operator-(const Vector2& other) const {
        return Vector2(x - other.x, y - other.y);
    }

    Vector2 operator*(long double scalar) const {
        return Vector2(x * scalar, y * scalar);
    }

    Vector2 operator/(long double scalar) const {
        return Vector2(x / scalar, y / scalar);
    }

    long double dot(const Vector2& other) const {
        return x * other.x + y * other.y;
    }

    long double magnitude() const {
        return sqrt(x * x + y * y);
    }

    Vector2 normalize() const {
        long double mag = magnitude();
        if (mag != 0.0f) {
            return Vector2(x / mag, y / mag);
        }
        return Vector2(0.0f, 0.0f);
    }

    string toString() const {
        return "(" + to_string(x) + ", " + to_string(y) + ")";
    }
};
class Complex {
public:
    long double real;
    long double imag;

    Complex() : real(0.0f), imag(0.0f) {}
    Complex(long double real, long double imag) : real(real), imag(imag) {}

    Complex(const Complex& other) : real(other.real), imag(other.imag) {}

    Complex operator+(const Complex& other) const {
        return Complex(real + other.real, imag + other.imag);
    }

    Complex operator-(const Complex& other) const {
        return Complex(real - other.real, imag - other.imag);
    }

    Complex operator*(const Complex& other) const {
        return Complex(real * other.real - imag * other.imag,
                       real * other.imag + imag * other.real);
    }

    Complex operator/(const Complex& other) const {
        return Complex((real * other.real + imag * other.imag) / (other.real * other.real + other.imag * other.imag),
                       (imag * other.real - real * other.imag) / (other.real * other.real + other.imag * other.imag));
    }

    long double abs() {
        return sqrt(real*real + imag*imag);
    }

    string toString() const {
        string sign = "+";
        if (imag < 0) {
            sign = "";
        }
        return "(" + to_string(real) + sign + to_string(imag) + "i)";
    }
};
class Color {
public:
    int r;
    int g;
    int b;

    Color() : r(0), g(0), b(0) {}
    Color(long double r, long double g, long double b) : r(floor(r * 255.0)), g(floor(g * 255.0)), b(floor(b * 255.0)) {}
};
class MandelbrotSet {
public:
    string name;
    int width;
    int height;
    Vector2 xrange;
    Vector2 yrange;
    int maxiterations;
    int palletelength;
    vector<long double> palletepositions;
    vector<Color> palletecolors;
    int palletesize;
    unsigned char* imageData;
    bool showpercent;

    MandelbrotSet() : name("output"),
                      width(400),
                      height(300),
                      xrange(Vector2(-2.2f, 0.8f)),
                      yrange(-1.2f, 1.2f),
                      maxiterations(100),
                      palletepositions({0,0.15,0.33, 0.67, 0.85, 1}),
                      palletecolors({Color(1.0f, 1.0f, 1.0f), Color(1.0f, 0.8f, 0.0f), Color(0.53f, 0.12f, 0.075f), Color(0.0f, 0.0f, 0.6f), Color(0.0f, 0.4f, 1.0f), Color(1.0f, 1.0f, 1.0f)}),
                      palletesize(6),
                      showpercent(true)
    {Init();}
    MandelbrotSet(string name, int width, int height, Vector2 xrange, Vector2 yrange, int iterations, int palletelength, vector<long double> palletepositions, vector<Color> palletecolors, int palletsize, bool showpercent) : name(name),
                                                                                                                                                                                                                                width(width),
                                                                                                                                                                                                                                height(height),
                                                                                                                                                                                                                                xrange(xrange),
                                                                                                                                                                                                                                yrange(yrange),
                                                                                                                                                                                                                                maxiterations(iterations),
                                                                                                                                                                                                                                palletelength(palletelength),
                                                                                                                                                                                                                                palletepositions(palletepositions),
                                                                                                                                                                                                                                palletecolors(palletecolors),
                                                                                                                                                                                                                                palletesize(palletsize),
                                                                                                                                                                                                                                showpercent(showpercent)
    {Init();}

    void Init() {
        imageData = new unsigned char[width * height * 3];
    };

    void SaveBitmap(string filename)
    {
        // Define the bitmap file header
        unsigned char bitmapFileHeader[14] = {
                'B', 'M',             // Signature
                0, 0, 0, 0,           // File size (to be filled later)
                0, 0, 0, 0,           // Reserved
                54, 0, 0, 0           // Pixel data offset
        };

        // Define the bitmap info header
        unsigned char bitmapInfoHeader[40] = {
                40, 0, 0, 0,          // Info header size
                0, 0, 0, 0,           // Image width (to be filled later)
                0, 0, 0, 0,           // Image height (to be filled later)
                1, 0,                // Number of color planes
                24, 0,               // Bits per pixel (24 bits for RGB)
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
    void PutPixel(int x, int y, int r, int g, int b) {
        int position = (x + y * width) * 3;
        imageData[position] = r;
        imageData[position + 1] = g;
        imageData[position + 2] = b;
    }

    long double Map(long double value, long double inmin, long double inmax, long double outmin, long double outmax) {
        return (value - inmin) * (outmax - outmin) / (inmax - inmin) + outmin;
    }

    Color ColorPallete(int iterations) {
        long double t = (long double) (iterations % palletelength) / (long double) maxiterations * ((long double) maxiterations / (long double) palletelength);

        if (iterations == maxiterations) {
            return Color(0.0,0.0,0.0);
        }

        int index = 0;
        while (index < palletesize - 1 && t > palletepositions[index + 1]) {
            index++;
        }

        long double startValue = palletepositions[index];
        long double endValue = palletepositions[index + 1];
        long double fraction = (t - startValue) / (endValue - startValue);

        Color startColor = palletecolors[index];
        Color endColor = palletecolors[index + 1];

        long double r = startColor.r + (endColor.r - startColor.r) * fraction;
        long double g = startColor.g + (endColor.g - startColor.g) * fraction;
        long double b = startColor.b + (endColor.b - startColor.b) * fraction;

        return Color(r/255.0, g/255.0, b/255.0);
    }

    Complex QP(Complex Z, Complex C) {
        return Z * Z + C;
    }

    Color CalcPixel(int x, int y) {
        long double locx = Map(x, 0, width-1, xrange.x, xrange.y);
        long double locy = Map(y, 0, height-1, yrange.x, yrange.y);

        Complex Z = Complex(0,0);
        Complex C = Complex(locx, locy);

        int iterations = 0;

        for (int i = 0; i < maxiterations; i++) {
            Z = QP(Z, C);
            iterations = i + 1;
            if (Z.abs() >= 2) {
                break;
            }
        }
        return ColorPallete(iterations);
    }

    void Generate() {
        vector<thread> threads;

        int maxthreads = 200;
        int pixelsPerThread = width * height / maxthreads;
        for (int t = 0; t < maxthreads; t++)
        {
            int startPixel = t * pixelsPerThread;
            int endPixel = (t == maxthreads - 1) ? width * height : (t + 1) * pixelsPerThread;
            threads.push_back(thread([startPixel, endPixel, this]()
                                     {
                                         for (int p = startPixel; p < endPixel; p++)
                                         {
                                             int x = p % width;
                                             int y = p / width;
                                             Color ret = CalcPixel(x, y);
                                             PutPixel(x, y, ret.r, ret.g, ret.b);
                                         }
                                     }));
            if (showpercent)
            {
                cout << "THREAD START --> " << ((long double)t / (long double)maxthreads) * 100.0 << "%" << endl;
            }
        }
        int threadi = 0;
        for (auto& thread : threads)
        {
            threadi++;
            thread.join();
            if (showpercent)
            {
                cout << "THREAD FINISH --> " << ((long double)threadi / (long double)maxthreads) * 100.0 << "%" << endl;
            }
        }


        SaveBitmap("..\\" + name + ".bmp");
        delete[] imageData;
    }
};
string extractTagValue(const string& xmlString, const string& tag, const string& params) {
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
vector<string> splitString(const string& str, char delimiter) {
    vector<string> tokens;
    stringstream ss(str);
    string token;

    while (getline(ss, token, delimiter)) {
        tokens.push_back(token);
    }

    return tokens;
}
int main()
{
    bool isxml = true;
    bool customzoom = false;
    bool custommid = false;

    auto start = chrono::high_resolution_clock::now();

    vector<long double> postionpallete = {
            0.00,
            0.16,
            0.33,
            0.50,
            0.66,
            0.83,
            1.00
    };
    vector<Color> colorpallete = {
            Color(0.6883375335545983,0.16871764516775567,0.8324354058673153),
            Color(0.9771288652228365,0.1044934047374777,0.08384453764972455),
            Color(0.9512752276787326,0.8025247061198186,0.9575240369100828),
            Color(0.8051087811064563,0.3783239554070077,0.20484907678274356),
            Color(0.21507387033140257,0.4119935529442995,0.8468136222792582),
            Color(0.9068445384437034,0.08729733839430565,0.39949722226246065),
            Color(0.6883375335545983,0.16871764516775567,0.8324354058673153)
    };

    int width = 160 * 10;
    int height = 90 * 10;

    long double xmin = -0.910569802097113617144;
    long double xmax = -0.909146263305206895768;
    long double ymin = -0.264522003480475537074;
    long double ymax = -0.263721579533424782706;

    if (isxml) {
        // https://math.hws.edu/eck/js/mandelbrot/MB.html

        ifstream file("..\\data.xml");
        string xmlString((istreambuf_iterator<char>(file)),
                         istreambuf_iterator<char>());

        long double i_xmin;
        long double i_xmax;
        long double i_ymin;
        long double i_ymax;

        stringstream s_xmin(extractTagValue(xmlString, "xmin", ""));
        stringstream s_xmax(extractTagValue(xmlString, "xmax", ""));
        stringstream s_ymin(extractTagValue(xmlString, "ymin", ""));
        stringstream s_ymax(extractTagValue(xmlString, "ymax", ""));

        s_xmin >> i_xmin;
        s_xmax >> i_xmax;
        s_ymin >> i_ymin;
        s_ymax >> i_ymax;

        string palette = extractTagValue(xmlString, "palette", " colorType='RGB'");
        vector<string> split = splitString(palette, '\n');
        split.pop_back();
        split.erase(split.begin());

        vector<long double> i_postionpallete;
        vector<Color> i_colorpallete;

        for (string a : split) {
            vector<string> params = splitString(a, '\'');
            vector<string> colorvals = splitString(params[3], ';');

            stringstream s_r(colorvals[0]);
            stringstream s_g(colorvals[1]);
            stringstream s_b(colorvals[2]);

            stringstream s_pos(params[1]);

            long double i_r;
            long double i_g;
            long double i_b;

            long double i_pos;

            s_r >> i_r;
            s_g >> i_g;
            s_b >> i_b;

            s_pos >> i_pos;

            i_postionpallete.push_back(i_pos);
            i_colorpallete.push_back(Color(i_r, i_g, i_b));
        }

        postionpallete = i_postionpallete;
        colorpallete = i_colorpallete;

        xmin = i_xmin;
        xmax = i_xmax;
        ymin = i_ymin;
        ymax = i_ymax;
    }

    int palletesize = colorpallete.size();

//    long double xmid = (xmin + xmax) / 2.0f;
//    long double ymid = (ymin + ymax) / 2.0f;

    long double xmid = 0.360240443437614363236125244449545308482607807958585750488;
    long double ymid = -0.641313061064803174860375015179302066579494952282305259556;

    long double ratio = (long double) width / (long double) height;
    long double div = 30.0;
    long double zoom = 4.0;

    int count = 1;

    int maxiterations = 1000;
    int palletelength = 250;

    for (int i = 1; i <= count; i++) {
        zoom = zoom + zoom / div;
        if (customzoom) {
            long double halfwidth = 1.0/(zoom/div) / 2.0;
            long double halfheight = halfwidth / ratio;

            xmin = xmid - halfwidth;
            xmax = xmid + halfwidth;

            ymin = ymid - halfheight;
            ymax = ymid + halfheight;
        }

        MandelbrotSet mainset(
                "output_" + to_string(i),
                width,
                height,
                Vector2(xmin, xmax),
                Vector2(ymin, ymax),
                maxiterations,
                palletelength,
                postionpallete,
                colorpallete,
                palletesize,
                true
        );
        mainset.Generate();
        cout << i << "/" << count << " (" << (long double)i / ((long double)count/100.0) << "%) " << zoom << endl << endl;
    }
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<long double> duration = end - start;
    cout << "Operation took " << duration.count() << " seconds." << endl;
    return 0;
}