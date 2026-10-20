#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    // Resimleri bulundukları dizinden okuyalım.
    string dosya_yolu = "inputs/*.jpg";
    vector<String> resimler;
    glob(dosya_yolu, resimler);

    if (resimler.empty()) {
        cout << "Klasorde okunacak gorsel bulunamadi!" << endl;
        return -1;
    }

    // Resize boyutu belirleyelim
    Size yeni_boyut(1024, 768);

    // Resimleri donguyle okuyalım ve resize edelim
    for (size_t i = 0; i < resimler.size(); i++) {
        Mat img = imread(resimler[i]);
        if (img.empty()) continue;

        Mat cikti;
        //  Inter linear fonksiyonu kullanılarak (En yakın 4 pikselin ağırlıklı ortalamasını alır. Düzgün ve yumuşak geçiş verir. Büyütme işleminde standart kullanım.)
        resize(img, cikti, yeni_boyut, 0, 0, INTER_LINEAR);

        string kayit_adi = "cikti_" + to_string(i + 1) + "_1024x768.jpg";
        imwrite(kayit_adi, cikti);
        cout << resimler[i] << " -> " << kayit_adi << " olarak boyutlandirildi." << endl;
    }


    
    cout << "Tum gorseller 1024x768 olarak kaydedildi." << endl;
    return 0;
}