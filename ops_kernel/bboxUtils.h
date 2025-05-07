#ifndef BBOX_UTILS_H
#define BBOX_UTILS_H

template <typename T>
struct Bbox{
    T x_min;
    T y_min;
    T x_max;
    T y_max;

    Bbox(T x1, T y1, T x2, T y2) : x_min(x1), y_min(y1), x_max(x2), y_max(y2) {}

    Bbox() =default;

    void print() const {
        std::cout << "Bbox(" << x_min << ", " << y_min << ", " << x_max << ", " << y_max << ")" << std::endl;
    }
};

template <typename T>
struct BboxInfo{
    T conf_score;
    int32_t label;
    int32_t bbox_idx;
    bool kept;
    BboxInfo(T conf_score,int32_t label,int32_t bbox_idx,bool kept)
    : conf_score(conf_score), label(label), bbox_idx(bbox_idx), kept(kept) {}

    BboxInfo() =default;

    void print() const {
        std::cout << "BboxInfo(" << conf_score << ", " << label << ", " << bbox_idx << ", " << kept << ")" << std::endl;
    }

};

template <typename TFloat>
bool operator<(const BboxInfo<TFloat>& lhs, const BboxInfo<TFloat>& rhs) {
    return lhs.x1 < rhs.x1;
}

template <typename TFloat>
bool operator==(const BboxInfo<TFloat>& lhs, const BboxInfo<TFloat>& rhs) {
    return lhs.x1 == rhs.x1 && lhs.y1 == rhs.y1 &&
           lhs.x2 == rhs.x2 && lhs.y2 == rhs.y2;
}


typedef enum
{
    NCHW = 0,
    NC4HW = 1,
    NC32HW = 2
} DLayout_t;


#endif // BBOX_UTILS_H