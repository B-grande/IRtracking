#ifndef TRACKER_C_H
#define TRACKER_C_H

#ifdef __cplusplus
extern "C" {
#endif

// Simple structure representing a detected blob
typedef struct {
    int x;      // x-coordinate (centroid)
    int y;      // y-coordinate (centroid)
    int area;   // number of pixels in the blob
} Blob;

/**
 * detect_blobs - Scan a grayscale image and detect connected bright regions.
 *
 * @gray: Pointer to the grayscale image data.
 * @width: Image width (number of pixels per row).
 * @height: Image height (number of rows).
 * @step: The number of bytes per row in the image.
 * @threshold: Pixel intensity threshold (only pixels > threshold are considered).
 * @min_area: Minimum number of pixels for a valid blob.
 * @max_area: Maximum number of pixels for a valid blob.
 * @blobs: Array (allocated by caller) to be filled with detected blobs.
 * @max_blobs: Maximum number of blobs to return.
 *
 * Returns the number of blobs detected.
 */
int detect_blobs(const unsigned char* gray, int width, int height, int step,
                 int threshold, int min_area, int max_area, Blob* blobs, int max_blobs);

#ifdef __cplusplus
}
#endif

#endif // TRACKER_C_H