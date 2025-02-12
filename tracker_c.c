#include "tracker_c.h"
#include <stdlib.h>
#include <string.h>

// A simple point structure for flood fill
typedef struct {
    int x;
    int y;
} Point;

int detect_blobs(const unsigned char* gray, int width, int height, int step,
                 int threshold, int min_area, int max_area, Blob* blobs, int max_blobs) {
    int num_blobs = 0;
    // Allocate a visited array (one byte per pixel)
    int size = width * height;
    char *visited = (char*)calloc(size, sizeof(char));
    if (!visited) return 0; // memory allocation failure

    // Allocate a queue for flood-fill (worst-case: every pixel)
    Point* queue = (Point*)malloc(size * sizeof(Point));
    if (!queue) {
        free(visited);
        return 0;
    }

    // Offsets for 8-connected neighbors
    int dx[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    int dy[8] = {-1, -1, -1,  0, 0,  1, 1, 1};

    // Loop over every pixel
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int index = y * width + x;
            if (visited[index])
                continue;
            unsigned char pixel = gray[y * step + x];
            if (pixel <= threshold) {
                visited[index] = 1; // mark as visited even if below threshold
                continue;
            }

            // Start a new blob using flood fill
            int queue_start = 0, queue_end = 0;
            queue[queue_end++] = (Point){ x, y };
            visited[index] = 1;
            int sum_x = 0;
            int sum_y = 0;
            int count = 0;

            while (queue_start < queue_end) {
                Point p = queue[queue_start++];
                sum_x += p.x;
                sum_y += p.y;
                count++;

                // Check all 8-connected neighbors
                for (int d = 0; d < 8; d++) {
                    int nx = p.x + dx[d];
                    int ny = p.y + dy[d];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                        continue;
                    int nindex = ny * width + nx;
                    if (visited[nindex])
                        continue;
                    unsigned char npixel = gray[ny * step + nx];
                    if (npixel > threshold) {
                        visited[nindex] = 1;
                        queue[queue_end++] = (Point){ nx, ny };
                    } else {
                        visited[nindex] = 1;
                    }
                }
            }
            // If the blob’s area is within our specified limits, record it.
            if (count >= min_area && count <= max_area) {
                if (num_blobs < max_blobs) {
                    blobs[num_blobs].x = sum_x / count;
                    blobs[num_blobs].y = sum_y / count;
                    blobs[num_blobs].area = count;
                    num_blobs++;
                }
            }
        }
    }

    free(queue);
    free(visited);
    return num_blobs;
}