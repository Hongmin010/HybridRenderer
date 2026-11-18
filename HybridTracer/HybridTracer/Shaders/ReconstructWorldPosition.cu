#include <cuda_fp16.h>

// D3D viewport: origin (0,0), depth [0,1], one sample at each pixel center.
// inverseViewProjection is a row-major DirectXMath matrix, without HLSL transpose.
extern "C" __global__ void ReconstructWorldPosition(
    const float *depth, const float *inverseViewProjection, float *positions,
    unsigned int width, unsigned int height)
{
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const unsigned int index = y * width + x;
    float *output = positions + index * 4;
    output[0] = output[1] = output[2] = output[3] = 0.0f;
    const float d = depth[index];
    // Clear depth 1 is background; the current LESS depth test excludes far-plane hits.
    if (!(d >= 0.0f && d < 1.0f)) return;

    const float nx = 2.0f * (static_cast<float>(x) + 0.5f) / width - 1.0f;
    const float ny = 1.0f - 2.0f * (static_cast<float>(y) + 0.5f) / height;
    const float *m = inverseViewProjection;
    const float hx = nx*m[0] + ny*m[4] + d*m[8]  + m[12];
    const float hy = nx*m[1] + ny*m[5] + d*m[9]  + m[13];
    const float hz = nx*m[2] + ny*m[6] + d*m[10] + m[14];
    const float hw = nx*m[3] + ny*m[7] + d*m[11] + m[15];
    if (hw == 0.0f) return;
    output[0] = hx / hw;
    output[1] = hy / hw;
    output[2] = hz / hw;
    output[3] = 1.0f; // Surface-valid flag, after perspective divide.
}

// Output ABI: position/valid float4, normalized normal/padding float4, albedo float4.
extern "C" __global__ void BuildSurfaceInputs(
    const float *positions, const __half *normals, const unsigned char *albedo,
    float *surface, unsigned int width, unsigned int height)
{
    const unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;
    const unsigned int index = y*width+x;
    float *output = surface+index*12;
    for (unsigned int c = 0; c < 12; ++c) output[c] = 0.0f;
    const float *position = positions+index*4;
    if (position[3] != 1.0f) return;
    const float nx = __half2float(normals[index*4]);
    const float ny = __half2float(normals[index*4+1]);
    const float nz = __half2float(normals[index*4+2]);
    const float lengthSquared = nx*nx + ny*ny + nz*nz;
    if (!(lengthSquared > 0.0f)) return;
    for (unsigned int c = 0; c < 4; ++c) output[c] = position[c];
    const float inverseLength = rsqrtf(lengthSquared);
    output[4] = nx*inverseLength;
    output[5] = ny*inverseLength;
    output[6] = nz*inverseLength;
    // RGBA8_UNORM is stored as bytes. This copy does not apply sRGB conversion.
    for (unsigned int c = 0; c < 4; ++c)
        output[8+c] = static_cast<float>(albedo[index*4+c])/255.0f;
}
