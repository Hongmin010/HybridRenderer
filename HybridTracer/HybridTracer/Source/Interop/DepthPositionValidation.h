#pragma once

#include "CudaDepthPositionReader.h"
#include "Scene/Camera.h"
#include "Scene/Scene.h"

#include <cstdint>
#include <ostream>
#include <vector>

// Full-image transport/reprojection checks plus independent geometric ray samples.
void ValidateDepthPositions(const float *depth, const WorldPosition *positions,
    const std::uint16_t *normalHalf, const std::vector<float> &dxDepth,
    const Camera &camera, const Scene &scene, UINT width, UINT height, UINT frame,
    std::ostream *samples, const SurfaceInput *surface = nullptr, std::ostream *surfaceSamples = nullptr);

void ValidateSurfaceInputs(const SurfaceInput *surface, const WorldPosition *positions,
    const std::uint16_t *normalHalf, const std::uint8_t *albedo,
    const std::vector<std::uint8_t> &dxAlbedo, UINT width, UINT height, UINT frame);
