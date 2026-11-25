#pragma once
#include "../../Shaders/OptixTraceShared.h"
#include <ostream>

class Scene;

// Diagnostic only: brute-force double-precision triangle tests from original Scene meshes.
void ValidateShadowResults(const ShadowResult *results, const SurfaceInput *surfaces,
    const Scene &scene, const ShadowSettings &settings, unsigned int width, unsigned int height,
    unsigned int frame, std::ostream *samples, bool checkFixture = true);
