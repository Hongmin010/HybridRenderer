#pragma once

namespace HybridBSDF
{
    struct Vector3 { float X, Y, Z; };

    // All directions point AWAY from the surface. PDF is per unit solid angle.
    // Weight already contains BRDF*cos/pdf. Do not apply those factors again.
    struct DiffuseSample
    {
        Vector3 Direction;
        float Pdf;
        Vector3 Weight;
        unsigned int Valid;
    };
    static_assert(sizeof(Vector3) == 12, "Vector3 ABI");
    static_assert(sizeof(DiffuseSample) == 32, "DiffuseSample ABI");
}
