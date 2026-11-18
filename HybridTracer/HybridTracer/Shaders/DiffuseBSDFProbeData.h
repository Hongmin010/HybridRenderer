#pragma once
#include "DiffuseBSDFTypes.h"

// Diagnostic ABI, not a new per-frame rendering pass.
namespace HybridBSDF
{
    struct DiffuseProbeInput
    {
        Vector3 Normal; float U1;
        Vector3 Outgoing; float U2;
        Vector3 Albedo; unsigned int Padding;
        Vector3 Environment; unsigned int EnvironmentPadding;
    };
    struct DiffuseProbeOutput
    {
        DiffuseSample Sample;
        Vector3 BRDF; float EvaluatedPdf;
        Vector3 ExplicitWeight; float CosTheta;
        Vector3 ConstantRadiance; float BacksidePdf;
        Vector3 ExplicitRadiance; float BacksideBRDFMax;
        Vector3 DirectionalRadiance; unsigned int Padding;
    };
    static_assert(sizeof(DiffuseProbeInput) == 64, "DiffuseProbeInput ABI");
    static_assert(sizeof(DiffuseProbeOutput) == 112, "DiffuseProbeOutput ABI");
    struct DiffuseEvalInput
    {
        Vector3 Normal; unsigned int Padding0;
        Vector3 Incoming; unsigned int Padding1;
        Vector3 Outgoing; unsigned int Padding2;
        Vector3 Albedo; unsigned int Padding3;
    };
    struct DiffuseEvalOutput { Vector3 BRDF; float Pdf; };
    static_assert(sizeof(DiffuseEvalInput) == 64, "DiffuseEvalInput ABI");
    static_assert(sizeof(DiffuseEvalOutput) == 16, "DiffuseEvalOutput ABI");
}
