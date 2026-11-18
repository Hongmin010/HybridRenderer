#include "DiffuseBSDF.cuh"
#include "DiffuseBSDFProbeData.h"
using namespace HybridBSDF;

extern "C" __global__ void ProbeDiffuseSamples(const DiffuseProbeInput *input, DiffuseProbeOutput *output, unsigned int count)
{
    const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    const auto query = input[i];
    DiffuseProbeOutput result = {};
    result.Sample = SampleDiffuse(query.Albedo,query.Normal,query.Outgoing,query.U1,query.U2);
    if (result.Sample.Valid)
    {
        const auto direction = result.Sample.Direction;
        result.BRDF = EvalDiffuseBRDF(query.Albedo,query.Normal,direction,query.Outgoing);
        result.EvaluatedPdf = PdfDiffuse(query.Normal,direction,query.Outgoing);
        result.CosTheta = fmaxf(Dot(Normalize(query.Normal),direction),0.0f);
        result.ExplicitWeight = Scale(result.BRDF,result.CosTheta/result.EvaluatedPdf);
        result.ConstantRadiance = {result.Sample.Weight.X*query.Environment.X,
            result.Sample.Weight.Y*query.Environment.Y,result.Sample.Weight.Z*query.Environment.Z};
        result.ExplicitRadiance = {result.ExplicitWeight.X*query.Environment.X,
            result.ExplicitWeight.Y*query.Environment.Y,result.ExplicitWeight.Z*query.Environment.Z};
        const auto reverse = Scale(direction,-1);
        result.BacksidePdf = PdfDiffuse(query.Normal,reverse,query.Outgoing);
        const auto backBRDF = EvalDiffuseBRDF(query.Albedo,query.Normal,reverse,query.Outgoing);
        result.BacksideBRDFMax = fmaxf(backBRDF.X,fmaxf(backBRDF.Y,backBRDF.Z));
        // Nonconstant world-space environment with an independently known hemisphere integral.
        result.DirectionalRadiance = {result.Sample.Weight.X*(2.0f+0.5f*direction.X),
            result.Sample.Weight.Y*(1.0f+0.3f*direction.Y),
            result.Sample.Weight.Z*(0.5f+0.25f*direction.Z*direction.Z)};
    }
    output[i] = result;
}

extern "C" __global__ void ProbeDiffuseEval(const DiffuseEvalInput *input, DiffuseEvalOutput *output, unsigned int count)
{
    const unsigned int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= count) return;
    const auto query = input[i];
    output[i] = {EvalDiffuseBRDF(query.Albedo,query.Normal,query.Incoming,query.Outgoing),
        PdfDiffuse(query.Normal,query.Incoming,query.Outgoing)};
}
