struct VSOutput { float4 Position : SV_POSITION; };
VSOutput VSMain(uint vertexID : SV_VertexID)
{
    float2 positions[3] = {float2(-1,-1),float2(-1,3),float2(3,-1)};
    VSOutput output;
    output.Position = float4(positions[vertexID],0,1);
    return output;
}
