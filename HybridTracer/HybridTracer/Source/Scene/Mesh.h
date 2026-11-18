#pragma once

#include <d3d12.h>
#include <wrl.h>
#include <DirectXMath.h>

#include <cstdint>
#include <vector>

struct Vertex
{
    DirectX::XMFLOAT3 Position;
    DirectX::XMFLOAT3 Normal;
};

class Mesh
{
public:
    void Initialize(
        ID3D12Device *device,
        const std::vector<Vertex> &vertices,
        const std::vector<std::uint16_t> &indices);

    const D3D12_VERTEX_BUFFER_VIEW &
    GetVertexBufferView() const
    {
        return mVertexBufferView;
    }

    const D3D12_INDEX_BUFFER_VIEW &
    GetIndexBufferView() const
    {
        return mIndexBufferView;
    }

    UINT GetIndexCount() const
    {
        return mIndexCount;
    }

    // Retain original geometry for CPU validation and later OptiX geometry upload.
    const std::vector<Vertex> &GetVertices() const { return mVertices; }
    const std::vector<std::uint16_t> &GetIndices() const { return mIndices; }

private:
    std::vector<Vertex> mVertices;
    std::vector<std::uint16_t> mIndices;

    Microsoft::WRL::ComPtr<ID3D12Resource>
        mVertexBuffer;

    Microsoft::WRL::ComPtr<ID3D12Resource>
        mIndexBuffer;

    D3D12_VERTEX_BUFFER_VIEW
    mVertexBufferView = {};

    D3D12_INDEX_BUFFER_VIEW
    mIndexBufferView = {};

    UINT mIndexCount = 0;
};
