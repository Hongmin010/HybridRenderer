#include "Mesh.h"

#include <cstring>
#include <stdexcept>

namespace
{
    void CheckHR(HRESULT hr)
    {
        if (FAILED(hr))
        {
            throw std::runtime_error(
                "Mesh D3D12 call failed.");
        }
    }
}

void Mesh::Initialize(
    ID3D12Device *device,
    const std::vector<Vertex> &vertices,
    const std::vector<std::uint16_t> &indices)
{
    mVertices = vertices;
    mIndices = indices;

    const UINT vertexBufferSize =
        static_cast<UINT>(
            vertices.size() *
            sizeof(Vertex));

    const UINT indexBufferSize =
        static_cast<UINT>(
            indices.size() *
            sizeof(std::uint16_t));

    mIndexCount =
        static_cast<UINT>(
            indices.size());

    D3D12_HEAP_PROPERTIES uploadHeap = {};

    uploadHeap.Type =
        D3D12_HEAP_TYPE_UPLOAD;

    uploadHeap.CreationNodeMask = 1;
    uploadHeap.VisibleNodeMask = 1;

    //
    // Vertex Buffer
    //

    D3D12_RESOURCE_DESC vertexDesc = {};

    vertexDesc.Dimension =
        D3D12_RESOURCE_DIMENSION_BUFFER;

    vertexDesc.Width =
        vertexBufferSize;

    vertexDesc.Height = 1;
    vertexDesc.DepthOrArraySize = 1;
    vertexDesc.MipLevels = 1;

    vertexDesc.SampleDesc.Count = 1;

    vertexDesc.Layout =
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    CheckHR(
        device->CreateCommittedResource(
            &uploadHeap,
            D3D12_HEAP_FLAG_NONE,
            &vertexDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&mVertexBuffer)));

    void *mappedData = nullptr;

    D3D12_RANGE readRange =
        {
            0,
            0};

    CheckHR(
        mVertexBuffer->Map(
            0,
            &readRange,
            &mappedData));

    std::memcpy(
        mappedData,
        vertices.data(),
        vertexBufferSize);

    mVertexBuffer->Unmap(
        0,
        nullptr);

    mVertexBufferView.BufferLocation =
        mVertexBuffer->GetGPUVirtualAddress();

    mVertexBufferView.SizeInBytes =
        vertexBufferSize;

    mVertexBufferView.StrideInBytes =
        sizeof(Vertex);

    //
    // Index Buffer
    //

    D3D12_RESOURCE_DESC indexDesc = {};

    indexDesc.Dimension =
        D3D12_RESOURCE_DIMENSION_BUFFER;

    indexDesc.Width =
        indexBufferSize;

    indexDesc.Height = 1;
    indexDesc.DepthOrArraySize = 1;
    indexDesc.MipLevels = 1;

    indexDesc.SampleDesc.Count = 1;

    indexDesc.Layout =
        D3D12_TEXTURE_LAYOUT_ROW_MAJOR;

    CheckHR(
        device->CreateCommittedResource(
            &uploadHeap,
            D3D12_HEAP_FLAG_NONE,
            &indexDesc,
            D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr,
            IID_PPV_ARGS(&mIndexBuffer)));

    mappedData = nullptr;

    CheckHR(
        mIndexBuffer->Map(
            0,
            &readRange,
            &mappedData));

    std::memcpy(
        mappedData,
        indices.data(),
        indexBufferSize);

    mIndexBuffer->Unmap(
        0,
        nullptr);

    mIndexBufferView.BufferLocation =
        mIndexBuffer->GetGPUVirtualAddress();

    mIndexBufferView.SizeInBytes =
        indexBufferSize;

    mIndexBufferView.Format =
        DXGI_FORMAT_R16_UINT;
}
