#include "SceneFile.h"
#include <Windows.h>
#include <shlwapi.h>
#include <xmllite.h>
#include <cmath>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

using DirectX::XMFLOAT3;
using Microsoft::WRL::ComPtr;

namespace
{
    using Attributes = std::map<std::wstring,std::wstring>;

    void Check(HRESULT result, const char *operation)
    {
        if (FAILED(result)) throw std::runtime_error(operation);
    }

    std::wstring Name(IXmlReader &reader)
    {
        const wchar_t *name = nullptr;
        Check(reader.GetQualifiedName(&name,nullptr),"Cannot read the XML element or attribute name.");
        return name;
    }

    Attributes ReadAttributes(IXmlReader &reader)
    {
        Attributes values;
        HRESULT result = reader.MoveToFirstAttribute();
        while (result == S_OK)
        {
            const wchar_t *value = nullptr;
            Check(reader.GetValue(&value,nullptr),"Cannot read the XML attribute value.");
            values.emplace(Name(reader),value);
            result = reader.MoveToNextAttribute();
        }
        Check(result,"Cannot read the XML attribute list.");
        Check(reader.MoveToElement(),"Cannot return to the XML element.");
        return values;
    }

    std::string Label(const wchar_t *text)
    {
        return std::filesystem::path(text).u8string();
    }

    std::wistringstream ReadField(Attributes &fields, const wchar_t *field)
    {
        const auto entry = fields.extract(field);
        if (entry.empty()) throw std::runtime_error("Missing attribute '" + Label(field) + "'.");
        std::wistringstream stream(entry.mapped());
        stream.imbue(std::locale::classic());
        return stream;
    }

    XMFLOAT3 Vector(Attributes &fields, const wchar_t *field)
    {
        auto stream = ReadField(fields,field);
        XMFLOAT3 value;
        wchar_t leftover;
        if (!(stream >> value.x >> value.y >> value.z) || stream >> leftover ||
            !std::isfinite(value.x) || !std::isfinite(value.y) || !std::isfinite(value.z))
            throw std::runtime_error("Attribute '" + Label(field) + "' expects three finite numbers.");
        return value;
    }

    XMFLOAT3 Vector(Attributes &fields, const wchar_t *field, XMFLOAT3 fallback)
    {
        return fields.count(field) ? Vector(fields,field) : fallback;
    }

    float Strength(Attributes &fields, const wchar_t *field)
    {
        auto stream = ReadField(fields,field);
        float value;
        wchar_t leftover;
        if (!(stream >> value) || stream >> leftover || !std::isfinite(value) || value < 0)
            throw std::runtime_error("Attribute '" + Label(field) + "' expects a finite nonnegative number.");
        return value;
    }

    void CheckAttributes(const Attributes &fields)
    {
        if (!fields.empty())
            throw std::runtime_error("Unexpected attribute '" + Label(fields.begin()->first.c_str()) + "'.");
    }

    SceneObjectDescription ReadObject(const std::wstring &name, Attributes fields)
    {
        if (name != L"cube" && name != L"plane")
            throw std::runtime_error("Expected a cube or plane in <objects>.");
        SceneObjectDescription object{name == L"cube" ? SceneShape::Cube : SceneShape::Plane};
        object.Position = Vector(fields,L"position",object.Position);
        object.Rotation = Vector(fields,L"rotation",object.Rotation);
        object.Scale = Vector(fields,L"scale",object.Scale);
        object.Albedo = Vector(fields,L"albedo",object.Albedo);
        const auto &scale = object.Scale;
        const auto &albedo = object.Albedo;
        if (scale.x <= 0 || scale.y <= 0 || scale.z <= 0)
            throw std::runtime_error("Object scale components must be positive.");
        if (albedo.x < 0 || albedo.y < 0 || albedo.z < 0 || albedo.x > 1 || albedo.y > 1 || albedo.z > 1)
            throw std::runtime_error("Object albedo components must be in [0, 1].");
        CheckAttributes(fields);
        return object;
    }

    SceneLight ReadLight(const std::wstring &name, Attributes fields)
    {
        const auto color = Vector(fields,L"color",{1,1,1});
        if (color.x < 0 || color.y < 0 || color.z < 0)
            throw std::runtime_error("Light color components must be nonnegative.");
        SceneLight light;
        if (name == L"directional")
        {
            const auto direction = Vector(fields,L"direction");
            const auto lengthSquared = direction.x*direction.x+direction.y*direction.y+direction.z*direction.z;
            if (!(lengthSquared > 0) || !std::isfinite(lengthSquared))
                throw std::runtime_error("Directional light needs a finite nonzero direction.");
            light = DirectionalLight{direction,color,Strength(fields,L"irradiance")};
        }
        else if (name == L"point")
        {
            light = PointLight{Vector(fields,L"position"),color,Strength(fields,L"intensity")};
        }
        else if (name == L"rectangle")
        {
            light = RectangleLight{Vector(fields,L"center"),Vector(fields,L"half_u"),Vector(fields,L"half_v"),
                color,Strength(fields,L"radiance")};
        }
        else
        {
            throw std::runtime_error("Expected a directional, point, or rectangle in <lights>.");
        }
        CheckAttributes(fields);
        return light;
    }

    SceneDescription ReadElements(IXmlReader &reader)
    {
        enum class Section { Objects, Lights };
        Section section = Section::Objects;
        SceneDescription scene;
        bool hasScene = false, hasObjects = false, hasLights = false;
        XmlNodeType type;
        HRESULT result;
        while ((result = reader.Read(&type)) == S_OK)
        {
            if (type == XmlNodeType_Whitespace || type == XmlNodeType_XmlDeclaration ||
                type == XmlNodeType_Comment || type == XmlNodeType_EndElement) continue;
            if (type != XmlNodeType_Element)
                throw std::runtime_error("Expected XML elements with scene attributes.");
            UINT depth = 0;
            Check(reader.GetDepth(&depth),"Cannot read the XML element depth.");
            const auto name = Name(reader);
            auto fields = ReadAttributes(reader);
            if (depth == 0 && name == L"scene" && !std::exchange(hasScene,true))
            {
                CheckAttributes(fields);
            }
            else if (depth == 1 && name == L"objects" && !std::exchange(hasObjects,true))
            {
                CheckAttributes(fields);
                section = Section::Objects;
            }
            else if (depth == 1 && name == L"lights" && !std::exchange(hasLights,true))
            {
                CheckAttributes(fields);
                section = Section::Lights;
            }
            else if (depth == 2 && section == Section::Objects)
            {
                scene.Objects.push_back(ReadObject(name,std::move(fields)));
            }
            else if (depth == 2 && section == Section::Lights)
            {
                scene.Lights.push_back(ReadLight(name,std::move(fields)));
            }
            else
            {
                throw std::runtime_error("Expected one <scene> with <objects> and <lights> sections.");
            }
        }
        Check(result,"Malformed XML.");
        if (!hasScene || !hasObjects || !hasLights || scene.Objects.empty())
            throw std::runtime_error("A scene needs at least one object and an explicit <lights> section.");
        return scene;
    }
}

void SceneDescription::Apply(Scene &scene, ID3D12Device *device) const
{
    scene.Initialize(device,Objects);
    scene.SetLights(Lights);
}

SceneDescription ReadSceneFile(const std::filesystem::path &path)
{
    ComPtr<IXmlReader> reader;
    try
    {
        ComPtr<IStream> file;
        Check(SHCreateStreamOnFileEx(path.c_str(),STGM_READ | STGM_SHARE_DENY_WRITE,0,FALSE,nullptr,&file),
            "Cannot open scene file.");
        Check(CreateXmlReader(IID_PPV_ARGS(&reader),nullptr),"Cannot initialize the XML reader.");
        Check(reader->SetProperty(XmlReaderProperty_DtdProcessing,DtdProcessing_Prohibit),"Cannot configure the XML reader.");
        Check(reader->SetInput(file.Get()),"Cannot read the scene file.");
        return ReadElements(*reader.Get());
    }
    catch (const std::exception &error)
    {
        UINT line = 0;
        if (reader) reader->GetLineNumber(&line);
        throw std::runtime_error(path.u8string() + ":" + std::to_string(line) + ": " + error.what());
    }
}

std::filesystem::path DefaultSceneFile()
{
    std::wstring executable(32768,L'\0');
    const DWORD length = GetModuleFileNameW(nullptr,executable.data(),static_cast<DWORD>(executable.size()));
    if (!length || length >= executable.size())
        throw std::runtime_error("Cannot locate the default scene beside the executable.");
    executable.resize(length);
    return std::filesystem::path(executable).parent_path()/L"Scenes"/L"Default.xml";
}
