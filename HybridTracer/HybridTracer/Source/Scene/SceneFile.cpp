#include "SceneFile.h"
#include "Camera.h"
#include <Windows.h>
#include <shlwapi.h>
#include <xmllite.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <utility>

using DirectX::XMFLOAT3;
using namespace DirectX;
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

    struct Element
    {
        std::wstring Tag;
        Attributes Fields;
        std::vector<Element> Children;
    };

    bool Ignore(XmlNodeType type)
    {
        return type == XmlNodeType_Whitespace || type == XmlNodeType_XmlDeclaration || type == XmlNodeType_Comment;
    }

    Element ReadSubtree(IXmlReader &reader, std::wstring tag, Attributes fields)
    {
        Element element{std::move(tag),std::move(fields),{}};
        if (reader.IsEmptyElement()) return element;
        XmlNodeType type;
        HRESULT result;
        while ((result = reader.Read(&type)) == S_OK)
        {
            if (type == XmlNodeType_EndElement) return element;
            if (Ignore(type)) continue;
            if (type != XmlNodeType_Element) throw std::runtime_error("Expected a Mitsuba XML element.");
            auto childTag = Name(reader);
            auto childFields = ReadAttributes(reader);
            element.Children.push_back(ReadSubtree(reader,std::move(childTag),std::move(childFields)));
        }
        Check(result,"Malformed Mitsuba XML.");
        throw std::runtime_error("Unexpected end of Mitsuba XML.");
    }

    const std::wstring &Attribute(const Element &element, const wchar_t *name)
    {
        const auto found = element.Fields.find(name);
        if (found == element.Fields.end())
            throw std::runtime_error("Missing Mitsuba XML attribute '" + Label(name) + "'.");
        return found->second;
    }

    const Element *Child(const Element &element, const wchar_t *tag, const wchar_t *parameter = nullptr)
    {
        const auto found = std::find_if(element.Children.begin(), element.Children.end(), [&](const Element &child)
        {
            if (child.Tag != tag) return false;
            if (!parameter) return true;
            const auto name = child.Fields.find(L"name");
            return name != child.Fields.end() && name->second == parameter;
        });
        return found == element.Children.end() ? nullptr : &*found;
    }

    const Element &RequiredChild(const Element &element, const wchar_t *tag, const wchar_t *parameter = nullptr)
    {
        if (const auto *child = Child(element,tag,parameter)) return *child;
        throw std::runtime_error("Missing Mitsuba scene element '" + Label(parameter ? parameter : tag) + "'.");
    }

    template<size_t N>
    std::array<float,N> Numbers(std::wstring text)
    {
        std::replace(text.begin(),text.end(),L',',L' ');
        std::wistringstream stream(text);
        stream.imbue(std::locale::classic());
        std::array<float,N> values;
        for (float &value : values)
            if (!(stream >> value) || !std::isfinite(value))
                throw std::runtime_error("Invalid numeric value in Mitsuba XML.");
        wchar_t leftover;
        if (stream >> leftover) throw std::runtime_error("Unexpected extra number in Mitsuba XML.");
        return values;
    }

    XMMATRIX ReadMitsubaTransform(const Element &element)
    {
        const auto *transform = Child(element,L"transform",L"to_world");
        if (!transform) return XMMatrixIdentity();
        if (transform->Children.size() != 1 || transform->Children.front().Tag != L"matrix")
            throw std::runtime_error("Mitsuba to_world requires a single affine matrix.");
        const auto values = Numbers<16>(Attribute(transform->Children.front(),L"value"));
        if (values[12] != 0 || values[13] != 0 || values[14] != 0 || values[15] != 1)
            throw std::runtime_error("Mitsuba to_world matrix must be affine.");
        const XMFLOAT4X4 packed(values.data());
        const XMMATRIX matrix = XMMatrixTranspose(XMLoadFloat4x4(&packed));
        const float determinant = XMVectorGetX(XMMatrixDeterminant(matrix));
        if (!std::isfinite(determinant) || determinant <= 0)
            throw std::runtime_error("Mitsuba to_world needs a positive determinant.");
        return matrix;
    }

    XMMATRIX ReflectZ() { return XMMatrixScaling(1,1,-1); }

    XMFLOAT3 Float3(FXMVECTOR vector)
    {
        XMFLOAT3 value;
        XMStoreFloat3(&value,vector);
        return value;
    }

    SceneObjectDescription MitsubaShape(const Element &shape, const std::map<std::wstring,Material> &materials,
        std::vector<SceneLight> &lights)
    {
        const auto &type = Attribute(shape,L"type");
        if (type != L"cube" && type != L"rectangle")
            throw std::runtime_error("Supported Mitsuba shapes are cube and rectangle.");
        for (const auto &child : shape.Children)
            if (child.Tag != L"transform" && child.Tag != L"ref" && child.Tag != L"emitter")
                throw std::runtime_error("Expected a Mitsuba transform, material reference, or emitter.");
        const auto material = materials.find(Attribute(RequiredChild(shape,L"ref"),L"id"));
        if (material == materials.end()) throw std::runtime_error("Unknown Mitsuba material ID.");
        const bool cube = type == L"cube";
        SceneObjectDescription result{cube ? SceneShape::Cube : SceneShape::Plane};
        result.Albedo = material->second.Albedo;
        result.TwoSided = material->second.TwoSided;
        // Our square lies in XZ with +Y normal. Change both local and world handedness
        // so triangle winding and inverse-transformed normals still agree.
        const XMMATRIX local = cube ? ReflectZ() : XMMATRIX(1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,1);
        const XMMATRIX world = local*ReadMitsubaTransform(shape)*ReflectZ();
        result.World.emplace();
        XMStoreFloat4x4(&*result.World,world);
        if (const auto *emitter = Child(shape,L"emitter"))
        {
            if (cube || Attribute(*emitter,L"type") != L"area")
                throw std::runtime_error("Mitsuba emitters must be rectangular area lights.");
            const auto rgb = Numbers<3>(Attribute(RequiredChild(*emitter,L"rgb",L"radiance"),L"value"));
            if (std::any_of(rgb.begin(),rgb.end(),[](float value) { return value < 0; }))
                throw std::runtime_error("Mitsuba emitter radiance cannot be negative.");
            lights.emplace_back(RectangleLight{
                Float3(world.r[3]),Float3(world.r[0]),Float3(XMVectorNegate(world.r[2])),{rgb[0],rgb[1],rgb[2]},1
            });
            result.EmitterLight = static_cast<unsigned int>(lights.size());
        }
        return result;
    }

    Material MitsubaMaterial(const Element &bsdf)
    {
        Material material;
        material.TwoSided = Attribute(bsdf,L"type") == L"twosided";
        const auto &base = material.TwoSided ? RequiredChild(bsdf,L"bsdf") : bsdf;
        if (Attribute(base,L"type") != L"diffuse" || base.Children.size() != 1)
            throw std::runtime_error("Supported Mitsuba materials are constant RGB diffuse and twosided diffuse.");
        const auto rgb = Numbers<3>(Attribute(RequiredChild(base,L"rgb",L"reflectance"),L"value"));
        if (std::any_of(rgb.begin(),rgb.end(),[](float value) { return value < 0 || value > 1; }))
            throw std::runtime_error("Mitsuba diffuse reflectance must be in [0, 1].");
        material.Albedo = {rgb[0],rgb[1],rgb[2]};
        return material;
    }

    SceneCamera MitsubaCamera(const Element &sensor, const Attributes &defaults)
    {
        if (Attribute(sensor,L"type") != L"perspective")
            throw std::runtime_error("Only a perspective Mitsuba sensor is supported.");
        const auto &film = RequiredChild(sensor,L"film");
        const auto extent = [&](const wchar_t *name)
        {
            const auto &text = Attribute(RequiredChild(film,L"integer",name),L"value");
            const auto entry = !text.empty() && text[0] == L'$' ? defaults.find(text.substr(1)) : defaults.end();
            const auto size = Numbers<1>(entry == defaults.end() ? text : entry->second)[0];
            if (size < 1 || size > 4096 || std::floor(size) != size)
                throw std::runtime_error("Mitsuba film dimensions must be integers in 1..4096.");
            return static_cast<unsigned int>(size);
        };
        const auto control = [&](const wchar_t *name, float fallback)
        {
            const auto *value = Child(sensor,L"float",name);
            return value ? Numbers<1>(Attribute(*value,L"value"))[0] : fallback;
        };
        const float degrees = Numbers<1>(Attribute(RequiredChild(sensor,L"float",L"fov"),L"value"))[0];
        const float nearClip = control(L"near_clip",0.01f), farClip = control(L"far_clip",10000);
        if (degrees <= 0 || degrees >= 180 || nearClip <= 0 || farClip <= nearClip)
            throw std::runtime_error("Invalid Mitsuba FOV or clip distances.");
        const auto *axis = Child(sensor,L"string",L"fov_axis");
        const auto axisName = axis ? Attribute(*axis,L"value") : L"x";
        if (axisName != L"x" && axisName != L"y") throw std::runtime_error("Mitsuba FOV axis must be x or y.");
        const XMMATRIX world = ReadMitsubaTransform(sensor)*ReflectZ();
        return {Float3(world.r[3]),Float3(XMVector3Normalize(world.r[2])),Float3(XMVector3Normalize(world.r[1])),
            XMConvertToRadians(degrees),nearClip,farClip,axisName == L"x",extent(L"width"),extent(L"height")};
    }

    SceneDescription ReadMitsuba(IXmlReader &reader, Attributes fields)
    {
        if (fields.at(L"version").rfind(L"3.",0) != 0)
            throw std::runtime_error("Expected a Mitsuba 3 scene version.");
        const auto document = ReadSubtree(reader,L"scene",std::move(fields));
        XmlNodeType type;
        HRESULT status;
        while ((status = reader.Read(&type)) == S_OK)
            if (!Ignore(type)) throw std::runtime_error("Unexpected content following the Mitsuba scene.");
        Check(status,"Malformed Mitsuba XML.");
        Attributes defaults;
        std::map<std::wstring,Material> materials;
        for (const auto &element : document.Children)
        {
            if (element.Tag == L"default") defaults.emplace(Attribute(element,L"name"),Attribute(element,L"value"));
            else if (element.Tag == L"bsdf")
            {
                if (!materials.emplace(Attribute(element,L"id"),MitsubaMaterial(element)).second)
                    throw std::runtime_error("Duplicate Mitsuba material ID.");
            }
            else if (element.Tag != L"shape" && element.Tag != L"sensor" && element.Tag != L"integrator")
                throw std::runtime_error("Unsupported top-level Mitsuba scene element.");
        }
        SceneDescription scene;
        scene.Camera = MitsubaCamera(RequiredChild(document,L"sensor"),defaults);
        for (const auto &element : document.Children)
            if (element.Tag == L"shape") scene.Objects.push_back(MitsubaShape(element,materials,scene.Lights));
        if (scene.Objects.empty()) throw std::runtime_error("A Mitsuba scene needs at least one shape.");
        return scene;
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
                if (fields.count(L"version")) return ReadMitsuba(reader,std::move(fields));
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

float SceneCamera::VerticalFov(float aspect) const
{
    return HorizontalFov ? 2*std::atan(std::tan(Fov/2)/aspect) : Fov;
}

void SceneCamera::Apply(::Camera &camera, unsigned int width, unsigned int height, float xOffset) const
{
    const float aspect = float(width)/height;
    camera.SetLens(VerticalFov(aspect),aspect,Near,Far);
    const XMFLOAT3 eye{Position.x+xOffset,Position.y,Position.z};
    camera.LookAt(eye,{eye.x+Forward.x,eye.y+Forward.y,eye.z+Forward.z},Up);
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
