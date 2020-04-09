/***************************************************************************
 # Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
 #
 # Redistribution and use in source and binary forms, with or without
 # modification, are permitted provided that the following conditions
 # are met:
 #  * Redistributions of source code must retain the above copyright
 #    notice, this list of conditions and the following disclaimer.
 #  * Redistributions in binary form must reproduce the above copyright
 #    notice, this list of conditions and the following disclaimer in the
 #    documentation and/or other materials provided with the distribution.
 #  * Neither the name of NVIDIA CORPORATION nor the names of its
 #    contributors may be used to endorse or promote products derived
 #    from this software without specific prior written permission.
 #
 # THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 # EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 # IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 # PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 # CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 # EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 # PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 # PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 # OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 # (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 # OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 **************************************************************************/
#include "stdafx.h"
#include "SceneBuilder.h"
#include "../Externals/mikktspace/mikktspace.h"
#include <filesystem>

#define AUTOMATIC_YARN_SEGMENTATION

namespace Falcor
{
    namespace
    {
        class MikkTSpaceWrapper
        {
        public:
            static std::vector<vec3> generateBitangents(const vec3* pPositions, const vec3* pNormals, const vec2* pTexCrd, const uint32_t* pIndices, size_t vertexCount, size_t indexCount)
            {
                if (!pNormals || !pPositions || !pTexCrd || !pIndices)
                {
                    logWarning("Can't generate tangent space. The mesh doesn't have positions/normals/texCrd/indices");
                    return std::vector<vec3>(vertexCount, vec3(0, 0, 0));
                }

                SMikkTSpaceInterface mikktspace = {};
                mikktspace.m_getNumFaces = [](const SMikkTSpaceContext* pContext) {return ((MikkTSpaceWrapper*)(pContext->m_pUserData))->getFaceCount(); };
                mikktspace.m_getNumVerticesOfFace = [](const SMikkTSpaceContext * pContext, int32_t face) {return 3; };
                mikktspace.m_getPosition = [](const SMikkTSpaceContext * pContext, float position[], int32_t face, int32_t vert) {((MikkTSpaceWrapper*)(pContext->m_pUserData))->getPosition(position, face, vert); };
                mikktspace.m_getNormal = [](const SMikkTSpaceContext * pContext, float normal[], int32_t face, int32_t vert) {((MikkTSpaceWrapper*)(pContext->m_pUserData))->getNormal(normal, face, vert); };
                mikktspace.m_getTexCoord = [](const SMikkTSpaceContext * pContext, float texCrd[], int32_t face, int32_t vert) {((MikkTSpaceWrapper*)(pContext->m_pUserData))->getTexCrd(texCrd, face, vert); };
                mikktspace.m_setTSpaceBasic = [](const SMikkTSpaceContext * pContext, const float tangent[], float sign, int32_t face, int32_t vert) {((MikkTSpaceWrapper*)(pContext->m_pUserData))->setTangent(tangent, sign, face, vert); };

                MikkTSpaceWrapper wrapper(pPositions, pNormals, pTexCrd, pIndices, vertexCount, indexCount);
                SMikkTSpaceContext context = {};
                context.m_pInterface = &mikktspace;
                context.m_pUserData = &wrapper;

                if (genTangSpaceDefault(&context) == false)
                {
                    logError("Failed to generate MikkTSpace tangents");
                    return std::vector<vec3>(vertexCount, vec3(0, 0, 0));
                }

                return wrapper.mBitangents;
            }

        private:
            MikkTSpaceWrapper(const vec3* pPositions, const vec3* pNormals, const vec2* pTexCrd, const uint32_t* pIndices, size_t vertexCount, size_t indexCount) :
                mpPositions(pPositions), mpNormals(pNormals), mpTexCrd(pTexCrd), mpIndices(pIndices), mFaceCount(indexCount / 3), mBitangents(vertexCount) {}
            const vec3* mpPositions;
            const vec3* mpNormals;
            const vec2* mpTexCrd;
            const uint32_t* mpIndices;
            size_t mFaceCount;
            std::vector<vec3> mBitangents;
            int32_t getFaceCount() const { return (int32_t)mFaceCount; }
            int32_t getIndex(int32_t face, int32_t vert) { return mpIndices[face * 3 + vert]; }
            void getPosition(float position[], int32_t face, int32_t vert) { *(vec3*)position = mpPositions[getIndex(face, vert)]; }
            void getNormal(float normal[], int32_t face, int32_t vert) { *(vec3*)normal = mpNormals[getIndex(face, vert)]; }
            void getTexCrd(float texCrd[], int32_t face, int32_t vert) { *(vec2*)texCrd = mpTexCrd[getIndex(face, vert)]; }

            void setTangent(const float tangent[], float sign, int32_t face, int32_t vert)
            {
                int32_t index = getIndex(face, vert);
                vec3 T(*(vec3*)tangent), N;
                getNormal(&N[0], face, vert);
                // bitangent = fSign * cross(vN, tangent);
                mBitangents[index] = cross(N, T); // Not using fSign because... I don't know why. It flips the tangent space. Need to go read the paper
            }
        };

        void validateTangentSpace(const vec3 bitangents[], uint32_t vertexCount)
        {
            auto isValid = [](const vec3& bitangent)
            {
                if (glm::any(glm::isinf(bitangent) || glm::isnan(bitangent))) return false;
                if (length(bitangent) < 1e-6f) return false;
                return true;
            };

            uint32_t numInvalid = 0;
            for (uint32_t i = 0; i < vertexCount; i++)
            {
                if (!isValid(bitangents[i])) numInvalid++;
            }

            if (numInvalid > 0)
            {
                logWarning("Loaded tangent space is invalid at " + std::to_string(numInvalid) + " vertices. Please fix the asset.");
            }
        }
    }

    SceneBuilder::SceneBuilder(Flags flags) : mFlags(flags) {};

    SceneBuilder::SharedPtr SceneBuilder::create(Flags flags)
    {
        return SharedPtr(new SceneBuilder(flags));
    }

    SceneBuilder::SharedPtr SceneBuilder::create(const std::string& filename, Flags buildFlags, const InstanceMatrices& instances)
    {
        auto pBuilder = create(buildFlags);
        return pBuilder->import(filename == "" ? "default.obj" : filename, instances) ? pBuilder : nullptr;
    }

    bool SceneBuilder::import(const std::string& filename, const InstanceMatrices& instances)
    {
        bool success = false;
        if (std::filesystem::path(filename).extension() == ".py")
        {
            success = PythonImporter::import(filename, *this);
        }
        else if (std::filesystem::path(filename).extension() == ".fscene")
        {
            success = SceneImporter::import(filename, *this);
        }
        else
        {
            success = AssimpImporter::import(filename, *this, instances);
        }
        mFilename = filename;
        return success;
    }

    size_t SceneBuilder::addNode(const Node& node)
    {
        assert(node.parent == kInvalidNode || node.parent < mSceneGraph.size());

        size_t newNodeID = mSceneGraph.size();
        assert(newNodeID <= UINT32_MAX);
        mSceneGraph.push_back(InternalNode(node));
        if(node.parent != kInvalidNode) mSceneGraph[node.parent].children.push_back(newNodeID);
        mDirty = true;
        return newNodeID;
    }

    void SceneBuilder::addMeshInstance(size_t nodeID, size_t meshID)
    {
        assert(meshID < mMeshes.size());
        mSceneGraph.at(nodeID).meshes.push_back(meshID);
        mMeshes.at(meshID).instances.push_back((uint32_t)nodeID);
        mDirty = true;
    }

    size_t SceneBuilder::addMesh(const Mesh& mesh)
    {
        assert(mesh.pLightMapUVs == nullptr);
        const auto& prevMesh = mMeshes.size() ? mMeshes.back() : MeshSpec();

        // Create the new mesh spec
        mMeshes.push_back({});
        MeshSpec& spec = mMeshes.back();
        assert(mBuffersData.staticData.size() <= UINT32_MAX && mBuffersData.dynamicData.size() <= UINT32_MAX && mBuffersData.indices.size() <= UINT32_MAX);
        spec.staticVertexOffset = (uint32_t)mBuffersData.staticData.size();
        spec.dynamicVertexOffset = (uint32_t)mBuffersData.dynamicData.size();
        spec.indexOffset = (uint32_t)mBuffersData.indices.size();
        spec.indexCount = mesh.indexCount;
        spec.vertexCount = mesh.vertexCount;
        spec.topology = mesh.topology;
        spec.materialId = addMaterial(mesh.pMaterial, is_set(mFlags, Flags::RemoveDuplicateMaterials));

        // Error checking
        auto throw_on_missing_element = [&](const std::string& element)
        {
            throw std::runtime_error("Error when adding the mesh " + mesh.name + " to the scene.\nThe mesh is missing " + element);
        };

        auto missing_element_warning = [&](const std::string& element)
        {
            logWarning("The mesh " + mesh.name + " is missing the element " + element + ". This is not an error, the element will be filled with zeros which may result in incorrect rendering");
        };

        // Initialize the static data
        if (mesh.indexCount == 0 || !mesh.pIndices) throw_on_missing_element("indices");
        mBuffersData.indices.insert(mBuffersData.indices.end(), mesh.pIndices, mesh.pIndices + mesh.indexCount);

        if (mesh.vertexCount == 0) throw_on_missing_element("vertices");
        if (mesh.pPositions == nullptr) throw_on_missing_element("positions");
        if (mesh.pNormals == nullptr) missing_element_warning("normals");
        if (mesh.pTexCrd == nullptr) missing_element_warning("texture coordinates");

        // Initialize the dynamic data
        if (mesh.pBoneWeights || mesh.pBoneIDs)
        {
            if (mesh.pBoneIDs == nullptr) throw_on_missing_element("bone IDs");
            if (mesh.pBoneWeights == nullptr) throw_on_missing_element("bone weights");
            spec.hasDynamicData = true;
        }

        // Generate tangent space if that's required
        std::vector<vec3> bitangents;
        if (!is_set(mFlags, Flags::UseOriginalTangentSpace) || !mesh.pBitangents)
        {
            bitangents = MikkTSpaceWrapper::generateBitangents(mesh.pPositions, mesh.pNormals, mesh.pTexCrd, mesh.pIndices, mesh.vertexCount, mesh.indexCount);
        }
        else
        {
            validateTangentSpace(mesh.pBitangents, mesh.vertexCount);
        }

        for (uint32_t v = 0; v < mesh.vertexCount; v++)
        {
            StaticVertexData s;
            s.position = mesh.pPositions[v];
            s.normal = mesh.pNormals ? mesh.pNormals[v] : vec3(0, 0, 0);
            s.texCrd = mesh.pTexCrd ? mesh.pTexCrd[v] : vec2(0, 0);
            s.bitangent = bitangents.size() ? bitangents[v] : mesh.pBitangents[v];
            s.prevPosition = s.position;
            mBuffersData.staticData.push_back(s);

            if (mesh.pBoneWeights)
            {
                DynamicVertexData d;
                d.boneWeight = mesh.pBoneWeights[v];
                d.boneID = mesh.pBoneIDs[v];
                d.staticIndex = (uint32_t)mBuffersData.staticData.size() - 1;
                mBuffersData.dynamicData.push_back(d);
            }

//             if (mesh.pLightMapUVs)
//             {
//                 spec.optionalData[v].lightmapUV = mesh.pLightMapUVs[v];
//             }
        }

        mDirty = true;

        return mMeshes.size() - 1;
    }

    uint32_t SceneBuilder::addMaterial(const Material::SharedPtr& pMaterial, bool removeDuplicate)
    {
        assert(pMaterial);

        // Reuse previously added materials
        if (auto it = std::find(mMaterials.begin(), mMaterials.end(), pMaterial); it != mMaterials.end())
        {
            return (uint32_t)std::distance(mMaterials.begin(), it);
        }

        // Try to find previously added material with equal properties (duplicate)
        if (auto it = std::find_if(mMaterials.begin(), mMaterials.end(), [&pMaterial] (const auto& m) { return *m == *pMaterial; }); it != mMaterials.end())
        {
            const auto& equalMaterial = *it;

            // ASSIMP sometimes creates internal copies of a material: Always de-duplicate if name and properties are equal.
            if (removeDuplicate || pMaterial->getName() == equalMaterial->getName())
            {
                return (uint32_t)std::distance(mMaterials.begin(), it);
            }
            else
            {
                logWarning("Material '" + pMaterial->getName() + "' is a duplicate (has equal properties) of material '" + equalMaterial->getName() + "'.");
            }
        }

        mMaterials.push_back(pMaterial);
        assert(mMaterials.size() <= UINT32_MAX);
        mDirty = true;
        return (uint32_t)mMaterials.size() - 1;
    }

    void SceneBuilder::setCamera(const Camera::SharedPtr& pCamera, size_t nodeID)
    {
        mCamera.nodeID = nodeID;
        mCamera.pObject = pCamera;
        mDirty = true;
    }

    size_t SceneBuilder::addLight(const Light::SharedPtr& pLight, size_t nodeID)
    {
        Scene::AnimatedObject<Light> light;
        light.pObject = pLight;
        light.nodeID = nodeID;
        mLights.push_back(light);
        mDirty = true;
        return mLights.size() - 1;
    }

    Vao::SharedPtr SceneBuilder::createVao(uint16_t drawCount)
    {
        for (auto& mesh : mMeshes) assert(mesh.topology == mMeshes[0].topology);
        size_t ibSize = sizeof(uint32_t) * mBuffersData.indices.size();
        size_t staticVbSize = sizeof(StaticVertexData) * mBuffersData.staticData.size();
        assert(ibSize <= UINT32_MAX && staticVbSize <= UINT32_MAX);
        ResourceBindFlags ibBindFlags = Resource::BindFlags::Index | ResourceBindFlags::ShaderResource;
        Buffer::SharedPtr pIB = Buffer::create((uint32_t)ibSize, ibBindFlags, Buffer::CpuAccess::None, mBuffersData.indices.data());

        // Create the static vertex data as a structured-buffer
        ResourceBindFlags vbBindFlags = ResourceBindFlags::ShaderResource | ResourceBindFlags::UnorderedAccess | ResourceBindFlags::Vertex;
        Buffer::SharedPtr pStaticBuffer = Buffer::createStructured(sizeof(StaticVertexData), (uint32_t)mBuffersData.staticData.size(), vbBindFlags);

        Vao::BufferVec pVBs(Scene::kVertexBufferCount);
        pVBs[Scene::kStaticDataBufferIndex] = pStaticBuffer;
        std::vector<uint16_t> drawIDs(drawCount);
        for (uint32_t i = 0; i < drawCount; i++) drawIDs[i] = i;
        pVBs[Scene::kDrawIdBufferIndex] = Buffer::create(drawCount*sizeof(uint16_t), ResourceBindFlags::Vertex, Buffer::CpuAccess::None, drawIDs.data());

        // The layout only initialized the static and optional data. The skinning data doesn't get passed into the vertex-shader
        VertexLayout::SharedPtr pLayout = VertexLayout::create();

        // Static data
        VertexBufferLayout::SharedPtr pStaticLayout = VertexBufferLayout::create();
        pStaticLayout->addElement(VERTEX_POSITION_NAME, offsetof(StaticVertexData, position), ResourceFormat::RGB32Float, 1, VERTEX_POSITION_LOC);
        pStaticLayout->addElement(VERTEX_NORMAL_NAME, offsetof(StaticVertexData, normal), ResourceFormat::RGB32Float, 1, VERTEX_NORMAL_LOC);
        pStaticLayout->addElement(VERTEX_BITANGENT_NAME, offsetof(StaticVertexData, bitangent), ResourceFormat::RGB32Float, 1, VERTEX_BITANGENT_LOC);
        pStaticLayout->addElement(VERTEX_TEXCOORD_NAME, offsetof(StaticVertexData, texCrd), ResourceFormat::RG32Float, 1, VERTEX_TEXCOORD_LOC);
        pStaticLayout->addElement(VERTEX_PREV_POSITION_NAME, offsetof(StaticVertexData, prevPosition), ResourceFormat::RGB32Float, 1, VERTEX_PREV_POSITION_LOC);
        pLayout->addBufferLayout(Scene::kStaticDataBufferIndex, pStaticLayout);

        // Add the draw ID layout
        VertexBufferLayout::SharedPtr pInstLayout = VertexBufferLayout::create();
        pInstLayout->addElement(INSTANCE_DRAW_ID_NAME, 0, ResourceFormat::R16Uint, 1, INSTANCE_DRAW_ID_LOC);
        pInstLayout->setInputClass(VertexBufferLayout::InputClass::PerInstanceData, 1);
        pLayout->addBufferLayout(Scene::kDrawIdBufferIndex, pInstLayout);

//         // #SCENE optional data
//         if (pVBs[sOptionalDataIndex])
//         {
//             VertexBufferLayout::SharedPtr pOptionalLayout = VertexBufferLayout::create();
//             pOptionalLayout->addElement(VERTEX_LIGHTMAP_UV_NAME, offsetof(SceneBuilder::MeshSpec::OptionalData, lightmapUV), ResourceFormat::RGB32Float, 1, VERTEX_LIGHTMAP_UV_LOC);
//             pLayout->addBufferLayout(sOptionalDataIndex, pOptionalLayout);
//         }

        Vao::SharedPtr pVao = Vao::create(mMeshes[0].topology, pLayout, pVBs, pIB, ResourceFormat::R32Uint);
        return pVao;
    }

    void SceneBuilder::createGlobalMatricesBuffer(Scene* pScene)
    {
        pScene->mSceneGraph.resize(mSceneGraph.size());

        for (uint32_t i = 0; i < mSceneGraph.size(); i++)
        {
            assert(mSceneGraph[i].parent <= UINT32_MAX);
            pScene->mSceneGraph[i] = Scene::Node( mSceneGraph[i].name, (uint32_t)mSceneGraph[i].parent, mSceneGraph[i].transform, mSceneGraph[i].localToBindPose);
        }
    }

    uint32_t SceneBuilder::createMeshData(Scene* pScene)
    {
        auto& meshData = pScene->mMeshDesc;
        auto& instanceData = pScene->mMeshInstanceData;
        meshData.resize(mMeshes.size());
        pScene->mMeshHasDynamicData.resize(mMeshes.size());

        size_t drawCount = 0;
        for (uint32_t meshID = 0; meshID < mMeshes.size(); meshID++)
        {
            // Mesh data
            const auto& mesh = mMeshes[meshID];
            meshData[meshID].materialID = mesh.materialId;
            meshData[meshID].vbOffset = mesh.staticVertexOffset;
            meshData[meshID].ibOffset = mesh.indexOffset;
            meshData[meshID].vertexCount = mesh.vertexCount;
            meshData[meshID].indexCount = mesh.indexCount;

            drawCount += mesh.instances.size();

            // Mesh instance data
            for (const auto& instance : mesh.instances)
            {
                instanceData.push_back({});
                auto& meshInstance = instanceData.back();
                meshInstance.globalMatrixID = instance;
                meshInstance.materialID = mesh.materialId;
                meshInstance.meshID = meshID;
            }

            if (mesh.hasDynamicData)
            {
                assert(mesh.instances.size() == 1);
                pScene->mMeshHasDynamicData[meshID] = true;

                for (uint32_t i = 0; i < mesh.vertexCount; i++)
                {
                    mBuffersData.dynamicData[mesh.dynamicVertexOffset + i].globalMatrixID = (uint32_t)mesh.instances[0];
                }
            }
        }
        assert(drawCount <= UINT32_MAX);
        return (uint32_t)drawCount;
    }

    uint32_t SceneBuilder::createCurveData(Scene* pScene)
    {
        auto& curveDesc = pScene->mCurveDesc;
        curveDesc.resize(mCurves.size());

        size_t drawCount = 0;
        for (uint32_t curveID = 0; curveID < mCurves.size(); curveID++)
        {
            // Mesh data
            const auto& curve = mCurves[curveID];
            curveDesc[curveID].materialID = curve.materialId;
            curveDesc[curveID].vbOffset = curve.vertexOffset;
            curveDesc[curveID].vertexCount = curve.vertexCount;

            drawCount++;
        }
        assert(drawCount <= UINT32_MAX);
        return (uint32_t)drawCount;
    }

    Scene::SharedPtr SceneBuilder::getScene()
    {
        // We cache the scene because creating it is not cheap.
        // With the PythonImporter, the scene is fetched twice, once for running
        // the scene script and another time when the scene has finished loading.
        if (mpScene && !mDirty) return mpScene;

        if (mMeshes.size() == 0)
        {
            logError("Can't build scene. No meshes were loaded");
            return nullptr;
        }
        mpScene = Scene::create();
        if (mCamera.pObject == nullptr) mCamera.pObject = Camera::create();
        mpScene->mCamera = mCamera;
        mpScene->mCameraSpeed = mCameraSpeed;
        mpScene->mLights = mLights;
        mpScene->mMaterials = mMaterials;
        mpScene->mpLightProbe = mpLightProbe;
        mpScene->mpEnvMap = mpEnvMap;
        mpScene->mFilename = mFilename;

        createGlobalMatricesBuffer(mpScene.get());
        uint32_t drawCount = createMeshData(mpScene.get());
        mpScene->mpVao = createVao(drawCount);
        mpScene->mCPUCurveVertexBuffer = mCurveData.staticData;
        createCurveData(mpScene.get());
        calculateMeshBoundingBoxes(mpScene.get());
        calculateCurvePatchBoundingBoxes(mpScene.get());
        createAnimationController(mpScene.get());
        mpScene->finalize();
        mDirty = false;

        return mpScene;
    }

    void SceneBuilder::calculateMeshBoundingBoxes(Scene* pScene)
    {
        // Calculate mesh bounding boxes
        pScene->mMeshBBs.resize(mMeshes.size());
        for (uint32_t i = 0; i < (uint32_t)mMeshes.size(); i++)
        {
            const auto& mesh = mMeshes[i];
            vec3 boxMin(FLT_MAX);
            vec3 boxMax(-FLT_MAX);

            const auto* staticData = &mBuffersData.staticData[mesh.staticVertexOffset];
            for (uint32_t v = 0; v < mesh.vertexCount; v++)
            {
                boxMin = glm::min(boxMin, staticData[v].position);
                boxMax = glm::max(boxMax, staticData[v].position);
            }

            pScene->mMeshBBs[i] = BoundingBox::fromMinMax(boxMin, boxMax);
        }
    }

    void SceneBuilder::calculateCurvePatchBoundingBoxes(Scene* pScene)
    {
        // Calculate curve patch bounding boxes
        for (uint32_t i = 0; i < (uint32_t)mCurves.size(); i++)
        {
            const auto& curve = mCurves[i];

            int numPatches = curve.vertexCount / 4;
            for (int patch = 0; patch < numPatches; patch++)
            {
                vec3 boxMin(FLT_MAX);
                vec3 boxMax(-FLT_MAX);
                float maxWidth = 0;
                for (int v = 0; v < 4; v++)
                {
                    boxMin = glm::min(boxMin, float3(mCurveData.staticData[curve.vertexOffset + 4 * patch + v].position));
                    boxMax = glm::max(boxMax, float3(mCurveData.staticData[curve.vertexOffset + 4 * patch + v].position));
                    maxWidth = max(maxWidth, mCurveData.staticData[curve.vertexOffset + 4 * patch + v].position.w);
                }
                maxWidth *= mCurveDisplayWidthMultiplier;
                // extend with width
                BoundingBox aabb = BoundingBox::fromMinMax(boxMin - 0.5f * maxWidth, boxMax + 0.5f * maxWidth);
                pScene->mCurvePatchBBs.push_back(aabb);
            }
        }
    }

    void SceneBuilder::updateCurveDisplayWidth(Scene::SharedPtr pScene, float displayWidth)
    {
        // Calculate curve patch bounding boxes
        mCurveDisplayWidthMultiplier = displayWidth;
        int count = 0;
        for (uint32_t i = 0; i < (uint32_t)mCurves.size(); i++)
        {
            const auto& curve = mCurves[i];

            int numPatches = curve.vertexCount / 4;
            for (int patch = 0; patch < numPatches; patch++)
            {
                vec3 boxMin(FLT_MAX);
                vec3 boxMax(-FLT_MAX);
                float maxWidth = 0;
                for (int v = 0; v < 4; v++)
                {
                    boxMin = glm::min(boxMin, float3(mCurveData.staticData[curve.vertexOffset + 4 * patch + v].position));
                    boxMax = glm::max(boxMax, float3(mCurveData.staticData[curve.vertexOffset + 4 * patch + v].position));
                    maxWidth = max(maxWidth, mCurveData.staticData[curve.vertexOffset + 4 * patch + v].position.w);
                }
                maxWidth *= mCurveDisplayWidthMultiplier;
                // extend with width
                pScene->mCurvePatchBBs[count++] = BoundingBox::fromMinMax(boxMin - 0.5f * maxWidth, boxMax + 0.5f * maxWidth);
            }
        }
        pScene->updateBounds();

        int globalPatchId = 0;
        std::vector<D3D12_RAYTRACING_AABB> aabbs;
        for (uint32_t curveId = 0; curveId < (uint32_t)pScene->mCurveDesc.size(); curveId++)
        {
            int numPatches = pScene->mCurveDesc[curveId].vertexCount / 4;
            // fill CPU AABB array
            for (uint32_t patchId = 0; patchId < (uint32_t)numPatches; patchId++)
            {
                D3D12_RAYTRACING_AABB aabb;
                aabb.MinX = pScene->mCurvePatchBBs[globalPatchId].getMinPos().x;
                aabb.MinY = pScene->mCurvePatchBBs[globalPatchId].getMinPos().y;
                aabb.MinZ = pScene->mCurvePatchBBs[globalPatchId].getMinPos().z;
                aabb.MaxX = pScene->mCurvePatchBBs[globalPatchId].getMaxPos().x;
                aabb.MaxY = pScene->mCurvePatchBBs[globalPatchId].getMaxPos().y;
                aabb.MaxZ = pScene->mCurvePatchBBs[globalPatchId].getMaxPos().z;
                aabbs.push_back(aabb);
                globalPatchId++;
            }
        }

        pScene->mpCurvePatchAABBBuffer->setBlob(aabbs.data(), 0, sizeof(D3D12_RAYTRACING_AABB) * aabbs.size());
        pScene->mHasCurveDisplayWidthChanged = true;
    }

    size_t SceneBuilder::addAnimation(size_t meshID, Animation::ConstSharedPtrRef pAnimation)
    {
        assert(meshID < mMeshes.size());
        mMeshes[meshID].animations.push_back(pAnimation);
        mDirty = true;
        return mMeshes[meshID].animations.size() - 1;
    }

    void SceneBuilder::createAnimationController(Scene* pScene)
    {
        pScene->mpAnimationController = AnimationController::create(pScene, mBuffersData.staticData, mBuffersData.dynamicData);
        for (uint32_t i = 0; i < mMeshes.size(); i++)
        {
            for (const auto& pAnim : mMeshes[i].animations)
            {
                pScene->mpAnimationController->addAnimation(i, pAnim);
            }
        }
    }

    // using a default width of 1
    void SceneBuilder::GenerateBezierPatchesFromKnitData(const std::vector<std::vector<vec3>>& knitData)
    {
        for (int yarnId = 0; yarnId < knitData.size(); yarnId++)
        {
            int curveId = (int)mCurves.size();
            CurveSpec spec;
            spec.topology = Vao::Topology::LineStrip;
            spec.vertexOffset = (uint32_t)mCurveData.staticData.size();
            const std::vector<vec3>& yarn = knitData[yarnId];
            int yarnSize = (int)yarn.size();
            spec.vertexCount = (yarnSize - 3) * 4; // there are yarnSize bezier patches, each with 4 vertices (TODO: remove redundant vertex storage)
            // TODO: assign curve material
            spec.materialId = -1;
            mCurves.push_back(spec);

            vec3 lastPoint = vec3(0);
            for (int i = 3; i < yarnSize; i++)
            {
                // convert B-spline patches to bezier patches
                const int i0 = min(max(0, i - 3), int(yarnSize - 1));
                const int i1 = min(max(0, i - 2), int(yarnSize - 1));
                const int i2 = min(max(0, i - 1), int(yarnSize - 1));
                const int i3 = min(max(0, i + 0), int(yarnSize - 1));

                float3 p012 = yarn[i0];
                float3 p123 = yarn[i1];
                float3 p234 = yarn[i2];
                float3 p345 = yarn[i3];

                float3 p122 = lerp(p012, p123, float3(2.f / 3.f));
                float3 p223 = lerp(p123, p234, float3(1.f / 3.f));
                float3 p233 = lerp(p123, p234, float3(2.f / 3.f));
                float3 p334 = lerp(p234, p345, float3(1.f / 3.f));

                float3 p222 = lerp(p122, p223, float3(0.5f));
                float3 p333 = lerp(p233, p334, float3(0.5f));

                CurveVertexData vertexData[4];

                vertexData[0].position = float4(p222, 1);
                vertexData[1].position = float4(p223, 1);
                vertexData[2].position = float4(p233, 1);
                vertexData[3].position = float4(p333, 1);

                mCurveData.staticData.push_back(vertexData[0]);
                mCurveData.staticData.push_back(vertexData[1]);
                mCurveData.staticData.push_back(vertexData[2]);
                mCurveData.staticData.push_back(vertexData[3]);
            }
        }
    }

    void SceneBuilder::loadKnitCCPFile(const std::string& filename, float scale, bool yz)
    {
        using KnitData = std::vector<std::vector<vec3>>;
        KnitData	knit_data;

        std::string filefullpath;
        findFileInDataDirectories(filename, filefullpath);

        FILE* fp = fopen(filefullpath.c_str(), "r");
        if (!fp) return;

        std::vector<vec3>	yarn_rnd_pts;

        class Buffer
        {
            char data[1024];
            int readLine;
        public:
            int ReadLine(FILE* fp)
            {
                char c = fgetc(fp);
                while (!feof(fp)) {
                    while (isspace(c) && (!feof(fp) || c != '\0')) c = fgetc(fp);	// skip empty space
                    if (c == '#') while (!feof(fp) && c != '\n' && c != '\r' && c != '\0') c = fgetc(fp);	// skip comment line
                    else break;
                }
                int i = 0;
                bool inspace = false;
                while (i < 1024 - 1) {
                    if (feof(fp) || c == '\n' || c == '\r' || c == '\0') break;
                    if (isspace(c)) {	// only use a single space as the space character
                        inspace = true;
                    }
                    else {
                        if (inspace) data[i++] = ' ';
                        inspace = false;
                        data[i++] = c;
                    }
                    c = fgetc(fp);
                }
                data[i] = '\0';
                readLine = i;
                return i;
            }
            char& operator[](int i) { return data[i]; }
            void ReadVertex(vec3& v) const { sscanf(data + 2, "%f %f %f", &v.x, &v.y, &v.z); }
            void ReadVertexYZ(vec3& v) const { sscanf(data + 2, "%f %f %f", &v.x, &v.z, &v.y); }
            void ReadFloat3(float f[3]) const { sscanf(data + 2, "%f %f %f", &f[0], &f[1], &f[2]); }
            void ReadFloat(float* f) const { sscanf(data + 2, "%f", f); }
            void ReadInt(int* i, int start) const { sscanf(data + start, "%d", i); }
            bool IsCommand(const char* cmd) const {
                int i = 0;
                while (cmd[i] != '\0') {
                    if (cmd[i] != data[i]) return false;
                    i++;
                }
                return (data[i] == '\0' || data[i] == ' ');
            }
            void Copy(char* a, int count, int start = 0) const {
                strncpy(a, data + start, count - 1);
                a[count - 1] = '\0';
            }
        };
        Buffer buffer;

        yarn_rnd_pts.clear();

        vec3 p;
        while (buffer.ReadLine(fp)) {
            if (buffer.IsCommand("v")) {
                if (yz) buffer.ReadVertex(p);
                else	buffer.ReadVertexYZ(p);
            }

            yarn_rnd_pts.push_back(p * scale);
            if (feof(fp)) break;
        }

        knit_data.clear();
        std::vector<vec3> yarn;
        const int subSegNum = 11;
        for (int i = 0; i < (int)yarn_rnd_pts.size() - 3; i++) {

            float localLength = 0.0;
            for (int j = 0; j < subSegNum - 1; j++) {
                float t0 = j / float(subSegNum - 1);
                float t1 = (j + 1) / float(subSegNum - 1);
                const vec3 p0 = pow(1 - t0, 3) * yarn_rnd_pts[i + 0] + 3 * pow(1 - t0, 2) * t0 * yarn_rnd_pts[i + 1] + 3 * pow(t0, 2) * (1 - t0) * yarn_rnd_pts[i + 2] + pow(t0, 3) * yarn_rnd_pts[i + 3];
                const vec3 p1 = pow(1 - t1, 3) * yarn_rnd_pts[i + 0] + 3 * pow(1 - t1, 2) * t1 * yarn_rnd_pts[i + 1] + 3 * pow(t1, 2) * (1 - t1) * yarn_rnd_pts[i + 2] + pow(t1, 3) * yarn_rnd_pts[i + 3];
                const vec3 diff = p0 - p1;
                localLength += length(diff);
            }

#ifdef AUTOMATIC_YARN_SEGMENTATION
            if (localLength > 10.0f) {
                if (yarn.size() > 3) knit_data.push_back(yarn);
                yarn.clear();
                continue;
            }
#endif

            yarn.push_back(yarn_rnd_pts[i]);
        }

        if (!yarn.empty()) knit_data.push_back(yarn);

        std::cout << "load " << knit_data.size() << " yarns\n";
        for (int yi = 0; yi < knit_data.size(); yi++)
            std::cout << "Yarn " << yi << " " << knit_data[yi].size() << std::endl;

        fclose(fp);

        GenerateBezierPatchesFromKnitData(knit_data);
    }

    SCRIPT_BINDING(SceneBuilder)
    {
        auto buildFlags = m.enum_<SceneBuilder::Flags>("SceneBuilderFlags");
        buildFlags.regEnumVal(SceneBuilder::Flags::None);
        buildFlags.regEnumVal(SceneBuilder::Flags::RemoveDuplicateMaterials);
        buildFlags.regEnumVal(SceneBuilder::Flags::UseOriginalTangentSpace);
        buildFlags.regEnumVal(SceneBuilder::Flags::AssumeLinearSpaceTextures);
        buildFlags.regEnumVal(SceneBuilder::Flags::DontMergeMeshes);
        buildFlags.regEnumVal(SceneBuilder::Flags::BuffersAsShaderResource);
        buildFlags.regEnumVal(SceneBuilder::Flags::UseSpecGlossMaterials);
        buildFlags.regEnumVal(SceneBuilder::Flags::UseMetalRoughMaterials);
        buildFlags.addBinaryOperators();
    }
}
