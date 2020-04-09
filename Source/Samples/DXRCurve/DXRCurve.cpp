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
#include "DXRCurve.h"
#include <stdio.h>
#include <io.h>
#include <fcntl.h>
#include <windows.h>

static const glm::vec4 kClearColor(0.38f, 0.52f, 0.10f, 1);
static const std::string kDefaultScene = "framerib_sim_13.ccp";

std::string to_string(const vec3& v)
{
    std::string s;
    s += "(" + std::to_string(v.x) + ", " + std::to_string(v.y) + ", " + std::to_string(v.z) + ")";
    return s;
}

void DXRCurve::onGuiRender(Gui* pGui)
{
    Gui::Window w(pGui, "Hello DXR Settings", { 300, 400 }, { 10, 80 });

    if (w.button("Load Curve"))
    {
        std::string filename;
        if (openFileDialog(Scene::kCurveFileExtensionFilters, filename))
        {
            loadBSplineCurveFromCCP(filename, gpFramework->getTargetFbo().get());
        }
    }

    Gui::DropdownList debugViews;
    debugViews.push_back({ 0, "World Position" });
    debugViews.push_back({ 1, "Curve u/v" });
    debugViews.push_back({ 2, "Curve tangent" });
    w.dropdown("Debug View", debugViews, mDebugViewId);

    mpScene->renderUI(w);
}

void DXRCurve::loadBSplineCurveFromCCP(const std::string& filename, const Fbo* pTargetFbo)
{
    SceneBuilder::SharedPtr pBuilder = SceneBuilder::create("");
    pBuilder->loadKnitCCPFile(filename, 10.f);
    mpScene = pBuilder->getScene();
    mpScene->setCameraController(Scene::CameraControllerType::Orbiter);
    if (!mpScene) return;

    mpCamera = mpScene->getCamera();

    // Update the controllers
    float radius = length(mpScene->getSceneBounds().extent);
    mpScene->setCameraSpeed(radius * 0.25f);
    float nearZ = std::max(0.1f, radius / 750.0f);
    float farZ = radius * 10;
    mpCamera->setDepthRange(nearZ, farZ);
    mpCamera->setAspectRatio((float)pTargetFbo->getWidth() / (float)pTargetFbo->getHeight());

    RtProgram::Desc rtProgDesc;
    rtProgDesc.addShaderLibrary("Samples/DXRCurve/DXRCurve.rt.slang").setRayGen("rayGen");
    rtProgDesc.addHitGroup(0, "primaryClosestHit", "primaryAnyHit").addMiss(0, "primaryMiss");
    rtProgDesc.addCurveHitGroup(0, "primaryCurveClosestHit", "primaryCurveAnyHit", "rayCurveIntersectionInCurveAABB");
    rtProgDesc.addDefines(mpScene->getSceneDefines());
    rtProgDesc.setMaxTraceRecursionDepth(1); 

    mpRaytraceProgram = RtProgram::create(rtProgDesc);
    mpRtVars = RtProgramVars::create(mpRaytraceProgram, mpScene);
    mpRaytraceProgram->setScene(mpScene);
}

void DXRCurve::onLoad(RenderContext* pRenderContext)
{
    if (gpDevice->isFeatureSupported(Device::SupportedFeatures::Raytracing) == false)
    {
        logFatal("Device does not support raytracing!");
    }

    loadBSplineCurveFromCCP(kDefaultScene, gpFramework->getTargetFbo().get());
}

void DXRCurve::setPerFrameVars(const Fbo* pTargetFbo)
{
    PROFILE("setPerFrameVars");
    auto cb = mpRtVars["PerFrameCB"];
    cb["viewportInvDims"] = vec2(1.f / pTargetFbo->getWidth(), 1.f / pTargetFbo->getHeight());
    cb["debugViewId"] = mDebugViewId;
    mpRtVars->getRootVar()["gOutput"] = mpRtOut;
}

void DXRCurve::renderRT(RenderContext* pContext, const Fbo* pTargetFbo)
{
    PROFILE("renderRT");
    setPerFrameVars(pTargetFbo);

    pContext->clearUAV(mpRtOut->getUAV().get(), kClearColor);
    mpScene->raytrace(pContext, mpRaytraceProgram.get(), mpRtVars, uvec3(pTargetFbo->getWidth(), pTargetFbo->getHeight(), 1));
    pContext->blit(mpRtOut->getSRV(), pTargetFbo->getRenderTargetView(0));
}

void DXRCurve::onFrameRender(RenderContext* pRenderContext, const Fbo::SharedPtr& pTargetFbo)
{
    pRenderContext->clearFbo(pTargetFbo.get(), kClearColor, 1.0f, 0, FboAttachmentType::All);

    if (mpScene)
    {
        mpScene->update(pRenderContext, gpFramework->getGlobalClock().now());
        renderRT(pRenderContext, pTargetFbo.get());
    }

    TextRenderer::render(pRenderContext, gpFramework->getFrameRate().getMsg(), pTargetFbo, { 20, 20 });
}

bool DXRCurve::onKeyEvent(const KeyboardEvent& keyEvent)
{
    if (mpScene && mpScene->onKeyEvent(keyEvent)) return true;
    return false;
}

bool DXRCurve::onMouseEvent(const MouseEvent& mouseEvent)
{
    return mpScene && mpScene->onMouseEvent(mouseEvent);
}

void DXRCurve::onResizeSwapChain(uint32_t width, uint32_t height)
{
    float h = (float)height;
    float w = (float)width;

    if (mpCamera)
    {
        mpCamera->setFocalLength(18);
        float aspectRatio = (w / h);
        mpCamera->setAspectRatio(aspectRatio);
    }

    mpRtOut = Texture::create2D(width, height, ResourceFormat::RGBA16Float, 1, 1, nullptr, Resource::BindFlags::UnorderedAccess | Resource::BindFlags::ShaderResource);
}

int wmain(int argc, wchar_t** argv)
 {
    DXRCurve::UniquePtr pRenderer = std::make_unique<DXRCurve>();
    SampleConfig config;
    config.windowDesc.title = "DXRCurve";
    config.windowDesc.resizableWindow = true;

    Sample::run(config, pRenderer);
}
