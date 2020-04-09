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
import Scene.Shading;
import Scene.Raytracing;
import Utils.Math.MathHelpers;

RWTexture2D<float4> gOutput;

static const int MaxCurveSubdivDepth = 2;

cbuffer PerFrameCB : register(b0)
{
    float2 viewportInvDims;
    int debugViewId;
};

/** Ray differentials for primary hit. Code from RayTracingGems, Chapter 20.
*/


inline float4x4 createMatrixFromBasis(float3 position, float3 forward, float3 up)
{
    float3 f = normalize(forward);
    float3 s = normalize(cross(up, forward));
    float3 u = cross(f, s);
    return float4x4(float4(s.x, u.x, f.x, 0), float4(s.y, u.y, f.y,0), float4(s.z, u.z, f.z,0), 
    float4(-s.x*position.x-s.y*position.y-s.z*position.z,
           -u.x*position.x-u.y*position.y-u.z*position.z,
           -f.x*position.x-f.y*position.y-f.z*position.z,           
           1));
}
inline float4x4 createMatrixFromLookAt(float3 position, float3 target, float3 up)
{
    return createMatrixFromBasis(position, normalize(target - position), up);
}

inline int Log2(float v)
{
    if (v < 1) return 0;
    uint bits = asuint(v);
    // https://graphics.stanford.edu/~seander/bithacks.html#IntegerLog
    // (With an additional add so get round-to-nearest rather than
    // round down.)
    return (bits >> 23) - 127 + (bits & (1 << 22) ? 1 : 0);
};

inline float3 BlossomBezier(in float3 p[4], float u0, float u1, float u2)
{
    float3 a[3] = {lerp(p[0], p[1], u0), lerp(p[1], p[2], u0),
                    lerp(p[2], p[3], u0)};
    float3 b[2] = {lerp(a[0], a[1], u1), lerp(a[1], a[2], u1)};
    return lerp(b[0], b[1], u2);
}

inline float3 EvalBezier(in float3 cp[4], float u, out float3 deriv)
{
    float3 cp1[3] = {lerp(cp[0], cp[1], u), lerp(cp[1], cp[2], u),
                      lerp(cp[2], cp[3], u)};
    float3 cp2[2] = {lerp(cp1[0], cp1[1], u), lerp(cp1[1], cp1[2], u)};

    deriv = 3 * (cp2[1] - cp2[0]);

    return lerp(cp2[0], cp2[1], u);
}

struct PrimaryRayData
{
    bool hit;
};

struct CurveIntersectionAttributes
{
    float3 dpdu;
    float u;
    float v;
};


[shader("miss")]
void primaryMiss(inout PrimaryRayData rayData)
{
}

[shader("anyhit")]
void primaryAnyHit(uniform HitShaderParams hitParams, inout PrimaryRayData rayData : SV_RayPayload, BuiltInTriangleIntersectionAttributes attribs : SV_IntersectionAttributes)
{
}

[shader("anyhit")]
void primaryCurveAnyHit(uniform HitShaderParams hitParams, inout PrimaryRayData rayData : SV_RayPayload, CurveIntersectionAttributes attribs : SV_IntersectionAttributes)
{
}


[shader("intersection")]
void rayCurveIntersectionInCurveAABB(uniform HitShaderParams hitParams)
{
    float TCurrent = RayTCurrent();
    float TMin = RayTMin();
    float TMax = 1e10;

    //CurveIntersectionAttributes attr;      
    //attr.u = 1;
    //attr.v = 1;
    //attr.dpdu = 1;
    //ReportHit(TCurrent, 0, attr);
    //return;

    const uint curveInstanceID = hitParams.geometryIndex;
    const uint patchIndex = PrimitiveIndex();
    PatchVertexData patch = gScene.getCurveVertexData(curveInstanceID, patchIndex);

    float3 rayDir = ObjectRayDirection();
    float3 rayOrigin = ObjectRayOrigin();

    // WARNING: requires p3 and p0 is not coincident
    float3 dx = cross(rayDir, patch.position[3].xyz - patch.position[0].xyz);

    float4x4 worldToRaySpaceMatrix = createMatrixFromLookAt(rayOrigin, rayOrigin + rayDir, dx);
    float3x3 raySpaceToWorldRotMatrix = transpose(float3x3(worldToRaySpaceMatrix));

    float3 cp[4] = {mul(float4(patch.position[0].xyz,1), worldToRaySpaceMatrix).xyz,
                    mul(float4(patch.position[1].xyz,1), worldToRaySpaceMatrix).xyz,
                    mul(float4(patch.position[2].xyz,1), worldToRaySpaceMatrix).xyz,
                    mul(float4(patch.position[3].xyz,1), worldToRaySpaceMatrix).xyz };

    float maxWidth = max(max(patch.position[0].w, patch.position[1].w), max(patch.position[2].w, patch.position[3].w));

    float rayLength = length(rayDir);
    float zMax = rayLength * TMax;
    if (max(max(cp[0].y, cp[1].y), max(cp[2].y, cp[3].y)) +
            0.5f * maxWidth < 0 ||
        min(min(cp[0].y, cp[1].y), min(cp[2].y, cp[3].y)) -
            0.5f * maxWidth > 0 ||
        max(max(cp[0].x, cp[1].x), max(cp[2].x, cp[3].x)) +
            0.5f * maxWidth < 0 ||
        min(min(cp[0].x, cp[1].x), min(cp[2].x, cp[3].x)) -
            0.5f * maxWidth > 0 ||
        max(max(cp[0].z, cp[1].z), max(cp[2].z, cp[3].z)) +
            0.5f * maxWidth < 0 ||
        min(min(cp[0].z, cp[1].z), min(cp[2].z, cp[3].z)) -
            0.5f * maxWidth > zMax
            )
        return;

    // CurveIntersectionAttributes attr;      
    // attr.u = maxWidth;
    // attr.v = maxWidth;
    // attr.dpdu = cp[0];
    // ReportHit(length(patch.position[0].xyz - rayOrigin) / length(rayDir), 0, attr);
    // return;

		// compute refinement depth

    float L0 = 0;
    for (int i = 0; i < 2; ++i)
        L0 = max(
            L0, max(
                max(abs(cp[i].x - 2 * cp[i + 1].x + cp[i + 2].x),
                    abs(cp[i].y - 2 * cp[i + 1].y + cp[i + 2].y)),
                abs(cp[i].z - 2 * cp[i + 1].z + cp[i + 2].z)));

    float eps = maxWidth * .05f;  // width / 20 (TODO: set parameter for precision)

    // Compute log base 4 by dividing log2 in half.
    int r0 = Log2(1.41421356237f * 6.f * L0 / (8.f * eps)) / 2;
    int maxDepth = clamp(r0, 0, MaxCurveSubdivDepth);

    // fix subdivision level
    // int maxDepth = 4;

    float u0[MaxCurveSubdivDepth];
    float u1[MaxCurveSubdivDepth];
    float depths[MaxCurveSubdivDepth];

    float3 cpSub[4];
    int stack_ptr = -1;

    float ulo = 0;
    float uhi = 1;

    int depth = 0;

    float tHit = TCurrent;
    float uHit;
    float vHit;
    float3 dpduHit;

    while (depth <= MaxCurveSubdivDepth)
    {
        if (depth >= MaxCurveSubdivDepth)
        {
            cpSub[0] = BlossomBezier(cp, ulo, ulo, ulo);
            cpSub[1] = BlossomBezier(cp, ulo, ulo, uhi);
            cpSub[2] = BlossomBezier(cp, ulo, uhi, uhi);
            cpSub[3] = BlossomBezier(cp, uhi, uhi, uhi);

            float edge =
                    (cpSub[1].y - cpSub[0].y) * -cpSub[0].y + cpSub[0].x * (cpSub[0].x - cpSub[1].x);
            if (edge >= 0) {
            // Test sample point against tangent perpendicular at curve end
            edge = (cpSub[2].y - cpSub[3].y) * -cpSub[3].y + cpSub[3].x * (cpSub[3].x - cpSub[2].x);
            if (edge >= 0) {
            // Compute line $w$ that gives minimum distance to sample point
            float2 segmentDirection = cpSub[3].xy - cpSub[0].xy;
            float denom = dot(segmentDirection, segmentDirection);
            if (denom != 0) {
            float w = dot(-cpSub[0].xy, segmentDirection) / denom;

            // Compute $u$ coordinate of curve intersection point and _hitWidth_
            float u = clamp(lerp(ulo, uhi, w), ulo, uhi);

            // Test intersection point against curve width
            float3 dpcdw;
            float3 pc = EvalBezier(cpSub, clamp(w, 0, 1), dpcdw);
            float ptCurveDist2 = pc.x * pc.x + pc.y * pc.y;

            // out of ray's range
            if (ptCurveDist2 <= maxWidth * maxWidth * .25 && pc.z >= 0 && pc.z <= zMax)
            {
            // Compute $v$ coordinate of curve intersection point
            float ptCurveDist = sqrt(ptCurveDist2);
            float edgeFunc = dpcdw.x * -pc.y + pc.x * dpcdw.y;
            float v = (edgeFunc > 0) ? 0.5f + ptCurveDist / maxWidth : 0.5f - ptCurveDist / maxWidth;
            // Compute hit _t_ and partial derivatives for curve intersection
            float t = pc.z / rayLength;            
            if (t < tHit) {
                tHit = t;
                uHit = u;
                vHit = v;
                EvalBezier(cp, u, dpduHit);
            }}}}}            
        }
        else
        {
            float umi = (ulo + uhi) / 2;
            cpSub[0] = BlossomBezier(cp, ulo, ulo, ulo);
            cpSub[1] = BlossomBezier(cp, ulo, ulo, umi);
            cpSub[2] = BlossomBezier(cp, ulo, umi, umi);
            cpSub[3] = BlossomBezier(cp, umi, umi, umi);

            bool leftHasIntersection = true;

            if (max(max(cpSub[0].y, cpSub[1].y), max(cpSub[2].y, cpSub[3].y)) +
                        0.5f * maxWidth < 0 ||
                min(min(cpSub[0].y, cpSub[1].y), min(cpSub[2].y, cpSub[3].y)) -
                        0.5f * maxWidth > 0 ||
                max(max(cpSub[0].x, cpSub[1].x), max(cpSub[2].x, cpSub[3].x)) +
                        0.5f * maxWidth < 0 ||
                min(min(cpSub[0].x, cpSub[1].x), min(cpSub[2].x, cpSub[3].x)) -
                        0.5f * maxWidth > 0	||
                max(max(cpSub[0].z, cpSub[1].z), max(cpSub[2].z, cpSub[3].z)) +
                        0.5f * maxWidth < 0 ||
                min(min(cpSub[0].z, cpSub[1].z), min(cpSub[2].z, cpSub[3].z)) -
                        0.5f * maxWidth > zMax )
            {
                leftHasIntersection = false;
            }

            cpSub[0] = BlossomBezier(cp, umi, umi, umi);
            cpSub[1] = BlossomBezier(cp, umi, umi, uhi);
            cpSub[2] = BlossomBezier(cp, umi, uhi, uhi);
            cpSub[3] = BlossomBezier(cp, uhi, uhi, uhi);

            if (max(max(cpSub[0].y, cpSub[1].y), max(cpSub[2].y, cpSub[3].y)) +
                        0.5f * maxWidth < 0 ||
                min(min(cpSub[0].y, cpSub[1].y), min(cpSub[2].y, cpSub[3].y)) -
                        0.5f * maxWidth > 0 ||
                max(max(cpSub[0].x, cpSub[1].x), max(cpSub[2].x, cpSub[3].x)) +
                        0.5f * maxWidth < 0 ||
                min(min(cpSub[0].x, cpSub[1].x), min(cpSub[2].x, cpSub[3].x)) -
                        0.5f * maxWidth > 0	||
                max(max(cpSub[0].z, cpSub[1].z), max(cpSub[2].z, cpSub[3].z)) +
                        0.5f * maxWidth < 0 ||
                min(min(cpSub[0].z, cpSub[1].z), min(cpSub[2].z, cpSub[3].z)) -
                        0.5f * maxWidth > zMax )
            {
                if (leftHasIntersection)
                {
                    uhi = umi;
                    depth++;
                    continue;
                }
            }
            else
            {
                if (leftHasIntersection)
                {
                    stack_ptr++;
                    u0[stack_ptr] = umi;
                    u1[stack_ptr] = uhi;
                    depths[stack_ptr] = depth + 1;

                    uhi = umi;
                    depth++;

                    continue;
                }
                else
                {
                    ulo = umi;
                    depth++;
                    continue;
                }
            }
        }

        // pop stack

        if (stack_ptr == -1) break;
        ulo = u0[stack_ptr];
        uhi = u1[stack_ptr];
        depth = depths[stack_ptr];
        stack_ptr--;
    }

    if (tHit < TCurrent && tHit > 0)
    {
        CurveIntersectionAttributes attr;      
        attr.u = uHit;
        attr.v = vHit;
        attr.dpdu = 0.5*mul(dpduHit, raySpaceToWorldRotMatrix)+0.5;
        ReportHit(tHit, 0, attr);
    }
}


/** Closest curve hit shader for primary rays. 
*/

[shader("closesthit")]
void primaryCurveClosestHit(uniform HitShaderParams hitParams, inout PrimaryRayData rayData, CurveIntersectionAttributes attribs)
{   
    rayData.hit = true;
    uint2 launchIndex = DispatchRaysIndex().xy;
    //const uint meshInstanceID = geometryIndex;
    //const uint curveIndex = PrimitiveIndex();
    //PatchVertexData v = getCurveVertexData(triangleIndex, attribs);

    if (debugViewId == 0) gOutput[launchIndex] = float4(WorldRayOrigin() + WorldRayDirection() * RayTCurrent(), 1);
    else if (debugViewId == 1) gOutput[launchIndex] = float4(attribs.u, attribs.v, 0, 0);
    else if (debugViewId == 2) gOutput[launchIndex] = float4(attribs.dpdu, 0);
}


/** Closest mesh hit shader for primary rays. 
*/
[shader("closesthit")]
void primaryClosestHit(uniform HitShaderParams hitParams, inout PrimaryRayData rayData, BuiltInTriangleIntersectionAttributes attribs)
{
    //uint2 launchIndex = DispatchRaysIndex().xy;
    //gOutput[launchIndex] = float4(1,1,1,1);
}

/** ******************************** RayGen ******************************** */

// p = position on the image plane in [0,1] where (0,0) is top-left corner.
PrimaryRayData shootPrimaryRay(float2 p)
{
    // Compute primary ray's origin and direction.
    float2 ndc = float2(2, -2) * p + float2(-1, 1);
    float3 rayDir = ndc.x * gScene.camera.data.cameraU + ndc.y * gScene.camera.data.cameraV + gScene.camera.data.cameraW; // rayDir = world-space direction to point on image plane (unnormalized)
    float3 origin = gScene.camera.data.posW;

    rayDir = normalize(rayDir);

    // Setup ray
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = rayDir;
    ray.TMin = 0.f;     // It's OK to start at 0.0 for the camera, as it should not sit exactly on a surface
    ray.TMax = 1e+38f;
    // TODO: Discard hits outside [Z_near, Z_far] to match raster behavior.

    // Setup ray payload
    PrimaryRayData rayData;
    rayData.hit = false;

    TraceRay(gRtScene, 0, 0xff /* instanceInclusionMask */, 0 /* hitIdx */, hitProgramCount, 0 /* missIdx */, ray, rayData);

    return rayData;
}

[shader("raygeneration")]
void rayGen()
{
    uint2 launchIndex = DispatchRaysIndex().xy;
    uint2 launchDim = DispatchRaysDimensions().xy;

    // Compute sample position in screen space in [0,1] with origin at the top-left corner.
    // The camera jitter offsets the sample by +-0.5 pixels from the pixel center.
    float2 p = (launchIndex + float2(0.5f, 0.5f)) / launchDim + float2(-gScene.camera.data.jitterX, gScene.camera.data.jitterY);

    // Shoot primary ray. The hit shader writes the G-buffer.
    shootPrimaryRay(p);
}
